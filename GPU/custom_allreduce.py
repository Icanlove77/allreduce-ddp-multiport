"""
GPU version custom AllReduce algorithms for PyTorch DDP communication hooks.

"""
import math
import torch
import torch.distributed as dist


def _is_power_of(n: int, base: int) -> bool:
    if n < 1:
        return False
    while n % base == 0:
        n //= base
    return n == 1


def _log_power(n: int, base: int) -> int:
    """
    Return k such that n = base^k.
    Assumes n is a power of base.
    """
    k = 0
    while n > 1:
        n //= base
        k += 1
    return k


def _swing_rho(step: int) -> int:
    """
    Swing rho(s) = sum_{i=0}^{s} (-2)^i = (1 - (-2)^(s+1)) / 3.
    """
    return int((1 - ((-2) ** (step + 1))) // 3)


def _swing_peer(rank: int, step: int, world_size: int) -> int:
    rho = _swing_rho(step)
    if rank % 2 == 0:
        return (rank + rho) % world_size
    return (rank - rho) % world_size


def _trivance_peers(rank: int, step: int, world_size: int):
    distance = 3 ** step
    left_peer = (rank - distance) % world_size
    right_peer = (rank + distance) % world_size
    return left_peer, right_peer


def _bruck_peers(rank: int, step: int, world_size: int):
    """
    Outgoing peers:
        rank + 3^step
        rank + 2 * 3^step

    Incoming peers:
        rank - 3^step
        rank - 2 * 3^step
    """
    distance = 3 ** step

    out1 = (rank + distance) % world_size
    out2 = (rank + 2 * distance) % world_size

    in1 = (rank - distance) % world_size
    in2 = (rank - 2 * distance) % world_size

    return out1, out2, in1, in2


def _prepare_work_buffer(x: torch.Tensor, world_size: int):
    """
    Flatten x and pad it so that it can be split into world_size equal chunks.
    """
    original_shape = x.shape
    flat = x.contiguous().view(-1)
    original_numel = flat.numel()

    remainder = original_numel % world_size
    if remainder != 0:
        pad_numel = world_size - remainder
        padding = torch.zeros(
            pad_numel,
            dtype=flat.dtype,
            device=flat.device,
        )
        work = torch.cat([flat, padding], dim=0)
    else:
        work = flat

    if work.numel() % world_size != 0:
        raise RuntimeError("Internal padding failed.")

    chunks = list(work.chunk(world_size))
    return original_shape, flat, original_numel, work, chunks


def _finish_work_buffer(
    x: torch.Tensor,
    original_shape,
    flat: torch.Tensor,
    original_numel: int,
    work: torch.Tensor,
):
    """
    Remove padding if needed and copy the final result back to x.
    """
    if work.data_ptr() != flat.data_ptr():
        flat.copy_(work[:original_numel])

    x.copy_(flat.view(original_shape))
    return x


def _pack_chunks(chunks, indices):
    """
    Pack non-contiguous chunks into a contiguous tensor.

    This is functionally correct but not optimized. A high-performance
    implementation should avoid repeated torch.cat() and do block remapping.
    """
    if len(indices) == 0:
        return torch.empty(
            0,
            dtype=chunks[0].dtype,
            device=chunks[0].device,
        )

    return torch.cat([chunks[i].contiguous() for i in indices], dim=0)


def _recv_buffer_like(chunks, indices):
    """
    Allocate a receive buffer for len(indices) chunks.
    """
    if len(indices) == 0:
        return torch.empty(
            0,
            dtype=chunks[0].dtype,
            device=chunks[0].device,
        )

    return torch.empty(
        chunks[0].numel() * len(indices),
        dtype=chunks[0].dtype,
        device=chunks[0].device,
    )


def _add_packed_to_chunks(chunks, indices, packed):
    """
    Unpack a received packed buffer and add it into the corresponding chunks.
    Used in Reduce-Scatter.
    """
    if len(indices) == 0:
        return

    rows = packed.view(len(indices), chunks[0].numel())
    for row, idx in enumerate(indices):
        chunks[idx].add_(rows[row])


def _copy_packed_to_chunks(chunks, indices, packed):
    """
    Unpack a received packed buffer and copy it into the corresponding chunks.
    Used in AllGather.
    """
    if len(indices) == 0:
        return

    rows = packed.view(len(indices), chunks[0].numel())
    for row, idx in enumerate(indices):
        chunks[idx].copy_(rows[row])


def _prepare_even_work_buffer(x: torch.Tensor):
    """
    Flatten x and pad it to an even number of elements.

    This is used by half-vector folding. The extra padded value, if any,
    is zero and will be removed before copying the result back to x.
    """
    original_shape = x.shape
    flat = x.contiguous().view(-1)
    original_numel = flat.numel()

    if original_numel % 2 != 0:
        padding = torch.zeros(
            1,
            dtype=flat.dtype,
            device=flat.device,
        )
        work = torch.cat([flat, padding], dim=0)
    else:
        work = flat

    return original_shape, flat, original_numel, work


def _finish_even_work_buffer(
    x: torch.Tensor,
    original_shape,
    flat: torch.Tensor,
    original_numel: int,
    work: torch.Tensor,
):
    """
    Remove padding introduced by _prepare_even_work_buffer and copy result back.
    """
    if work.data_ptr() != flat.data_ptr():
        flat.copy_(work[:original_numel])

    x.copy_(flat.view(original_shape))
    return x


def _inner_recursive_doubling_latency_active_(
    work: torch.Tensor,
    active_ranks,
    local_rank: int,
):
    """
    Recursive Doubling latency version on a power-of-two active group.

    active_ranks maps local active rank -> global rank.
    """
    active_size = len(active_ranks)

    if not _is_power_of(active_size, 2):
        raise RuntimeError("active_size must be a power of two.")

    steps = _log_power(active_size, 2)
    recv_buf = torch.empty_like(work)

    for step in range(steps):
        peer_local = local_rank ^ (1 << step)
        peer_global = active_ranks[peer_local]

        send_buf = work.contiguous()

        ops = [
            dist.P2POp(dist.isend, send_buf, peer_global),
            dist.P2POp(dist.irecv, recv_buf, peer_global),
        ]

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        work.add_(recv_buf)

    return work


def _inner_swing_latency_active_(
    work: torch.Tensor,
    active_ranks,
    local_rank: int,
):
    """
    Swing latency version on a power-of-two active group.

    The Swing peer is computed in the local active-rank space and then
    mapped back to global rank IDs.
    """
    active_size = len(active_ranks)

    if not _is_power_of(active_size, 2):
        raise RuntimeError("active_size must be a power of two.")

    steps = _log_power(active_size, 2)
    recv_buf = torch.empty_like(work)

    for step in range(steps):
        peer_local = _swing_peer(local_rank, step, active_size)
        peer_global = active_ranks[peer_local]

        send_buf = work.contiguous()

        ops = [
            dist.P2POp(dist.isend, send_buf, peer_global),
            dist.P2POp(dist.irecv, recv_buf, peer_global),
        ]

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        work.add_(recv_buf)

    return work


def _inner_recursive_doubling_bandwidth_active_(
    x: torch.Tensor,
    active_ranks,
    local_rank: int,
):
    """
    Recursive Doubling bandwidth version on a power-of-two active group.

    This is the same reduce-scatter + allgather logic as the original
    power-of-two version, but peer IDs are mapped through active_ranks.
    """
    active_size = len(active_ranks)

    if not _is_power_of(active_size, 2):
        raise RuntimeError("active_size must be a power of two.")

    steps = _log_power(active_size, 2)

    original_shape, flat, original_numel, work, chunks = _prepare_work_buffer(
        x, active_size
    )

    active = list(range(active_size))

    # Phase 1: Reduce-Scatter
    for step in range(steps):
        bit = 1 << step
        peer_local = local_rank ^ bit
        peer_global = active_ranks[peer_local]

        keep_indices = [
            idx for idx in active
            if ((idx >> step) & 1) == ((local_rank >> step) & 1)
        ]
        send_indices = [
            idx for idx in active
            if ((idx >> step) & 1) != ((local_rank >> step) & 1)
        ]

        send_buf = _pack_chunks(chunks, send_indices)
        recv_buf = _recv_buffer_like(chunks, keep_indices)

        ops = [
            dist.P2POp(dist.isend, send_buf, peer_global),
            dist.P2POp(dist.irecv, recv_buf, peer_global),
        ]

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        _add_packed_to_chunks(chunks, keep_indices, recv_buf)
        active = sorted(keep_indices)

    # Phase 2: AllGather
    for step in reversed(range(steps)):
        bit = 1 << step
        peer_local = local_rank ^ bit
        peer_global = active_ranks[peer_local]

        send_indices = sorted(active)
        recv_indices = sorted([idx ^ bit for idx in active])

        send_buf = _pack_chunks(chunks, send_indices)
        recv_buf = _recv_buffer_like(chunks, recv_indices)

        ops = [
            dist.P2POp(dist.isend, send_buf, peer_global),
            dist.P2POp(dist.irecv, recv_buf, peer_global),
        ]

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        _copy_packed_to_chunks(chunks, recv_indices, recv_buf)
        active = sorted(set(active).union(recv_indices))

    return _finish_work_buffer(x, original_shape, flat, original_numel, work)


def _inner_swing_bandwidth_active_(
    x: torch.Tensor,
    active_ranks,
    local_rank: int,
):
    """
    Swing bandwidth version on a power-of-two active group.

    The block indices are computed in local active-rank space.
    P2P peers are mapped back to global rank IDs through active_ranks.
    """
    active_size = len(active_ranks)

    if not _is_power_of(active_size, 2):
        raise RuntimeError("active_size must be a power of two.")

    steps = _log_power(active_size, 2)

    original_shape, flat, original_numel, work, chunks = _prepare_work_buffer(
        x, active_size
    )

    # Phase 1: Reduce-Scatter
    for step in range(steps):
        peer_local = _swing_peer(local_rank, step, active_size)
        peer_global = active_ranks[peer_local]

        send_indices = _swing_subtree_indices(
            root=peer_local,
            next_step=step + 1,
            world_size=active_size,
            total_steps=steps,
        )
        recv_indices = _swing_subtree_indices(
            root=local_rank,
            next_step=step + 1,
            world_size=active_size,
            total_steps=steps,
        )

        send_buf = _pack_chunks(chunks, send_indices)
        recv_buf = _recv_buffer_like(chunks, recv_indices)

        ops = [
            dist.P2POp(dist.isend, send_buf, peer_global),
            dist.P2POp(dist.irecv, recv_buf, peer_global),
        ]

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        _add_packed_to_chunks(chunks, recv_indices, recv_buf)

    # Phase 2: AllGather
    for step in reversed(range(steps)):
        peer_local = _swing_peer(local_rank, step, active_size)
        peer_global = active_ranks[peer_local]

        send_indices = _swing_subtree_indices(
            root=local_rank,
            next_step=step + 1,
            world_size=active_size,
            total_steps=steps,
        )
        recv_indices = _swing_subtree_indices(
            root=peer_local,
            next_step=step + 1,
            world_size=active_size,
            total_steps=steps,
        )

        send_buf = _pack_chunks(chunks, send_indices)
        recv_buf = _recv_buffer_like(chunks, recv_indices)

        ops = [
            dist.P2POp(dist.isend, send_buf, peer_global),
            dist.P2POp(dist.irecv, recv_buf, peer_global),
        ]

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        _copy_packed_to_chunks(chunks, recv_indices, recv_buf)

    return _finish_work_buffer(x, original_shape, flat, original_numel, work)


def _swing_subtree_indices(
    root: int,
    next_step: int,
    world_size: int,
    total_steps: int,
):
    """
    Return the block indices in the Swing subtree rooted at `root`.

    In Swing-B, when rank r sends to peer q at step s, it sends:
        q's block + all blocks that q will reach/forward in later steps.
    """
    out = {root}

    def rec(r: int, step: int):
        if step >= total_steps:
            return

        for s in range(step, total_steps):
            peer = _swing_peer(r, s, world_size)
            out.add(peer)
            rec(peer, s + 1)

    rec(root, next_step)
    return sorted(out)


def _trivance_subtree_indices(
    root: int,
    next_step: int,
    world_size: int,
    total_steps: int,
):
    """
    Return the block indices in the Trivance subtree rooted at `root`.

    In Trivance-B, when rank r sends to peer p at step k, it sends:
        p's block + all blocks p will reach through both directions
        in subsequent steps.
    """
    out = {root}

    def rec(r: int, step: int):
        if step >= total_steps:
            return

        for s in range(step, total_steps):
            left_peer, right_peer = _trivance_peers(r, s, world_size)

            out.add(left_peer)
            out.add(right_peer)

            rec(left_peer, s + 1)
            rec(right_peer, s + 1)

    rec(root, next_step)
    return sorted(out)


def _bruck_subtree_indices(
    root: int,
    next_step: int,
    world_size: int,
    total_steps: int,
):
    """
    Return the block indices in the Bruck-style subtree rooted at `root`.

    Bruck-style pattern uses two outgoing peers per step:
        r + 3^k
        r + 2 * 3^k

    In Bruck-B, when rank r sends to peer p at step k, it sends:
        p's block + all blocks p will reach through later Bruck steps.
    """
    out = {root}

    def rec(r: int, step: int):
        if step >= total_steps:
            return

        for s in range(step, total_steps):
            out1, out2, _, _ = _bruck_peers(r, s, world_size)

            out.add(out1)
            out.add(out2)

            rec(out1, s + 1)
            rec(out2, s + 1)

    rec(root, next_step)
    return sorted(out)


def builtin_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Built-in all-reduce SUM.
    This is used as a baseline.
    """
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def ring_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Bandwidth-optimal Ring AllReduce:
        Reduce-Scatter + AllGather.

    If x.numel() is not divisible by world_size, we pad internally
    and copy the valid part back to x.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    original_shape, flat, original_numel, work, chunks = _prepare_work_buffer(
        x, world_size
    )

    right = (rank + 1) % world_size
    left = (rank - 1 + world_size) % world_size

    recv_buf = torch.empty_like(chunks[0])

    # Phase 1: Reduce-Scatter
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1 + world_size) % world_size

        send_buf = chunks[send_idx].contiguous()

        ops = [
            dist.P2POp(dist.isend, send_buf, right),
            dist.P2POp(dist.irecv, recv_buf, left),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        chunks[recv_idx].add_(recv_buf)

    # Phase 2: AllGather
    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size

        send_buf = chunks[send_idx].contiguous()
        recv_target = chunks[recv_idx]

        ops = [
            dist.P2POp(dist.isend, send_buf, right),
            dist.P2POp(dist.irecv, recv_target, left),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return _finish_work_buffer(x, original_shape, flat, original_numel, work)


def bidirectional_ring_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Bidirectional Ring AllReduce for odd ring sizes n = 2d + 1.

    Reduce-Scatter step k sends:
        block (rank - d + k) mod n to the left neighbor
        block (rank + d - k) mod n to the right neighbor

    AllGather step k sends:
        block (rank + k) mod n to the left neighbor
        block (rank - k) mod n to the right neighbor

    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if world_size % 2 == 0:
        raise ValueError(
            "Bidirectional Ring requires odd world_size n=2d+1. "
            "On an 8-GPU RunPod, use --nproc_per_node=7 for this algorithm."
        )

    d = (world_size - 1) // 2

    original_shape, flat, original_numel, work, chunks = _prepare_work_buffer(
        x, world_size
    )

    left = (rank - 1 + world_size) % world_size
    right = (rank + 1) % world_size

    # Phase 1: Bidirectional Reduce-Scatter.
    for step in range(d):
        send_left_idx = (rank - d + step) % world_size
        send_right_idx = (rank + d - step) % world_size

        recv_from_right_idx = (rank - d + step + 1) % world_size
        recv_from_left_idx = (rank + d - step - 1) % world_size

        send_left = chunks[send_left_idx].contiguous()
        send_right = chunks[send_right_idx].contiguous()
        recv_from_right = torch.empty_like(chunks[0])
        recv_from_left = torch.empty_like(chunks[0])

        ops = [
            dist.P2POp(dist.isend, send_left, left),
            dist.P2POp(dist.irecv, recv_from_right, right),
            dist.P2POp(dist.isend, send_right, right),
            dist.P2POp(dist.irecv, recv_from_left, left),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        chunks[recv_from_right_idx].add_(recv_from_right)
        chunks[recv_from_left_idx].add_(recv_from_left)

    # Phase 2: Bidirectional AllGather.
    for step in range(d):
        send_left_idx = (rank + step) % world_size
        send_right_idx = (rank - step) % world_size

        recv_from_right_idx = (rank + step + 1) % world_size
        recv_from_left_idx = (rank - step - 1) % world_size

        send_left = chunks[send_left_idx].contiguous()
        send_right = chunks[send_right_idx].contiguous()

        ops = [
            dist.P2POp(dist.isend, send_left, left),
            dist.P2POp(dist.irecv, chunks[recv_from_right_idx], right),
            dist.P2POp(dist.isend, send_right, right),
            dist.P2POp(dist.irecv, chunks[recv_from_left_idx], left),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return _finish_work_buffer(x, original_shape, flat, original_numel, work)


def recursive_doubling_latency_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Recursive Doubling latency AllReduce.

    GPU version requirement:
    - world_size must be a power of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if not _is_power_of(world_size, 2):
        raise ValueError(
            f"Recursive Doubling latency version requires world_size=2^k, "
            f"got world_size={world_size}."
        )

    active_ranks = list(range(world_size))
    work = x.contiguous().view(-1)
    _inner_recursive_doubling_latency_active_(work, active_ranks, rank)
    x.copy_(work.view(x.shape))
    return x

def recursive_doubling_bandwidth_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Recursive Doubling bandwidth AllReduce.

    GPU version requirement:
    - world_size must be a power of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if not _is_power_of(world_size, 2):
        raise ValueError(
            f"Recursive Doubling bandwidth version requires world_size=2^k, "
            f"got world_size={world_size}."
        )

    active_ranks = list(range(world_size))
    return _inner_recursive_doubling_bandwidth_active_(x, active_ranks, rank)

def swing_latency_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Swing latency AllReduce.

    world_size must be a power of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if not _is_power_of(world_size, 2):
        raise ValueError(
            f"Swing latency version requires world_size=2^k, "
            f"got world_size={world_size}."
        )

    active_ranks = list(range(world_size))
    work = x.contiguous().view(-1)
    _inner_swing_latency_active_(work, active_ranks, rank)
    x.copy_(work.view(x.shape))
    return x

def swing_bandwidth_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Swing bandwidth AllReduce.

    world_size must be a power of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if not _is_power_of(world_size, 2):
        raise ValueError(
            f"Swing bandwidth version requires world_size=2^k, "
            f"got world_size={world_size}."
        )

    active_ranks = list(range(world_size))
    return _inner_swing_bandwidth_active_(x, active_ranks, rank)

def bruck_latency_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Latency-optimal Bruck AllReduce.

    world_size must be a power of 3.

    Logic:
    - At step k, rank r sends the whole buffer to:
        r + 3^k
        r + 2 * 3^k
    - It receives from:
        r - 3^k
        r - 2 * 3^k
    - Then it adds both received buffers.
    - Total steps: log3(world_size).
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if not _is_power_of(world_size, 3):
        raise ValueError(
            f"Bruck latency version requires world_size=3^k, "
            f"got world_size={world_size}."
        )

    steps = _log_power(world_size, 3)

    original_shape = x.shape
    work = x.contiguous().view(-1)

    recv_a = torch.empty_like(work)
    recv_b = torch.empty_like(work)

    for step in range(steps):
        out1, out2, in1, in2 = _bruck_peers(rank, step, world_size)

        if out1 == out2 or in1 == in2:
            raise RuntimeError("Invalid Bruck peer selection.")

        send_a = work.contiguous()
        send_b = work.contiguous()

        ops = [
            dist.P2POp(dist.isend, send_a, out1),
            dist.P2POp(dist.irecv, recv_a, in1),
            dist.P2POp(dist.isend, send_b, out2),
            dist.P2POp(dist.irecv, recv_b, in2),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        work.add_(recv_a)
        work.add_(recv_b)

    x.copy_(work.view(original_shape))
    return x


def bruck_bandwidth_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Bandwidth-optimal Bruck AllReduce.

    world_size must be a power of 3.

    Bruck-style pattern:
    - Reduce-scatter step k:
        send block subtree to r + 3^k and r + 2*3^k;
        receive local subtree from r - 3^k and r - 2*3^k.
    - AllGather reverses this direction.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if not _is_power_of(world_size, 3):
        raise ValueError(
            f"Bruck bandwidth version requires world_size=3^k, "
            f"got world_size={world_size}."
        )

    steps = _log_power(world_size, 3)

    original_shape, flat, original_numel, work, chunks = _prepare_work_buffer(
        x, world_size
    )

    # Phase 1: Reduce-Scatter.
    for step in range(steps):
        out1, out2, in1, in2 = _bruck_peers(rank, step, world_size)

        local_indices = _bruck_subtree_indices(
            root=rank,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )
        send_out1_indices = _bruck_subtree_indices(
            root=out1,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )
        send_out2_indices = _bruck_subtree_indices(
            root=out2,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )

        send_out1 = _pack_chunks(chunks, send_out1_indices)
        send_out2 = _pack_chunks(chunks, send_out2_indices)

        recv_in1 = _recv_buffer_like(chunks, local_indices)
        recv_in2 = _recv_buffer_like(chunks, local_indices)

        ops = [
            dist.P2POp(dist.isend, send_out1, out1),
            dist.P2POp(dist.irecv, recv_in1, in1),
            dist.P2POp(dist.isend, send_out2, out2),
            dist.P2POp(dist.irecv, recv_in2, in2),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        _add_packed_to_chunks(chunks, local_indices, recv_in1)
        _add_packed_to_chunks(chunks, local_indices, recv_in2)

    # Phase 2: AllGather.
    # Reverse of the RS edges:
    # send local subtree back to the incoming sources,
    # receive outgoing subtrees back from the outgoing peers.
    for step in reversed(range(steps)):
        out1, out2, in1, in2 = _bruck_peers(rank, step, world_size)

        local_indices = _bruck_subtree_indices(
            root=rank,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )
        recv_out1_indices = _bruck_subtree_indices(
            root=out1,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )
        recv_out2_indices = _bruck_subtree_indices(
            root=out2,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )

        send_to_in1 = _pack_chunks(chunks, local_indices)
        send_to_in2 = _pack_chunks(chunks, local_indices)

        recv_from_out1 = _recv_buffer_like(chunks, recv_out1_indices)
        recv_from_out2 = _recv_buffer_like(chunks, recv_out2_indices)

        ops = [
            dist.P2POp(dist.isend, send_to_in1, in1),
            dist.P2POp(dist.irecv, recv_from_out1, out1),
            dist.P2POp(dist.isend, send_to_in2, in2),
            dist.P2POp(dist.irecv, recv_from_out2, out2),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        _copy_packed_to_chunks(chunks, recv_out1_indices, recv_from_out1)
        _copy_packed_to_chunks(chunks, recv_out2_indices, recv_from_out2)

    return _finish_work_buffer(x, original_shape, flat, original_numel, work)


def trivance_latency_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Latency-optimal Trivance AllReduce.

    Requirement:
    - world_size must be a power of 3.

    Logic:
    - At step k, rank r communicates with:
        left  = r - 3^k mod n
        right = r + 3^k mod n
    - Each node uses both directions simultaneously.
    - This latency version exchanges the entire vector at each step.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if not _is_power_of(world_size, 3):
        raise ValueError(
            f"Trivance latency version requires world_size=3^k, "
            f"got world_size={world_size}."
        )

    steps = _log_power(world_size, 3)

    original_shape = x.shape
    work = x.contiguous().view(-1)

    recv_left = torch.empty_like(work)
    recv_right = torch.empty_like(work)

    for step in range(steps):
        left_peer, right_peer = _trivance_peers(rank, step, world_size)

        if left_peer == right_peer:
            raise RuntimeError("Invalid Trivance peer selection.")

        send_left = work.contiguous()
        send_right = work.contiguous()

        ops = [
            dist.P2POp(dist.isend, send_left, left_peer),
            dist.P2POp(dist.irecv, recv_left, left_peer),
            dist.P2POp(dist.isend, send_right, right_peer),
            dist.P2POp(dist.irecv, recv_right, right_peer),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        work.add_(recv_left)
        work.add_(recv_right)

    x.copy_(work.view(original_shape))
    return x


def trivance_bandwidth_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Bandwidth-optimal Trivance AllReduce.

    Requirement:
    - world_size must be a power of 3.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if not _is_power_of(world_size, 3):
        raise ValueError(
            f"Trivance bandwidth version requires world_size=3^k, "
            f"got world_size={world_size}."
        )

    steps = _log_power(world_size, 3)

    original_shape, flat, original_numel, work, chunks = _prepare_work_buffer(
        x, world_size
    )

    # Phase 1: Reduce-Scatter.
    for step in range(steps):
        left_peer, right_peer = _trivance_peers(rank, step, world_size)

        if left_peer == right_peer:
            raise RuntimeError("Invalid Trivance peer selection.")

        local_indices = _trivance_subtree_indices(
            root=rank,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )
        send_left_indices = _trivance_subtree_indices(
            root=left_peer,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )
        send_right_indices = _trivance_subtree_indices(
            root=right_peer,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )

        send_left = _pack_chunks(chunks, send_left_indices)
        send_right = _pack_chunks(chunks, send_right_indices)

        recv_left = _recv_buffer_like(chunks, local_indices)
        recv_right = _recv_buffer_like(chunks, local_indices)

        ops = [
            dist.P2POp(dist.isend, send_left, left_peer),
            dist.P2POp(dist.irecv, recv_left, left_peer),
            dist.P2POp(dist.isend, send_right, right_peer),
            dist.P2POp(dist.irecv, recv_right, right_peer),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        _add_packed_to_chunks(chunks, local_indices, recv_left)
        _add_packed_to_chunks(chunks, local_indices, recv_right)

    # Phase 2: AllGather.
    for step in reversed(range(steps)):
        left_peer, right_peer = _trivance_peers(rank, step, world_size)

        local_indices = _trivance_subtree_indices(
            root=rank,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )
        recv_left_indices = _trivance_subtree_indices(
            root=left_peer,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )
        recv_right_indices = _trivance_subtree_indices(
            root=right_peer,
            next_step=step + 1,
            world_size=world_size,
            total_steps=steps,
        )

        send_left = _pack_chunks(chunks, local_indices)
        send_right = _pack_chunks(chunks, local_indices)

        recv_left = _recv_buffer_like(chunks, recv_left_indices)
        recv_right = _recv_buffer_like(chunks, recv_right_indices)

        ops = [
            dist.P2POp(dist.isend, send_left, left_peer),
            dist.P2POp(dist.irecv, recv_left, left_peer),
            dist.P2POp(dist.isend, send_right, right_peer),
            dist.P2POp(dist.irecv, recv_right, right_peer),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        _copy_packed_to_chunks(chunks, recv_left_indices, recv_left)
        _copy_packed_to_chunks(chunks, recv_right_indices, recv_right)

    return _finish_work_buffer(x, original_shape, flat, original_numel, work)


def allreduce_sum_(x: torch.Tensor, algo: str) -> torch.Tensor:
    if algo == "builtin":
        return builtin_allreduce_sum_(x)
    elif algo == "ring":
        return ring_allreduce_sum_(x)
    elif algo in ("bidirectional-ring", "bidir-ring", "bi-ring"):
        return bidirectional_ring_allreduce_sum_(x)
    elif algo == "recursive-doubling-latency":
        return recursive_doubling_latency_allreduce_sum_(x)
    elif algo == "recursive-doubling-bandwidth":
        return recursive_doubling_bandwidth_allreduce_sum_(x)
    elif algo == "swing-latency":
        return swing_latency_allreduce_sum_(x)
    elif algo == "swing-bandwidth":
        return swing_bandwidth_allreduce_sum_(x)
    elif algo == "bruck-latency":
        return bruck_latency_allreduce_sum_(x)
    elif algo == "bruck-bandwidth":
        return bruck_bandwidth_allreduce_sum_(x)
    elif algo == "trivance-latency":
        return trivance_latency_allreduce_sum_(x)
    elif algo == "trivance-bandwidth":
        return trivance_bandwidth_allreduce_sum_(x)
    else:
        raise ValueError(f"Unknown all-reduce algorithm: {algo}")
