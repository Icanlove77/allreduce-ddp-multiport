import math
import threading
import torch
import torch.distributed as dist


_BIDIRECTIONAL_RING_GROUPS = {}


def _get_bidirectional_ring_groups(world_size: int):
    groups = _BIDIRECTIONAL_RING_GROUPS.get(world_size)
    if groups is None:
        ranks = list(range(world_size))
        groups = (
            dist.new_group(ranks=ranks),
            dist.new_group(ranks=ranks),
        )
        _BIDIRECTIONAL_RING_GROUPS[world_size] = groups
    return groups


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

def _largest_power_of_two_leq(n: int) -> int:
    """
    Return the largest power of two <= n.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    return 1 << (n.bit_length() - 1)


def _power2_active_info(rank: int, world_size: int):
    """
    Non-power-of-two folding information.

    Let p2 = largest power of two <= world_size
    and r = world_size - p2.

    The first 2r ranks are paired:
        even ranks: active
        odd ranks: eliminated temporarily

    Remaining ranks [2r, world_size) are active.

    Example for world_size=9:
        p2 = 8, r = 1
        rank 0 active, rank 1 eliminated
        active ranks = [0, 2, 3, 4, 5, 6, 7, 8]
    """
    p2 = _largest_power_of_two_leq(world_size)
    r = world_size - p2

    if r == 0:
        return {
            "p2": p2,
            "r": r,
            "is_power_of_two": True,
            "is_active": True,
            "is_eliminated": False,
            "partner": None,
            "active_ranks": list(range(world_size)),
            "local_rank": rank,
        }

    active_ranks = list(range(0, 2 * r, 2)) + list(range(2 * r, world_size))

    if rank < 2 * r:
        if rank % 2 == 0:
            local_rank = rank // 2
            return {
                "p2": p2,
                "r": r,
                "is_power_of_two": False,
                "is_active": True,
                "is_eliminated": False,
                "partner": rank + 1,
                "active_ranks": active_ranks,
                "local_rank": local_rank,
            }
        else:
            return {
                "p2": p2,
                "r": r,
                "is_power_of_two": False,
                "is_active": False,
                "is_eliminated": True,
                "partner": rank - 1,
                "active_ranks": active_ranks,
                "local_rank": None,
            }

    local_rank = rank - r
    return {
        "p2": p2,
        "r": r,
        "is_power_of_two": False,
        "is_active": True,
        "is_eliminated": False,
        "partner": None,
        "active_ranks": active_ranks,
        "local_rank": local_rank,
    }


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


def _fold_full_buffer_then_run_active_(
    x: torch.Tensor,
    info,
    inner_func,
):
    """
    Non-power-of-two latency-style folding.

    Eliminated odd ranks send the full buffer to their even partner.
    Active ranks run a power-of-two algorithm.
    The partner sends the final result back to the eliminated rank.
    """
    original_shape = x.shape
    work = x.contiguous().view(-1)

    if info["is_power_of_two"]:
        inner_func(work, info["active_ranks"], info["local_rank"])
        x.copy_(work.view(original_shape))
        return x

    partner = info["partner"]

    if info["is_eliminated"]:
        send_buf = work.contiguous()

        send_req = dist.isend(send_buf, partner)
        send_req.wait()

        final_buf = torch.empty_like(work)
        recv_req = dist.irecv(final_buf, partner)
        recv_req.wait()

        work.copy_(final_buf)
        x.copy_(work.view(original_shape))
        return x

    # Active paired rank receives and reduces the eliminated partner.
    if partner is not None:
        recv_buf = torch.empty_like(work)
        recv_req = dist.irecv(recv_buf, partner)
        recv_req.wait()
        work.add_(recv_buf)

    # Active power-of-two algorithm.
    inner_func(work, info["active_ranks"], info["local_rank"])

    # Send final result back to eliminated partner.
    if partner is not None:
        send_buf = work.contiguous()
        send_req = dist.isend(send_buf, partner)
        send_req.wait()

    x.copy_(work.view(original_shape))
    return x


def _fold_half_vector_then_run_active_(
    x: torch.Tensor,
    info,
    inner_func,
):
    """
    Non-power-of-two bandwidth-style half-vector folding.

    For each extra pair:
        even active rank sends second half to odd eliminated rank;
        odd eliminated rank sends first half to even active rank;
        both reduce the half they receive;
        odd sends the reduced second half back to even;
        even now has the full reduced vector for both ranks.

    Then active ranks run a power-of-two bandwidth algorithm.
    Finally, the even active rank sends the final result back to its
    eliminated odd partner.
    """
    original_shape, flat, original_numel, work = _prepare_even_work_buffer(x)

    if info["is_power_of_two"]:
        inner_func(work, info["active_ranks"], info["local_rank"])
        return _finish_even_work_buffer(
            x, original_shape, flat, original_numel, work
        )

    partner = info["partner"]
    half = work.numel() // 2

    first_half = work[:half]
    second_half = work[half:]

    if info["is_eliminated"]:
        # Odd eliminated rank:
        # send first half to active even partner;
        # receive second half from active even partner;
        # reduce second half locally;
        # send reduced second half back;
        # wait for final full result.
        send_first = first_half.contiguous()
        recv_second = torch.empty_like(second_half)

        ops = [
            dist.P2POp(dist.isend, send_first, partner),
            dist.P2POp(dist.irecv, recv_second, partner),
        ]

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        second_half.add_(recv_second)

        send_reduced_second = second_half.contiguous()
        send_req = dist.isend(send_reduced_second, partner)
        send_req.wait()

        final_buf = torch.empty_like(work)
        recv_req = dist.irecv(final_buf, partner)
        recv_req.wait()

        work.copy_(final_buf)

        return _finish_even_work_buffer(
            x, original_shape, flat, original_numel, work
        )

    # Active paired even rank.
    if partner is not None:
        # send second half to eliminated odd partner;
        # receive first half from eliminated odd partner;
        # reduce first half locally;
        # receive reduced second half back.
        send_second = second_half.contiguous()
        recv_first = torch.empty_like(first_half)

        ops = [
            dist.P2POp(dist.isend, send_second, partner),
            dist.P2POp(dist.irecv, recv_first, partner),
        ]

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        first_half.add_(recv_first)

        recv_reduced_second = torch.empty_like(second_half)
        recv_req = dist.irecv(recv_reduced_second, partner)
        recv_req.wait()

        second_half.copy_(recv_reduced_second)

    # Active power-of-two algorithm.
    inner_func(work, info["active_ranks"], info["local_rank"])

    # Send final result back to eliminated odd partner.
    if partner is not None:
        send_buf = work.contiguous()
        send_req = dist.isend(send_buf, partner)
        send_req.wait()

    return _finish_even_work_buffer(
        x, original_shape, flat, original_numel, work
    )

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

    The two directions are independent pipelines. 
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if world_size % 2 == 0:
        raise ValueError(
            f"Bidirectional Ring requires odd world_size n=2d+1, got world_size={world_size}."
        )

    d = (world_size - 1) // 2

    original_shape, flat, original_numel, work, chunks = _prepare_work_buffer(
        x, world_size
    )

    left = (rank - 1 + world_size) % world_size
    right = (rank + 1) % world_size
    left_group, right_group = _get_bidirectional_ring_groups(world_size)

    def run_workers(*workers):
        errors = []

        def wrapped(worker):
            try:
                worker()
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=wrapped, args=(worker,)) for worker in workers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

    final_reduce_parts = {"from_right": None, "from_left": None}

    def reduce_scatter_left_direction():
        # Partial reductions move from right to left.
        for step in range(d):
            send_idx = (rank - d + step) % world_size
            recv_idx = (rank - d + step + 1) % world_size

            send_buf = chunks[send_idx].contiguous()
            recv_buf = torch.empty_like(chunks[0])

            ops = [
                dist.P2POp(dist.isend, send_buf, left, group=left_group),
                dist.P2POp(dist.irecv, recv_buf, right, group=left_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            if step == d - 1:
                final_reduce_parts["from_right"] = recv_buf
            else:
                chunks[recv_idx].add_(recv_buf)

    def reduce_scatter_right_direction():
        # Partial reductions move from left to right.
        for step in range(d):
            send_idx = (rank + d - step) % world_size
            recv_idx = (rank + d - step - 1) % world_size

            send_buf = chunks[send_idx].contiguous()
            recv_buf = torch.empty_like(chunks[0])

            ops = [
                dist.P2POp(dist.isend, send_buf, right, group=right_group),
                dist.P2POp(dist.irecv, recv_buf, left, group=right_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            if step == d - 1:
                final_reduce_parts["from_left"] = recv_buf
            else:
                chunks[recv_idx].add_(recv_buf)

    run_workers(reduce_scatter_left_direction, reduce_scatter_right_direction)

    if final_reduce_parts["from_right"] is None or final_reduce_parts["from_left"] is None:
        raise RuntimeError("Bidirectional ring Reduce-Scatter did not receive both final partial reductions.")

    chunks[rank].add_(final_reduce_parts["from_right"])
    chunks[rank].add_(final_reduce_parts["from_left"])

    def allgather_left_direction():
        # Reduced blocks move from right to left.
        for step in range(d):
            send_idx = (rank + step) % world_size
            recv_idx = (rank + step + 1) % world_size

            send_buf = chunks[send_idx].contiguous()
            ops = [
                dist.P2POp(dist.isend, send_buf, left, group=left_group),
                dist.P2POp(dist.irecv, chunks[recv_idx], right, group=left_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

    def allgather_right_direction():
        # Reduced blocks move from left to right.
        for step in range(d):
            send_idx = (rank - step) % world_size
            recv_idx = (rank - step - 1) % world_size

            send_buf = chunks[send_idx].contiguous()
            ops = [
                dist.P2POp(dist.isend, send_buf, right, group=right_group),
                dist.P2POp(dist.irecv, chunks[recv_idx], left, group=right_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

    run_workers(allgather_left_direction, allgather_right_direction)

    return _finish_work_buffer(x, original_shape, flat, original_numel, work)


def recursive_doubling_latency_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Recursive Doubling latency AllReduce.

    Supports:
    - power-of-two world_size: normal recursive doubling.
    - non-power-of-two world_size: full-buffer folding to the nearest
      lower power-of-two active group.

    For world_size=9:
        rank 1 is folded into rank 0;
        active ranks [0,2,3,4,5,6,7,8] run 8-rank RD;
        rank 0 sends the final result back to rank 1.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    info = _power2_active_info(rank, world_size)

    return _fold_full_buffer_then_run_active_(
        x,
        info,
        _inner_recursive_doubling_latency_active_,
    )


def recursive_doubling_bandwidth_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Recursive Doubling / Rabenseifner-style bandwidth AllReduce.

    Supports:
    - power-of-two world_size: normal reduce-scatter + allgather.
    - non-power-of-two world_size: half-vector folding to the nearest
      lower power-of-two active group.

    For world_size=9:
        rank 0 and rank 1 first exchange half vectors;
        rank 1 is temporarily eliminated;
        active ranks [0,2,3,4,5,6,7,8] run 8-rank bandwidth RD;
        rank 0 sends the final result back to rank 1.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    info = _power2_active_info(rank, world_size)

    return _fold_half_vector_then_run_active_(
        x,
        info,
        _inner_recursive_doubling_bandwidth_active_,
    )


def swing_latency_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Swing latency AllReduce.

    Supports:
    - power-of-two world_size: normal Swing.
    - non-power-of-two world_size: full-buffer folding to the nearest
      lower power-of-two active group.

    Note:
    This is a functional non-power-of-two adaptation. For world_size=9,
    it folds one extra rank and runs the original Swing schedule on 8
    active ranks.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    info = _power2_active_info(rank, world_size)

    return _fold_full_buffer_then_run_active_(
        x,
        info,
        _inner_swing_latency_active_,
    )


def swing_bandwidth_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Swing bandwidth AllReduce.

    Supports:
    - power-of-two world_size: normal Swing-B.
    - non-power-of-two world_size: half-vector folding to the nearest
      lower power-of-two active group, then Swing-B on active ranks.

    For world_size=9:
        rank 0 and rank 1 first exchange half vectors;
        rank 1 is temporarily eliminated;
        active ranks [0,2,3,4,5,6,7,8] run 8-rank Swing-B;
        rank 0 sends the final result back to rank 1.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    info = _power2_active_info(rank, world_size)

    return _fold_half_vector_then_run_active_(
        x,
        info,
        _inner_swing_bandwidth_active_,
    )


def bruck_latency_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Latency-optimal Bruck AllReduce.

    Requirement:
    - world_size must be a power of 3.

    Logic:
    - At step k, rank r sends the whole buffer to:
        r + 3^k
        r + 2 * 3^k
    - It receives from:
        r - 3^k
        r - 2 * 3^k
    - Then it adds both received buffers.
    - Total steps: log3(world_size).

    This follows the Bruck-style 2-port expansion pattern used as
    an AllReduce baseline in Trivance.
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

    Requirement:
    - world_size must be a power of 3.

    Bruck-style pattern:
    - Reduce-scatter step k:
        send block subtree to r + 3^k and r + 2*3^k;
        receive local subtree from r - 3^k and r - 2*3^k.
    - AllGather reverses this direction.

    Note:
    - This is a correctness implementation.
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

    Structure:
    - Reduce-Scatter with bidirectional Trivance communication.
    - AllGather in reverse order.

    Important:
    - This implementation is functionally correct but not performance optimized.
    - It packs non-contiguous block sets using torch.cat().
    - At each Reduce-Scatter step, both incoming messages target the same
      local subtree and must both be reduced before the next step.
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

# ============================================================================
# Special 16-rank Trivance adaptations
# ============================================================================

_TRIVANCE_N16_DISTANCES = [1, 3, 4]
_TRIVANCE_N16_BANDWIDTH_SCHEDULE = [{(0, 1): [2, 3, 10, 11, 16, 20, 21, 28, 29],
  (0, 15): [4, 5, 12, 13, 17, 22, 23, 30, 31],
  (1, 0): [0, 1, 6, 7, 14, 15, 19, 24, 25],
  (1, 2): [4, 5, 12, 13, 18, 22, 23, 30, 31],
  (2, 1): [2, 3, 8, 9, 16, 17, 21, 26, 27],
  (2, 3): [0, 1, 6, 7, 14, 15, 20, 24, 25],
  (3, 2): [4, 5, 10, 11, 18, 19, 23, 28, 29],
  (3, 4): [2, 3, 8, 9, 16, 17, 22, 26, 27],
  (4, 3): [6, 7, 12, 13, 20, 21, 25, 30, 31],
  (4, 5): [4, 5, 10, 11, 18, 19, 24, 28, 29],
  (5, 4): [0, 1, 8, 9, 14, 15, 22, 23, 27],
  (5, 6): [6, 7, 12, 13, 20, 21, 26, 30, 31],
  (6, 5): [2, 3, 10, 11, 16, 17, 24, 25, 29],
  (6, 7): [0, 1, 8, 9, 14, 15, 22, 23, 28],
  (7, 6): [4, 5, 12, 13, 18, 19, 26, 27, 31],
  (7, 8): [2, 3, 10, 11, 16, 17, 24, 25, 30],
  (8, 7): [1, 6, 7, 14, 15, 20, 21, 28, 29],
  (8, 9): [0, 4, 5, 12, 13, 18, 19, 26, 27],
  (9, 8): [3, 8, 9, 16, 17, 22, 23, 30, 31],
  (9, 10): [2, 6, 7, 14, 15, 20, 21, 28, 29],
  (10, 9): [0, 1, 5, 10, 11, 18, 19, 24, 25],
  (10, 11): [4, 8, 9, 16, 17, 22, 23, 30, 31],
  (11, 10): [2, 3, 7, 12, 13, 20, 21, 26, 27],
  (11, 12): [0, 1, 6, 10, 11, 18, 19, 24, 25],
  (12, 11): [4, 5, 9, 14, 15, 22, 23, 28, 29],
  (12, 13): [2, 3, 8, 12, 13, 20, 21, 26, 27],
  (13, 12): [6, 7, 11, 16, 17, 24, 25, 30, 31],
  (13, 14): [4, 5, 10, 14, 15, 22, 23, 28, 29],
  (14, 13): [0, 1, 8, 9, 13, 18, 19, 26, 27],
  (14, 15): [6, 7, 12, 16, 17, 24, 25, 30, 31],
  (15, 0): [0, 1, 8, 9, 14, 18, 19, 26, 27],
  (15, 14): [2, 3, 10, 11, 15, 20, 21, 28, 29]},
 {(0, 3): [6, 7, 14, 15],
  (0, 13): [18, 19, 26, 27],
  (1, 4): [8, 9, 16, 17],
  (1, 14): [20, 21, 28, 29],
  (2, 5): [10, 11, 18, 19],
  (2, 15): [22, 23, 30, 31],
  (3, 0): [0, 1, 24, 25],
  (3, 6): [12, 13, 20, 21],
  (4, 1): [2, 3, 26, 27],
  (4, 7): [14, 15, 22, 23],
  (5, 2): [4, 5, 28, 29],
  (5, 8): [16, 17, 24, 25],
  (6, 3): [6, 7, 30, 31],
  (6, 9): [18, 19, 26, 27],
  (7, 4): [0, 1, 8, 9],
  (7, 10): [20, 21, 28, 29],
  (8, 5): [2, 3, 10, 11],
  (8, 11): [22, 23, 30, 31],
  (9, 6): [4, 5, 12, 13],
  (9, 12): [0, 1, 24, 25],
  (10, 7): [6, 7, 14, 15],
  (10, 13): [2, 3, 26, 27],
  (11, 8): [8, 9, 16, 17],
  (11, 14): [4, 5, 28, 29],
  (12, 9): [10, 11, 18, 19],
  (12, 15): [6, 7, 30, 31],
  (13, 0): [0, 1, 8, 9],
  (13, 10): [12, 13, 20, 21],
  (14, 1): [2, 3, 10, 11],
  (14, 11): [14, 15, 22, 23],
  (15, 2): [4, 5, 12, 13],
  (15, 12): [16, 17, 24, 25]},
 {(0, 4): [8, 9],
  (0, 12): [24, 25],
  (1, 5): [10, 11],
  (1, 13): [26, 27],
  (2, 6): [12, 13],
  (2, 14): [28, 29],
  (3, 7): [14, 15],
  (3, 15): [30, 31],
  (4, 0): [0, 1],
  (4, 8): [16, 17],
  (5, 1): [2, 3],
  (5, 9): [18, 19],
  (6, 2): [4, 5],
  (6, 10): [20, 21],
  (7, 3): [6, 7],
  (7, 11): [22, 23],
  (8, 4): [8, 9],
  (8, 12): [24, 25],
  (9, 5): [10, 11],
  (9, 13): [26, 27],
  (10, 6): [12, 13],
  (10, 14): [28, 29],
  (11, 7): [14, 15],
  (11, 15): [30, 31],
  (12, 0): [0, 1],
  (12, 8): [16, 17],
  (13, 1): [2, 3],
  (13, 9): [18, 19],
  (14, 2): [4, 5],
  (14, 10): [20, 21],
  (15, 3): [6, 7],
  (15, 11): [22, 23]}]


def _trivance_n16_peers(rank: int, step: int):
    distance = _TRIVANCE_N16_DISTANCES[step]
    return (rank - distance) % 16, (rank + distance) % 16


def _trivance_n16_latency_allgather_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    A naive latency implementation would overcount in the final step because the coverage sets overlap. To keep the result
    correct, this implementation propagates one row per source rank together
    with a source mask, deduplicates sources, and sums the 16 unique source
    rows at the end.

    This is for feasibility, not performance.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != 16:
        raise ValueError(
            f"This special Trivance latency path only supports world_size=16, got {world_size}."
        )

    original_shape = x.shape
    work = x.contiguous().view(-1)
    numel = work.numel()

    # state[src] stores the contribution from source rank src if mask[src] == 1.
    state = torch.zeros((16, numel), dtype=work.dtype, device=work.device)
    state[rank].copy_(work)

    mask = torch.zeros(16, dtype=torch.bool, device=work.device)
    mask[rank] = True

    # Pack mask and rows into a single tensor so each peer exchange uses one
    # P2P message. This avoids tag/order issues with multiple small messages.
    payload_width = max(numel, 16)

    for step, distance in enumerate(_TRIVANCE_N16_DISTANCES):
        left_peer = (rank - distance) % 16
        right_peer = (rank + distance) % 16

        payload = torch.zeros((17, payload_width), dtype=work.dtype, device=work.device)
        payload[0, :16].copy_(mask.to(dtype=work.dtype))
        payload[1:17, :numel].copy_(state)

        send_left = payload.contiguous()
        send_right = payload.contiguous().clone()
        recv_left = torch.empty_like(payload)
        recv_right = torch.empty_like(payload)

        ops = [
            dist.P2POp(dist.isend, send_left, left_peer),
            dist.P2POp(dist.irecv, recv_left, left_peer),
            dist.P2POp(dist.isend, send_right, right_peer),
            dist.P2POp(dist.irecv, recv_right, right_peer),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        for recv_payload in (recv_left, recv_right):
            recv_mask = recv_payload[0, :16] != 0
            recv_state = recv_payload[1:17, :numel]
            for src in range(16):
                if bool(recv_mask[src].item()) and not bool(mask[src].item()):
                    state[src].copy_(recv_state[src])
                    mask[src] = True

    if int(mask.sum().item()) != 16:
        raise RuntimeError(
            f"Trivance n=16 latency propagation incomplete on rank {rank}: "
            f"got {int(mask.sum().item())} sources."
        )

    result = state.sum(dim=0)
    work.copy_(result)
    x.copy_(work.view(original_shape))
    return x

def trivance_latency_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Trivance latency AllReduce.

    Supports:
    - world_size = 3^k: original Trivance latency schedule.
    - world_size = 16: special [1, 3, 4] source-propagation adaptation.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if world_size == 16:
        return _trivance_n16_latency_allgather_sum_(x)

    if not _is_power_of(world_size, 3):
        raise ValueError(
            f"Trivance latency version requires world_size=3^k, or special world_size=16; "
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
    Trivance bandwidth AllReduce.

    Supports:
    - world_size = 3^k: original Trivance-B schedule.
    - world_size = 16: special schedule from n16-propagation.txt.

    For world_size=16, the logical 16 blocks are split into 32 half-blocks.
    A token like 08(1) maps to the first half of block 8, while 08 maps to
    both halves of block 8. 
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if world_size == 16:
        original_shape, flat, original_numel, work, chunks = _prepare_work_buffer(x, 32)

        # Phase 1: Reduce-Scatter.
        for step, schedule in enumerate(_TRIVANCE_N16_BANDWIDTH_SCHEDULE):
            outgoing = sorted([(dst, indices) for (src, dst), indices in schedule.items() if src == rank])
            incoming = sorted([(src, indices) for (src, dst), indices in schedule.items() if dst == rank])

            if len(outgoing) != 2 or len(incoming) != 2:
                raise RuntimeError(f"Invalid n=16 Trivance schedule at step {step}, rank {rank}.")

            recv_buffers = []
            ops = []
            for dst, indices in outgoing:
                ops.append(dist.P2POp(dist.isend, _pack_chunks(chunks, indices), dst))
            for src, indices in incoming:
                recv_buf = _recv_buffer_like(chunks, indices)
                recv_buffers.append((indices, recv_buf))
                ops.append(dist.P2POp(dist.irecv, recv_buf, src))

            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            for indices, recv_buf in recv_buffers:
                _add_packed_to_chunks(chunks, indices, recv_buf)

        # Phase 2: AllGather.
        for step in reversed(range(len(_TRIVANCE_N16_BANDWIDTH_SCHEDULE))):
            schedule = _TRIVANCE_N16_BANDWIDTH_SCHEDULE[step]

            # Original incoming edges src -> rank become outgoing rank -> src.
            outgoing = sorted([(src, indices) for (src, dst), indices in schedule.items() if dst == rank])
            # Original outgoing edges rank -> dst become incoming dst -> rank.
            incoming = sorted([(dst, indices) for (src, dst), indices in schedule.items() if src == rank])

            if len(outgoing) != 2 or len(incoming) != 2:
                raise RuntimeError(f"Invalid n=16 Trivance reverse schedule at step {step}, rank {rank}.")

            recv_buffers = []
            ops = []
            for dst, indices in outgoing:
                ops.append(dist.P2POp(dist.isend, _pack_chunks(chunks, indices), dst))
            for src, indices in incoming:
                recv_buf = _recv_buffer_like(chunks, indices)
                recv_buffers.append((indices, recv_buf))
                ops.append(dist.P2POp(dist.irecv, recv_buf, src))

            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            for indices, recv_buf in recv_buffers:
                _copy_packed_to_chunks(chunks, indices, recv_buf)

        return _finish_work_buffer(x, original_shape, flat, original_numel, work)

    if not _is_power_of(world_size, 3):
        raise ValueError(
            f"Trivance bandwidth version requires world_size=3^k, or special world_size=16; "
            f"got world_size={world_size}."
        )

    steps = _log_power(world_size, 3)
    original_shape, flat, original_numel, work, chunks = _prepare_work_buffer(x, world_size)

    for step in range(steps):
        left_peer, right_peer = _trivance_peers(rank, step, world_size)
        if left_peer == right_peer:
            raise RuntimeError("Invalid Trivance peer selection.")

        local_indices = _trivance_subtree_indices(rank, step + 1, world_size, steps)
        send_left_indices = _trivance_subtree_indices(left_peer, step + 1, world_size, steps)
        send_right_indices = _trivance_subtree_indices(right_peer, step + 1, world_size, steps)

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

    for step in reversed(range(steps)):
        left_peer, right_peer = _trivance_peers(rank, step, world_size)

        local_indices = _trivance_subtree_indices(rank, step + 1, world_size, steps)
        recv_left_indices = _trivance_subtree_indices(left_peer, step + 1, world_size, steps)
        recv_right_indices = _trivance_subtree_indices(right_peer, step + 1, world_size, steps)

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
