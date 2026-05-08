
import math
import torch
import torch.distributed as dist

def _is_power_of(n: int, base: int) -> bool:
    if n < 1:
        return False
    while n % base == 0:
        n //= base
    return n == 1


def _ceil_log(n: int, base: int) -> int:
    if n <= 1:
        return 0
    return math.ceil(math.log(n, base))


def _swing_rho(step: int) -> int:
    """
    Swing rho(s) = sum_{i=0}^{s} (-2)^i = (1 - (-2)^(s+1)) / 3
    """
    return int((1 - ((-2) ** (step + 1))) // 3)

def builtin_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Built-in all-reduce SUM.
    This is used as a baseline.
    """
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def ring_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Custom AllReduce SUM using the ring algorithm.

    Works with CPU tensors + Gloo backend.
    For DDP gradient buckets, x is usually a flattened 1D tensor.

    If x.numel() is not divisible by world_size, we pad internally
    and copy the valid part back to x.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

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

    right = (rank + 1) % world_size
    left = (rank - 1 + world_size) % world_size

    chunks = list(work.chunk(world_size))
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

    if work.data_ptr() != flat.data_ptr():
        flat.copy_(work[:original_numel])

    x.copy_(flat.view(original_shape))
    return x

def swing_latency_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Latency-optimal Swing AllReduce prototype.

    Requirement:
    - world_size must be a power of 2.

    Paper logic:
    - At step s, rank r communicates with pi(r, s).
    - pi(r, s) = r + rho(s) mod p, if r is even
               = r - rho(s) mod p, if r is odd
    - rho(s) = sum_{i=0}^{s} (-2)^i
    - The latency-optimal version exchanges the entire vector at each step.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if not _is_power_of(world_size, 2):
        raise ValueError(
            f"Swing latency-optimal prototype currently requires "
            f"world_size to be a power of 2, got world_size={world_size}."
        )

    steps = int(math.log2(world_size))

    original_shape = x.shape
    work = x.contiguous().view(-1)

    recv_buf = torch.empty_like(work)

    for step in range(steps):
        rho = _swing_rho(step)

        if rank % 2 == 0:
            peer = (rank + rho) % world_size
        else:
            peer = (rank - rho) % world_size

        send_buf = work.contiguous()

        ops = [
            dist.P2POp(dist.isend, send_buf, peer),
            dist.P2POp(dist.irecv, recv_buf, peer),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        work.add_(recv_buf)

    x.copy_(work.view(original_shape))
    return x


def trivance_latency_allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """
    Latency-optimal Trivance AllReduce prototype.

    Requirement:
    - world_size must be a power of 3.

    Paper logic:
    - At step k, rank r communicates with:
        left  = r - 3^k mod n
        right = r + 3^k mod n
    - Each node uses both directions simultaneously.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return x

    if not _is_power_of(world_size, 3):
        raise ValueError(
            f"Trivance latency-optimal prototype currently requires "
            f"world_size to be a power of 3, got world_size={world_size}."
        )

    steps = int(round(math.log(world_size, 3)))

    original_shape = x.shape
    work = x.contiguous().view(-1)

    recv_left = torch.empty_like(work)
    recv_right = torch.empty_like(work)

    for step in range(steps):
        distance = 3 ** step

        left_peer = (rank - distance) % world_size
        right_peer = (rank + distance) % world_size

        if left_peer == right_peer:
            raise RuntimeError(
                "Invalid Trivance peer selection: left_peer == right_peer. "
                "This usually means world_size is too small or not a valid "
                "power-of-three configuration."
            )

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

        # Joint reduction from both incoming communications.
        work.add_(recv_left)
        work.add_(recv_right)

    x.copy_(work.view(original_shape))
    return x

def allreduce_sum_(x: torch.Tensor, algo: str) -> torch.Tensor:
    """
    Unified interface for all-reduce algorithms.
    """
    if algo == "builtin":
        return builtin_allreduce_sum_(x)
    elif algo == "ring":
        return ring_allreduce_sum_(x)
    elif algo == "swing-latency":
        return swing_latency_allreduce_sum_(x)
    elif algo == "trivance-latency":
        return trivance_latency_allreduce_sum_(x)
    else:
        raise ValueError(f"Unknown all-reduce algorithm: {algo}")