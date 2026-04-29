
import torch
import torch.distributed as dist


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


def allreduce_sum_(x: torch.Tensor, algo: str) -> torch.Tensor:
    """
    Unified interface for all-reduce algorithms.
    """
    if algo == "builtin":
        return builtin_allreduce_sum_(x)
    elif algo == "ring":
        return ring_allreduce_sum_(x)
    else:
        raise ValueError(f"Unknown all-reduce algorithm: {algo}")