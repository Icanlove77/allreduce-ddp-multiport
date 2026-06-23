import os
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from custom_allreduce import allreduce_sum_


class SmallMLP(nn.Module):
    """
    A small synthetic model for validating DDP communication hooks on GPUs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def setup_distributed(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This GPU script requires CUDA.")

    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {num_gpus} CUDA devices are visible."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Explicitly bind this process group to the local CUDA device when the
    # PyTorch version supports it. This removes NCCL warnings such as:
    # "No device id is provided via init_process_group or barrier".
    try:
        dist.init_process_group(backend=backend, device_id=device)
    except TypeError:
        # Older PyTorch versions may not support the device_id argument.
        # In that case, the device-aware barriers below are still enough.
        dist.init_process_group(backend=backend)

    return rank, world_size, local_rank, device


def gpu_barrier(local_rank: int):
    """NCCL barrier explicitly bound to the local GPU."""
    dist.barrier(device_ids=[local_rank])


def build_teacher(input_dim: int, num_classes: int, device: torch.device):
    # Use a fixed seed so every rank builds the same teacher matrix.
    g = torch.Generator(device=device)
    g.manual_seed(2026)
    return torch.randn(input_dim, num_classes, generator=g, device=device)


def make_batch(batch_size: int, input_dim: int, teacher: torch.Tensor, seed: int, device: torch.device):
    # Different ranks use different synthetic mini-batches, but deterministic seeds.
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(batch_size, input_dim, generator=g, device=device)
    logits = x @ teacher
    y = torch.argmax(logits, dim=1)
    return x, y


def make_comm_hook(algo: str, world_size: int):
    """
    Synchronous DDP communication hook.

    The hook receives a CUDA gradient bucket, runs the selected custom
    allreduce in-place, divides by world_size, and returns a completed Future.
    This validates correctness/feasibility. It does not try to overlap
    communication with backward computation.
    """
    state = {
        "algo": algo,
        "world_size": world_size,
        "calls": 0,
    }

    def hook(state, bucket):
        buf = bucket.buffer()
        if not buf.is_cuda:
            raise RuntimeError("Expected a CUDA DDP bucket in the GPU training script.")

        with torch.no_grad():
            allreduce_sum_(buf, state["algo"])
            buf.div_(state["world_size"])

        state["calls"] += 1
        fut = torch.futures.Future()
        fut.set_result(buf)
        return fut

    return state, hook


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    params = [p.detach().float().view(-1) for p in model.parameters()]
    return torch.cat(params)


def check_parameter_consistency(model: nn.Module, rank: int, world_size: int, device: torch.device):
    """
    Check whether all ranks ended with the same parameters.
    """
    vec = flatten_parameters(model).to(device)
    base = torch.empty_like(vec)
    if rank == 0:
        base.copy_(vec)

    dist.broadcast(base, src=0)
    local_max = (vec - base).abs().max()
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"parameter_max_diff_across_ranks={local_max.item():.8f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        default="ring",
        choices=[
            "builtin",
            "ring",
            "recursive-doubling-latency",
            "recursive-doubling-bandwidth",
            "swing-latency",
            "swing-bandwidth",
            "bruck-latency",
            "bruck-bandwidth",
            "trivance-latency",
            "trivance-bandwidth",
        ],
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bucket-cap-mb", type=float, default=25.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl"])
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = True

    rank, world_size, local_rank, device = setup_distributed(args.backend)

    # Make sure all ranks start from the same model parameters.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = SmallMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
    ).to(device)

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=args.bucket_cap_mb,
    )

    hook_state, hook = make_comm_hook(args.algo, world_size)
    ddp_model.register_comm_hook(hook_state, hook)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
    teacher = build_teacher(args.input_dim, args.num_classes, device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print("===== Training Config =====", flush=True)
        print(f"algo={args.algo}", flush=True)
        print(f"backend={args.backend}", flush=True)
        print(f"world_size={world_size}", flush=True)
        print(f"visible_cuda_devices={torch.cuda.device_count()}", flush=True)
        print(f"total_params={total_params}", flush=True)
        print(f"bucket_cap_mb={args.bucket_cap_mb}", flush=True)
        print(f"threads_per_rank={args.threads}", flush=True)
        print("===========================", flush=True)

    gpu_barrier(local_rank)
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for step in range(args.steps):
        seed = 100000 + step * world_size + rank
        x, y = make_batch(args.batch_size, args.input_dim, teacher, seed, device)

        logits = ddp_model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            loss_detached = loss.detach().clone()
            dist.all_reduce(loss_detached, op=dist.ReduceOp.SUM)
            loss_avg = loss_detached / world_size
            if rank == 0:
                print(
                    f"step={step:04d} loss_avg={loss_avg.item():.6f} algo={args.algo}",
                    flush=True,
                )

    gpu_barrier(local_rank)
    torch.cuda.synchronize(device)
    end_time = time.perf_counter()

    local_hook_calls = torch.tensor([hook_state["calls"]], dtype=torch.long, device=device)
    gathered_calls = [torch.zeros_like(local_hook_calls) for _ in range(world_size)]
    dist.all_gather(gathered_calls, local_hook_calls)

    if rank == 0:
        calls = [x.item() for x in gathered_calls]
        elapsed = end_time - start_time
        print(f"hook_calls_per_rank={calls}", flush=True)
        print(f"total_training_time_sec={elapsed:.3f}", flush=True)
        print(f"avg_step_time_ms={elapsed * 1000.0 / args.steps:.3f}", flush=True)

    check_parameter_consistency(ddp_model.module, rank, world_size, device)

    gpu_barrier(local_rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
