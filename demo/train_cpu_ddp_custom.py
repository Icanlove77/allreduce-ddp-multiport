
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
    A small ML model for demonstration purposes.

    We shall train a large langue model (LLM) in GPU clusters in the future.

    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="gloo")
    return rank, world_size


def build_teacher(input_dim: int, num_classes: int):
    g = torch.Generator()
    g.manual_seed(2026)
    teacher = torch.randn(input_dim, num_classes, generator=g)
    return teacher


def make_batch(batch_size: int, input_dim: int, teacher: torch.Tensor, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)

    x = torch.randn(batch_size, input_dim, generator=g)
    logits = x @ teacher
    y = torch.argmax(logits, dim=1)

    return x, y


def make_comm_hook(algo: str, world_size: int):
    """
    Distributed data parallel (DDP) communication hook. See more at https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html.

    We replace the default all-reduce with our custom algorithm.

    """
    state = {
        "algo": algo,
        "world_size": world_size,
        "calls": 0,
    }

    def hook(state, bucket):
        buf = bucket.buffer()

        with torch.no_grad():
            allreduce_sum_(buf, state["algo"])
            buf.div_(state["world_size"])

        state["calls"] += 1

        fut = torch.futures.Future()
        fut.set_result(buf)
        return fut

    return state, hook


def flatten_parameters(model: nn.Module):
    params = []
    for p in model.parameters():
        params.append(p.detach().float().view(-1))
    return torch.cat(params)


def check_parameter_consistency(model: nn.Module, rank: int, world_size: int):
    vec = flatten_parameters(model)

    gathered = [torch.empty_like(vec) for _ in range(world_size)]
    dist.all_gather(gathered, vec)

    if rank == 0:
        base = gathered[0]
        max_diff = 0.0
        for i in range(1, world_size):
            diff = (gathered[i] - base).abs().max().item()
            max_diff = max(max_diff, diff)
        print(f"parameter_max_diff_across_ranks={max_diff:.8f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, default="ring", choices=["builtin", "ring", "swing-latency", "trivance-latency"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bucket-cap-mb", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)

    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    rank, world_size = setup_distributed()

    # Make sure all ranks start from the same model parameters.
    torch.manual_seed(1234)

    model = SmallMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
    )

    ddp_model = DDP(
        model,
        bucket_cap_mb=args.bucket_cap_mb,
    )

    hook_state, hook = make_comm_hook(args.algo, world_size)
    ddp_model.register_comm_hook(hook_state, hook)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
    teacher = build_teacher(args.input_dim, args.num_classes)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print("===== Training Config =====")
        print(f"algo={args.algo}")
        print(f"world_size={world_size}")
        print(f"total_params={total_params}")
        print(f"bucket_cap_mb={args.bucket_cap_mb}")
        print(f"threads_per_rank={args.threads}")
        print("===========================")

    dist.barrier()
    start_time = time.perf_counter()

    for step in range(args.steps):
        # Different ranks use different synthetic mini-batches.
        seed = 100000 + step * world_size + rank
        x, y = make_batch(args.batch_size, args.input_dim, teacher, seed)

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
                    f"step={step:04d} "
                    f"loss_avg={loss_avg.item():.6f} "
                    f"algo={args.algo}"
                )

    dist.barrier()
    end_time = time.perf_counter()

    local_hook_calls = torch.tensor([hook_state["calls"]], dtype=torch.long)
    gathered_calls = [torch.zeros_like(local_hook_calls) for _ in range(world_size)]
    dist.all_gather(gathered_calls, local_hook_calls)

    if rank == 0:
        calls = [x.item() for x in gathered_calls]
        print(f"hook_calls_per_rank={calls}")
        print(f"total_training_time_sec={end_time - start_time:.3f}")
        print(f"avg_step_time_ms={(end_time - start_time) * 1000.0 / args.steps:.3f}")

    check_parameter_consistency(ddp_model.module, rank, world_size)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()