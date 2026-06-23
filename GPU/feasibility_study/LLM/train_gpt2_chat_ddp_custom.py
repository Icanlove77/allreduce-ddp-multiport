#!/usr/bin/env python3
"""
Fine-tune GPT-2 with PyTorch DDP and a custom DDP communication hook.

Use cases:
  - 8 GPUs: builtin / ring / recursive-doubling / swing
  - 3 GPUs: bruck / trivance
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_allreduce import allreduce_sum_


# ----------------------------- distributed utils -----------------------------

def setup_distributed() -> Tuple[int, int, int, torch.device]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    try:
        dist.init_process_group(backend="nccl", device_id=device)
    except TypeError:
        dist.init_process_group(backend="nccl")

    return rank, local_rank, world_size, device


def barrier(local_rank: int) -> None:
    dist.barrier(device_ids=[local_rank])


def validate_world_size(algo: str, world_size: int) -> None:
    power2_algos = {
        "recursive-doubling-latency",
        "recursive-doubling-bandwidth",
        "swing-latency",
        "swing-bandwidth",
    }
    power3_algos = {
        "bruck-latency",
        "bruck-bandwidth",
        "trivance-latency",
        "trivance-bandwidth",
    }

    if algo in power2_algos and (world_size & (world_size - 1)) != 0:
        raise ValueError(f"{algo} requires world_size=2^k; got {world_size}")

    if algo in power3_algos:
        tmp = world_size
        while tmp > 1 and tmp % 3 == 0:
            tmp //= 3
        if tmp != 1:
            raise ValueError(f"{algo} requires world_size=3^k; got {world_size}")


# ----------------------------- DDP communication hook -----------------------------

def make_comm_hook(algo: str, world_size: int):
    state = {"algo": algo, "world_size": world_size, "calls": 0}

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


# ----------------------------- data processing -----------------------------

class TokenizedChatDataset(Dataset):
    def __init__(self, encoded_samples: List[Dict[str, torch.Tensor]]):
        self.samples = encoded_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def _safe_get(row: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _format_dialogue_pair(user_text: str, assistant_text: str, system_prompt: str = "") -> str:
    user_text = str(user_text).strip()
    assistant_text = str(assistant_text).strip()
    if not user_text or not assistant_text:
        return ""
    prefix = ""
    if system_prompt:
        prefix = f"System: {system_prompt.strip()}\n"
    return f"{prefix}User: {user_text}\nAssistant: {assistant_text}<|endoftext|>"


def _messages_to_text(messages: List[Any], system_prompt: str = "") -> str:
    parts = []
    if system_prompt:
        parts.append(f"System: {system_prompt.strip()}")
    for m in messages:
        if isinstance(m, dict):
            role = str(m.get("role", m.get("from", ""))).lower()
            content = str(m.get("content", m.get("value", ""))).strip()
            if not content:
                continue
            if role in {"user", "human"}:
                parts.append(f"User: {content}")
            elif role in {"assistant", "gpt", "bot"}:
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        elif isinstance(m, str) and m.strip():
            # Unknown role; keep as plain text.
            parts.append(m.strip())
    if not parts:
        return ""
    return "\n".join(parts) + "<|endoftext|>"


def row_to_training_texts(row: Dict[str, Any], system_prompt: str = "") -> List[str]:
    texts: List[str] = []

    # prompt-response style.
    prompt = _safe_get(row, ["prompt", "question", "instruction", "query", "input"])
    response = _safe_get(row, ["response", "answer", "output", "completion", "target"])
    if prompt is not None and response is not None:
        text = _format_dialogue_pair(str(prompt), str(response), system_prompt)
        if text:
            texts.append(text)
            return texts

    # role-based messages.
    messages = _safe_get(row, ["messages", "conversations"])
    if isinstance(messages, list):
        text = _messages_to_text(messages, system_prompt)
        if text:
            texts.append(text)
            return texts

    # list-of-utterances dialogue datasets, e.g., DailyDialog-like schemas.
    dialog = _safe_get(row, ["dialog", "dialogue", "utterances", "conversation"])
    if isinstance(dialog, list):
        utterances = [str(x).strip() for x in dialog if str(x).strip()]
        # Build one sample per adjacent pair to make chat-style prompt-response data.
        for i in range(len(utterances) - 1):
            text = _format_dialogue_pair(utterances[i], utterances[i + 1], system_prompt)
            if text:
                texts.append(text)
        if texts:
            return texts

    # Plain LM fallback.
    plain = _safe_get(row, ["text", "content"])
    if isinstance(plain, str) and plain.strip():
        texts.append(plain.strip() + "<|endoftext|>")

    return texts


def load_training_texts(args, rank: int) -> List[str]:
    if args.local_jsonl:
        path = Path(args.local_jsonl)
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        raw_rows = rows[: args.max_samples] if args.max_samples > 0 else rows
    else:
        if rank == 0:
            print(
                f"Loading dataset: {args.dataset_name}/{args.dataset_config}:{args.dataset_split}",
                flush=True,
            )
        load_kwargs = {"split": args.dataset_split}
        if args.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        if args.dataset_config:
            ds = load_dataset(args.dataset_name, args.dataset_config, **load_kwargs)
        else:
            ds = load_dataset(args.dataset_name, **load_kwargs)
        if args.max_samples > 0:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        raw_rows = [ds[i] for i in range(len(ds))]

    texts: List[str] = []
    for row in raw_rows:
        texts.extend(row_to_training_texts(row, args.system_prompt))

    # Filter very short samples.
    texts = [t for t in texts if len(t.strip()) >= args.min_chars]
    if not texts:
        raise RuntimeError("No usable training texts were produced. Check dataset schema or use --local-jsonl.")
    return texts


def build_dataset(args, tokenizer, rank: int) -> TokenizedChatDataset:
    texts = load_training_texts(args, rank)
    if rank == 0:
        print(f"training_text_samples={len(texts)}", flush=True)
        print("example_training_text_begin", flush=True)
        print(texts[0][:500].replace("\n", "\\n"), flush=True)
        print("example_training_text_end", flush=True)

    encoded_samples: List[Dict[str, torch.Tensor]] = []
    for text in texts:
        enc = tokenizer(
            text,
            max_length=args.block_size,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if len(input_ids) < 8:
            continue
        encoded_samples.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(input_ids, dtype=torch.long),
            }
        )

    if not encoded_samples:
        raise RuntimeError("All tokenized samples are too short. Try a different dataset or lower --min-chars.")
    return TokenizedChatDataset(encoded_samples)


def collate_pad_batch(features: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(f["input_ids"].numel() for f in features)
    batch = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        rows = []
        for f in features:
            x = f[key]
            pad_len = max_len - x.numel()
            if key == "input_ids":
                pad_value = pad_token_id
            elif key == "attention_mask":
                pad_value = 0
            else:
                pad_value = -100
            if pad_len > 0:
                x = torch.cat([x, torch.full((pad_len,), pad_value, dtype=x.dtype)], dim=0)
            rows.append(x)
        batch[key] = torch.stack(rows, dim=0)
    return batch


# ----------------------------- model/checkpoint utils -----------------------------

def dtype_from_arg(dtype_name: str):
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model(model_name: str, dtype: torch.dtype):
    # Current Transformers prefers dtype=..., older versions expect torch_dtype=...
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parameter_checksum(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    checksum = torch.zeros(4, device=device, dtype=torch.float64)
    for p in model.parameters():
        data = p.detach().float()
        checksum[0] += data.sum().double()
        checksum[1] += (data * data).sum().double()
        checksum[2] += data.abs().sum().double()
        checksum[3] += float(data.numel())
    return checksum


def full_parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().float().reshape(-1) for p in model.parameters()], dim=0)


def check_parameter_consistency(
    model: torch.nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    full_check: bool,
):
    local = parameter_checksum(model, device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)

    if rank == 0:
        stacked = torch.stack(gathered, dim=0)
        max_diff = (stacked - stacked[0]).abs().max().item()
        print(f"parameter_checksum_max_diff_across_ranks={max_diff:.12e}", flush=True)
        print(f"parameter_checksum_rank0={stacked[0].detach().cpu().tolist()}", flush=True)

    if not full_check:
        return

    # Strict check. This is acceptable for GPT-2 small on A100, but do not use it
    # for 1B+ models unless you are comfortable with the temporary memory cost.
    vec = full_parameter_vector(model)
    gathered_vecs = [torch.empty_like(vec) for _ in range(world_size)]
    dist.all_gather(gathered_vecs, vec)
    if rank == 0:
        base = gathered_vecs[0]
        max_param_diff = max((v - base).abs().max().item() for v in gathered_vecs[1:]) if world_size > 1 else 0.0
        print(f"parameter_full_max_diff_across_ranks={max_param_diff:.12e}", flush=True)


def quick_generation_preview(model, tokenizer, device, prompt: str, rank: int, local_rank: int, max_new_tokens: int = 80):
    barrier(local_rank)
    if rank != 0:
        barrier(local_rank)
        return
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("generation_preview_begin", flush=True)
    print(tokenizer.decode(out[0], skip_special_tokens=True), flush=True)
    print("generation_preview_end", flush=True)
    model.train()
    barrier(local_rank)


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="trivance-bandwidth", choices=[
        "builtin", "ring", "recursive-doubling-latency", "recursive-doubling-bandwidth",
        "swing-latency", "swing-bandwidth", "bruck-latency", "bruck-bandwidth",
        "trivance-latency", "trivance-bandwidth",
    ])
    parser.add_argument("--model-name", type=str, default="openai-community/gpt2")
    parser.add_argument("--dataset-name", type=str, default="daily_dialog")
    parser.add_argument("--dataset-config", type=str, default="")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow Hugging Face datasets that require executing dataset loading code, e.g., daily_dialog.")
    parser.add_argument("--local-jsonl", type=str, default="", help="Optional local JSONL with prompt/response, messages, or dialog fields.")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bucket-cap-mb", type=float, default=25.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--full-param-check", action="store_true", help="Strict full all_gather parameter check. OK for GPT-2 small on A100.")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./gpt2_chat_custom_allreduce")
    parser.add_argument("--preview-prompt", type=str, default="User: Hello, how are you?\nAssistant:")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rank, local_rank, world_size, device = setup_distributed()
    validate_world_size(args.algo, world_size)

    if rank == 0:
        print("===== GPT-2 Chat DDP Config =====", flush=True)
        for k, v in vars(args).items():
            print(f"{k}={v}", flush=True)
        print(f"world_size={world_size}", flush=True)
        print("=================================", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(args, tokenizer, rank)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda feats: collate_pad_batch(feats, tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    model_dtype = dtype_from_arg(args.dtype)
    model = load_model(args.model_name, model_dtype)
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=args.bucket_cap_mb,
        find_unused_parameters=False,
    )
    hook_state, hook = make_comm_hook(args.algo, world_size)
    ddp_model.register_comm_hook(hook_state, hook)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if rank == 0:
        total_params, trainable_params = count_parameters(ddp_model.module)
        print(f"total_params={total_params}", flush=True)
        print(f"trainable_params={trainable_params}", flush=True)
        print(f"dataset_sequences={len(dataset)}", flush=True)

    barrier(local_rank)
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    step = 0
    epoch = 0
    while step < args.steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = ddp_model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if step % args.log_interval == 0:
                loss_avg = loss.detach().clone()
                dist.all_reduce(loss_avg, op=dist.ReduceOp.SUM)
                loss_avg = loss_avg / world_size
                if rank == 0:
                    print(f"step={step:05d} loss_avg={loss_avg.item():.6f} algo={args.algo}", flush=True)

            step += 1
            if step >= args.steps:
                break
        epoch += 1

    barrier(local_rank)
    torch.cuda.synchronize(device)
    end_time = time.perf_counter()

    local_calls = torch.tensor([hook_state["calls"]], dtype=torch.long, device=device)
    gathered_calls = [torch.zeros_like(local_calls) for _ in range(world_size)]
    dist.all_gather(gathered_calls, local_calls)
    if rank == 0:
        calls = [x.item() for x in gathered_calls]
        elapsed = end_time - start_time
        print(f"hook_calls_per_rank={calls}", flush=True)
        print(f"total_training_time_sec={elapsed:.3f}", flush=True)
        print(f"avg_step_time_ms={elapsed * 1000.0 / args.steps:.3f}", flush=True)

    check_parameter_consistency(ddp_model.module, rank, world_size, device, args.full_param_check)

    quick_generation_preview(ddp_model.module, tokenizer, device, args.preview_prompt, rank, local_rank)

    barrier(local_rank)
    if args.save_model and rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        ddp_model.module.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        # Save a small note for reproducibility.
        with open(os.path.join(args.output_dir, "training_args_custom.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        print(f"saved_model_dir={args.output_dir}", flush=True)
    barrier(local_rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
