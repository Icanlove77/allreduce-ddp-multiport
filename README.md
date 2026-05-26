# Custom AllReduce for PyTorch DDP

This repository provides a prototype for validating custom all-reduce algorithms in PyTorch Distributed Data Parallel (DDP) training.

The current goal is to verify whether custom collective communication algorithms, such as Ring AllReduce and later Trivance/Swing, can be integrated into the gradient synchronization path of PyTorch DDP.

## Environment Setup

Create a conda environment:

```bash
conda create -n ddp_cpu python=3.10 -y
conda activate ddp_cpu
```

Install the CPU version of PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install NumPy:

```bash
pip install numpy
```

Check the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.distributed.is_available())"
```

Expected output should show a CPU PyTorch version and:

```text
True
```

## Files

```text
custom_allreduce.py
```

Contains custom all-reduce implementations.

```text
train_cpu_ddp_custom.py
```

A small PyTorch DDP training script.

It trains a synthetic MLP model and replaces DDP's default gradient synchronization with a custom communication hook.

## Run & Test Commands

### Test Built-in AllReduce

```bash
GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29501 \
train_cpu_ddp_custom.py \
  --algo builtin \
  --steps 20 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --bucket-cap-mb 1 \
  --threads 2
```

### Test Ring

```bash
GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29502 \
train_cpu_ddp_custom.py \
  --algo ring \
  --steps 20 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --bucket-cap-mb 1 \
  --threads 2
```

### Test Recursive Doubling

Recursive Doubling currently requires:

```text
world_size = 2^k
```

For example, use 4 ranks.

#### Recursive Doubling Latency Version

```bash
GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29503 \
train_cpu_ddp_custom.py \
  --algo recursive-doubling-latency \
  --steps 20 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --bucket-cap-mb 1 \
  --threads 2
```

#### Recursive Doubling Bandwidth Version

```bash
GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29504 \
train_cpu_ddp_custom.py \
  --algo recursive-doubling-bandwidth \
  --steps 20 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --bucket-cap-mb 1 \
  --threads 2
```

### Test Swing

Swing currently requires:

```text
world_size = 2^k
```

For example, use 4 or 8 ranks.

#### Swing Latency Version

```bash
GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29505 \
train_cpu_ddp_custom.py \
  --algo swing-latency \
  --steps 20 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --bucket-cap-mb 1 \
  --threads 2
```

#### Swing Bandwidth Version

```bash
GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29506 \
train_cpu_ddp_custom.py \
  --algo swing-bandwidth \
  --steps 20 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --bucket-cap-mb 1 \
  --threads 2
```

### Test Trivance

Trivance currently requires:

```text
world_size = 3^k
```

For example, use 3 or 9 ranks.

#### Trivance Latency Version

```bash
GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
torchrun --nnodes=1 --nproc_per_node=9 --master_addr=127.0.0.1 --master_port=29507 \
train_cpu_ddp_custom.py \
  --algo trivance-latency \
  --steps 20 \
  --batch-size 16 \
  --hidden-dim 64 \
  --num-layers 2 \
  --bucket-cap-mb 1 \
  --threads 1
```

#### Trivance Bandwidth Version

```bash
GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
torchrun --nnodes=1 --nproc_per_node=9 --master_addr=127.0.0.1 --master_port=29508 \
train_cpu_ddp_custom.py \
  --algo trivance-bandwidth \
  --steps 20 \
  --batch-size 16 \
  --hidden-dim 64 \
  --num-layers 2 \
  --bucket-cap-mb 1 \
  --threads 1
```

### Test Bruck

Bruck currently requires:

```text
world_size = 3^k
```

For example, use 3 or 9 ranks.

#### Bruck Latency Version

```bash
GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
torchrun --nnodes=1 --nproc_per_node=9 --master_addr=127.0.0.1 --master_port=29509 \
train_cpu_ddp_custom.py \
  --algo bruck-latency \
  --steps 20 \
  --batch-size 16 \
  --hidden-dim 64 \
  --num-layers 2 \
  --bucket-cap-mb 1 \
  --threads 1
```

#### Bruck Bandwidth Version

```bash
GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
torchrun --nnodes=1 --nproc_per_node=9 --master_addr=127.0.0.1 --master_port=29510 \
train_cpu_ddp_custom.py \
  --algo bruck-bandwidth \
  --steps 20 \
  --batch-size 16 \
  --hidden-dim 64 \
  --num-layers 2 \
  --bucket-cap-mb 1 \
  --threads 1
```

If the script runs correctly, the output should include fields such as:

```text
===== Training Config =====
algo=ring
world_size=2
total_params=...
bucket_cap_mb=...
threads_per_rank=...
===========================
step=0000 loss_avg=...
hook_calls_per_rank=[50, 50]
total_training_time_sec=...
avg_step_time_ms=...
parameter_max_diff_across_ranks=0.00000000
```

## Parameters

### `--algo`

Selects the gradient synchronization algorithm.

Available options:

```text
builtin
ring
recursive-doubling-latency
recursive-doubling-bandwidth
swing-latency
swing-bandwidth
bruck-latency
bruck-bandwidth
trivance-latency
trivance-bandwidth
```

### `--steps`

Number of training iterations.

Example:

```bash
--steps 50
```

This means the model will train for 50 iterations.

### `--batch-size`

Mini-batch size used by each rank.

Example:

```bash
--batch-size 128
```

Note that this is the local batch size per rank.  
If `world_size = 2`, the effective global batch size is:

```text
128 × 2 = 256
```

### `--threads`

Number of CPU computation threads used by each rank.

Example:

```bash
--threads 4
```

This is passed to:

```python
torch.set_num_threads(args.threads)
```

For CPU experiments, make sure:

```text
nproc_per_node × threads
```

does not significantly exceed the number of available CPU cores.

### `--bucket-cap-mb`

Maximum DDP gradient bucket size in MB.

Example:

```bash
--bucket-cap-mb 1
```

DDP groups gradients into buckets. Each bucket triggers one communication hook call when ready.

Smaller buckets lead to more frequent all-reduce operations. Larger buckets lead to fewer but larger all-reduce operations.

### `--hidden-dim`

Hidden dimension of the synthetic MLP model.

Example:

```bash
--hidden-dim 512
```

Larger hidden dimensions increase the number of model parameters and therefore increase the total gradient size.

### `--num-layers`

Number of hidden layers in the MLP.

Example:

```bash
--num-layers 4
```

Increasing this value also increases model size and gradient communication volume.

## What the Model Trains

The script trains a small synthetic MLP model.

The dataset is generated on the fly. A fixed random teacher matrix is used to generate labels:

```python
logits = x @ teacher
y = torch.argmax(logits, dim=1)
```

The MLP then learns to predict these synthetic labels.

This task is not intended to evaluate model accuracy on a real dataset. It is used to create a controlled training workload with forward propagation, backward propagation, gradient synchronization, and optimizer updates.

## Correctness Checks

The prototype checks correctness in two ways.

### 1. Hook Invocation Count

Example output:

```text
hook_calls_per_rank=[50, 50]
```

This means each rank invoked the DDP communication hook 50 times.

If the number is zero, the custom communication hook was not used.

### 2. Parameter Consistency Across Ranks

Example output:

```text
parameter_max_diff_across_ranks=0.00000000
```

This means all ranks have identical model parameters after training.

If this value is large, the custom all-reduce implementation may be incorrect.

## Notes

This prototype currently targets CPU feasibility validation.

It demonstrates that custom all-reduce algorithms can be integrated into PyTorch DDP's gradient synchronization path.

It does not yet prove GPU/NCCL performance. Future work will extend this prototype to GPU-resident tensors and multi-GPU clusters.

## Future Work

- Extend to GPU/NCCL-based distributed training
- Compare custom algorithms with NCCL all-reduce on GPU clusters