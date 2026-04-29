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

Currently implemented:

- `builtin_allreduce_sum_`
- `ring_allreduce_sum_`

Future algorithms:

- `trivance_allreduce_sum_`
- `swing_allreduce_sum_`

```text
train_cpu_ddp_custom.py
```

A small PyTorch DDP training script.

It trains a synthetic MLP model and replaces DDP's default gradient synchronization with a custom communication hook.

## How to Run

Basic command:

```bash
torchrun --standalone --nproc_per_node=2 train_cpu_ddp_custom.py \
  --algo ring \
  --steps 50 \
  --batch-size 128 \
  --threads 4
```

Recommended command for single-machine CPU testing:

```bash
GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 \
torchrun --standalone --nproc_per_node=2 train_cpu_ddp_custom.py \
  --algo ring \
  --steps 50 \
  --batch-size 128 \
  --threads 4
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
```

- `builtin`: uses PyTorch/Gloo built-in all-reduce
- `ring`: uses the custom Ring AllReduce implementation

Future options may include:

```text
trivance
swing
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

- Implement Trivance in Python
- Implement Swing in Python
- Extend to GPU/NCCL-based distributed training
- Compare custom algorithms with NCCL all-reduce on GPU clusters