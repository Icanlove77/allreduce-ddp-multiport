#!/usr/bin/env bash
set -euo pipefail

export GLOO_SOCKET_IFNAME=lo
export PYTHONUNBUFFERED=1

CSV_FILE="training_completion_results.csv"
LOG_DIR="logs_training_bench"
mkdir -p "${LOG_DIR}"

STEPS=300
BATCH_SIZE=16
THREADS=1

# Make bucket large enough so small/medium models are usually kept in one DDP bucket.
# This makes the gradient communication size roughly equal to model parameter size.
BUCKET_CAP_MB=512

# Repeat each setting several times. You can start with 1 for debugging.
REPEATS=5

# Model sizes from small to large.
# Format: "hidden_dim num_layers label"
MODEL_CONFIGS=(
  "32 2 h32_l2"
  "64 2 h64_l2"
  "128 2 h128_l2"
  "256 2 h256_l2"
  "512 2 h512_l2"
  "1024 2 h1024_l2"
  "1024 4 h1024_l4"
)

# Larger configs may be slow on CPU. Uncomment later if needed.
# MODEL_CONFIGS+=(
#   "2048 2 h2048_l2"
# )

# 9-rank algorithms.
# Ring can run with arbitrary world_size, so we put it in the 9-rank group.
ALGOS_9=(
  "ring"
  "bruck-latency"
  "bruck-bandwidth"
  "trivance-latency"
  "trivance-bandwidth"
)

# 8-rank algorithms.
# Recursive Doubling and Swing require power-of-two world_size.
ALGOS_8=(
  "recursive-doubling-latency"
  "recursive-doubling-bandwidth"
  "swing-latency"
  "swing-bandwidth"
)

# Optional built-in baselines. Uncomment if needed.
# ALGOS_9+=("builtin")
# ALGOS_8+=("builtin")

rm -f "${CSV_FILE}"

echo "algo,world_size,hidden_dim,num_layers,model_label,total_params,grad_size_bytes,grad_size_mib,steps,batch_size,bucket_cap_mb,threads,repeat_id,total_training_time_sec,avg_step_time_ms,normalized_total_training_time_sec,normalized_avg_step_time_ms,parameter_max_diff" > "${CSV_FILE}"

next_port() {
python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

run_one() {
  local algo="$1"
  local nproc="$2"
  local hidden_dim="$3"
  local num_layers="$4"
  local model_label="$5"
  local repeat_id="$6"

  local port
  port=$(next_port)

  local log_file="${LOG_DIR}/${algo}_n${nproc}_${model_label}_rep${repeat_id}.log"

  echo "============================================================"
  echo "Running algo=${algo}, nproc=${nproc}, model=${model_label}, repeat=${repeat_id}, port=${port}"
  echo "============================================================"

  OMP_NUM_THREADS=${THREADS} \
  torchrun \
    --nnodes=1 \
    --nproc_per_node="${nproc}" \
    --master_addr=127.0.0.1 \
    --master_port="${port}" \
    train_cpu_ddp_custom.py \
      --algo "${algo}" \
      --steps "${STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --hidden-dim "${hidden_dim}" \
      --num-layers "${num_layers}" \
      --bucket-cap-mb "${BUCKET_CAP_MB}" \
      --threads "${THREADS}" \
    2>&1 | tee "${log_file}"

  local total_params
  local total_time
  local avg_step
  local max_diff

  total_params=$(grep -oP "total_params=\K[0-9]+" "${log_file}" | tail -n 1 || echo "NA")
  total_time=$(grep -oP "total_training_time_sec=\K[0-9.]+" "${log_file}" | tail -n 1 || echo "NA")
  avg_step=$(grep -oP "avg_step_time_ms=\K[0-9.]+" "${log_file}" | tail -n 1 || echo "NA")
  max_diff=$(grep -oP "parameter_max_diff_across_ranks=\K[0-9.eE+-]+" "${log_file}" | tail -n 1 || echo "NA")

  local grad_size_bytes="NA"
  local grad_size_mib="NA"
  local norm_total="NA"
  local norm_avg="NA"

  if [[ "${total_params}" != "NA" ]]; then
    grad_size_bytes=$(( total_params * 4 ))
    grad_size_mib=$(python - <<PY
print(f"{${grad_size_bytes} / 1024 / 1024:.6f}")
PY
)
  fi

  # Normalize 8-rank algorithms to 9 ranks using x 9/8.
  # 9-rank algorithms keep raw values.
  if [[ "${total_time}" != "NA" && "${avg_step}" != "NA" ]]; then
    if [[ "${nproc}" == "8" ]]; then
      norm_total=$(python - <<PY
print(f"{float('${total_time}') * 9.0 / 8.0:.6f}")
PY
)
      norm_avg=$(python - <<PY
print(f"{float('${avg_step}') * 9.0 / 8.0:.6f}")
PY
)
    else
      norm_total="${total_time}"
      norm_avg="${avg_step}"
    fi
  fi

  echo "${algo},${nproc},${hidden_dim},${num_layers},${model_label},${total_params},${grad_size_bytes},${grad_size_mib},${STEPS},${BATCH_SIZE},${BUCKET_CAP_MB},${THREADS},${repeat_id},${total_time},${avg_step},${norm_total},${norm_avg},${max_diff}" >> "${CSV_FILE}"
}

for repeat_id in $(seq 1 "${REPEATS}"); do
  for config in "${MODEL_CONFIGS[@]}"; do
    read -r hidden_dim num_layers model_label <<< "${config}"

    for algo in "${ALGOS_9[@]}"; do
      run_one "${algo}" 9 "${hidden_dim}" "${num_layers}" "${model_label}" "${repeat_id}"
    done

    for algo in "${ALGOS_8[@]}"; do
      run_one "${algo}" 8 "${hidden_dim}" "${num_layers}" "${model_label}" "${repeat_id}"
    done
  done
done

echo "All experiments finished."
echo "CSV saved to ${CSV_FILE}"
echo "Logs saved to ${LOG_DIR}/"