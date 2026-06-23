#!/usr/bin/env bash
set -euo pipefail

export GLOO_SOCKET_IFNAME=lo
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

CSV_FILE="training_completion_results.csv"
LOG_DIR="logs_training_bench"
mkdir -p "${LOG_DIR}"

STEPS=300
BATCH_SIZE=16
THREADS=1

# Keep this large if you want one large DDP bucket for most model sizes.
# If Trivance 16nodes version becomes too memory-heavy, try BUCKET_CAP_MB=1 or 4.
BUCKET_CAP_MB=512

REPEATS=5
NPROC=16

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

# Only test Ring, Recursive Doubling, and Trivance with 16 ranks.
ALGOS_16=(
  "ring"
  "recursive-doubling-latency"
  "recursive-doubling-bandwidth"
  "trivance-latency"
  "trivance-bandwidth"
)

rm -f "${CSV_FILE}"

echo "algo,world_size,hidden_dim,num_layers,model_label,total_params,grad_size_bytes,grad_size_mib,steps,batch_size,bucket_cap_mb,threads,repeat_id,total_training_time_sec,avg_step_time_ms,parameter_max_diff" > "${CSV_FILE}"

next_port() {
python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
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

  # Do not let a torchrun cleanup SIGABRT stop the whole benchmark immediately.
  # Some runs may print valid metrics and then return non-zero during process cleanup.
  set +e
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
  local run_status=${PIPESTATUS[0]}
  set -e

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

  if [[ "${total_params}" != "NA" ]]; then
    grad_size_bytes=$(( total_params * 4 ))
    grad_size_mib=$(python - <<PY
print(f"{${grad_size_bytes} / 1024 / 1024:.6f}")
PY
)
  fi

  local accepted="no"
  if [[ "${total_time}" != "NA" && "${avg_step}" != "NA" && "${max_diff}" != "NA" ]]; then
    accepted="yes"
  fi

  if [[ "${run_status}" -ne 0 && "${accepted}" == "yes" ]]; then
    echo "WARNING: torchrun exited with status ${run_status}, but metrics were found. Treating this run as completed."
  elif [[ "${run_status}" -ne 0 && "${accepted}" != "yes" ]]; then
    echo "ERROR: torchrun exited with status ${run_status} before producing complete metrics. Recording NA and continuing."
  fi

  echo "${algo},${nproc},${hidden_dim},${num_layers},${model_label},${total_params},${grad_size_bytes},${grad_size_mib},${STEPS},${BATCH_SIZE},${BUCKET_CAP_MB},${THREADS},${repeat_id},${total_time},${avg_step},${max_diff}" >> "${CSV_FILE}"

  echo ""
  echo "==================== FINISHED RESULT ===================="
  echo "algo=${algo}"
  echo "world_size=${nproc}"
  echo "model=${model_label}"
  echo "total_params=${total_params}"
  echo "grad_size_mib=${grad_size_mib}"
  echo "repeat=${repeat_id}"
  echo "total_training_time_sec=${total_time}"
  echo "avg_step_time_ms=${avg_step}"
  echo "parameter_max_diff=${max_diff}"

  echo "csv=${CSV_FILE}"
  echo "========================================================="
  echo ""

  echo "Recent finished results:"
  tail -n 10 "${CSV_FILE}"
  echo ""

  # Give Gloo/torchrun a short pause to release local resources cleanly.
  sleep 2
}

for repeat_id in $(seq 1 "${REPEATS}"); do
  for config in "${MODEL_CONFIGS[@]}"; do
    read -r hidden_dim num_layers model_label <<< "${config}"

    for algo in "${ALGOS_16[@]}"; do
      run_one "${algo}" "${NPROC}" "${hidden_dim}" "${num_layers}" "${model_label}" "${repeat_id}"
    done
  done
done

echo "All experiments finished."
echo "CSV saved to ${CSV_FILE}"
echo "Logs saved to ${LOG_DIR}/"
