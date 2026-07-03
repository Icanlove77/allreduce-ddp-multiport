#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CSV_FILE=${CSV_FILE:-gpu_bidirectional_ring_small_vector_results.csv}
LOG_DIR=${LOG_DIR:-logs_gpu_bidirectional_ring_small_vector}
mkdir -p "${LOG_DIR}"

TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_gpu_ddp_custom.py}
STEPS=${STEPS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}
THREADS=${THREADS:-1}
BUCKET_CAP_MB=${BUCKET_CAP_MB:-25}
REPEATS=${REPEATS:-3}
NPROC_7=${NPROC_7:-7}
NUM_CLASSES=2

if (( NPROC_7 % 2 == 0 )); then
  echo "ERROR: bidirectional-ring requires an odd world_size. Use NPROC_7=7 on an 8-GPU pod."
  exit 1
fi

MODEL_CONFIGS=(
  "251 1 1 1KiB 1024"
  "507 1 1 2KiB 2048"
  "1019 1 1 4KiB 4096"
  "2043 1 1 8KiB 8192"
  "4091 1 1 16KiB 16384"
  "8187 1 1 32KiB 32768"
  "16379 1 1 64KiB 65536"
  "32763 1 1 128KiB 131072"
  "65531 1 1 256KiB 262144"
  "131067 1 1 512KiB 524288"
  "262139 1 1 1MiB 1048576"
)

ALGOS_7=(
  "bidirectional-ring"
  "ring"
  "builtin"
)

rm -f "${CSV_FILE}"
echo "algo,world_size,input_dim,hidden_dim,num_classes,num_layers,model_label,target_vector_size_bytes,total_params,grad_size_bytes,grad_size_mib,steps,batch_size,bucket_cap_mb,threads,repeat_id,total_training_time_sec,avg_step_time_ms,parameter_max_diff,torchrun_status,accepted" > "${CSV_FILE}"

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
  local input_dim="$3"
  local hidden_dim="$4"
  local num_layers="$5"
  local model_label="$6"
  local target_vector_size_bytes="$7"
  local repeat_id="$8"

  local port
  port=$(next_port)

  local log_file="${LOG_DIR}/${algo}_n${nproc}_${model_label}_rep${repeat_id}.log"

  echo "============================================================"
  echo "Running algo=${algo}, nproc=${nproc}, model=${model_label}, repeat=${repeat_id}, port=${port}"
  echo "input_dim=${input_dim}, hidden_dim=${hidden_dim}, num_layers=${num_layers}, target_vector_size_bytes=${target_vector_size_bytes}"
  echo "============================================================"

  set +e
  torchrun \
    --nnodes=1 \
    --nproc_per_node="${nproc}" \
    --master_addr=127.0.0.1 \
    --master_port="${port}" \
    "${TRAIN_SCRIPT}" \
      --algo "${algo}" \
      --steps "${STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --input-dim "${input_dim}" \
      --hidden-dim "${hidden_dim}" \
      --num-classes "${NUM_CLASSES}" \
      --num-layers "${num_layers}" \
      --bucket-cap-mb "${BUCKET_CAP_MB}" \
      --threads "${THREADS}" \
    2>&1 | tee "${log_file}"
  local run_status=${PIPESTATUS[0]}
  set -e

  local total_params total_time avg_step max_diff accepted
  total_params=$(grep -oP "total_params=\K[0-9]+" "${log_file}" | tail -n 1 || echo "NA")
  total_time=$(grep -oP "total_training_time_sec=\K[0-9.]+" "${log_file}" | tail -n 1 || echo "NA")
  avg_step=$(grep -oP "avg_step_time_ms=\K[0-9.]+" "${log_file}" | tail -n 1 || echo "NA")
  max_diff=$(grep -oP "parameter_max_diff_across_ranks=\K[0-9.eE+-]+" "${log_file}" | tail -n 1 || echo "NA")

  if [[ "${total_time}" != "NA" && "${avg_step}" != "NA" && "${max_diff}" != "NA" ]]; then
    accepted="completed"
  else
    accepted="failed"
  fi

  local grad_size_bytes="NA"
  local grad_size_mib="NA"
  if [[ "${total_params}" != "NA" ]]; then
    grad_size_bytes=$(( total_params * 4 ))
    grad_size_mib=$(python - <<PY
print(f"{${grad_size_bytes} / 1024 / 1024:.6f}")
PY
)
  fi

  echo "${algo},${nproc},${input_dim},${hidden_dim},${NUM_CLASSES},${num_layers},${model_label},${target_vector_size_bytes},${total_params},${grad_size_bytes},${grad_size_mib},${STEPS},${BATCH_SIZE},${BUCKET_CAP_MB},${THREADS},${repeat_id},${total_time},${avg_step},${max_diff},${run_status},${accepted}" >> "${CSV_FILE}"

  echo "==================== FINISHED RESULT ===================="
  echo "algo=${algo} world_size=${nproc} model=${model_label} repeat=${repeat_id} status=${run_status} accepted=${accepted}"
  echo "total_params=${total_params} grad_size_mib=${grad_size_mib} avg_step_time_ms=${avg_step} parameter_max_diff=${max_diff}"
  echo "========================================================="

  if [[ "${accepted}" == "failed" ]]; then
    echo "ERROR: run failed before producing metrics. See ${log_file}"
    exit 1
  fi
}

for repeat_id in $(seq 1 "${REPEATS}"); do
  for config in "${MODEL_CONFIGS[@]}"; do
    read -r input_dim hidden_dim num_layers model_label target_vector_size_bytes <<< "${config}"

    for algo in "${ALGOS_7[@]}"; do
      run_one "${algo}" "${NPROC_7}" "${input_dim}" "${hidden_dim}" "${num_layers}" "${model_label}" "${target_vector_size_bytes}" "${repeat_id}"
    done
  done
done

echo "All GPU bidirectional-ring small-vector experiments finished."
echo "CSV saved to ${CSV_FILE}"
echo "Logs saved to ${LOG_DIR}/"
