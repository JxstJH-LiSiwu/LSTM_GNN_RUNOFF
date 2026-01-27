#!/bin/bash
set -euo pipefail

# ===================== 路径配置 =====================
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

cd "${WORKDIR}"

# ===================== GPU 配置 =====================
GPUS=(0 1 2 3 4 5 6)
NUM_WORKERS=5

# ===================== 实验配置 =====================
LOSSES=(0.0 0.05 0.1)
MODELS=("LSTM" "LSTM-GAT" "LSTM-GCN" "LSTM-Cheb" "LSTM-GraphSAGE")
HOPS=(2 3 4)
LEAD_DAYS=(1 2 3 4 5 6 7)

LOG_DIR="checkpoints/logs/fifo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# ===================== 构建 FIFO 队列 =====================
TASKS=()
for loss in "${LOSSES[@]}"; do
  for model in "${MODELS[@]}"; do
    for hop in "${HOPS[@]}"; do
      for lead in "${LEAD_DAYS[@]}"; do
        TASKS+=("${loss}|${model}|${hop}|${lead}")
      done
    done
  done
done

TOTAL_TASKS=${#TASKS[@]}
echo "Total tasks: ${TOTAL_TASKS}"
echo "GPUs: ${GPUS[*]}"
echo "Logs: ${LOG_DIR}"
echo

# ===================== 状态 =====================
task_idx=0
declare -A GPU_PID   # gpu -> pid

# ===================== GPU 空闲检测 =====================
gpu_is_free () {
  local gpu=$1
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  local cnt
  cnt=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | awk -v g="${gpu}" 'BEGIN{c=0} {c++} END{print c+0}')
  if [ "${cnt}" -eq 0 ]; then
    return 0
  fi
  local gpu_uuid
  gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
    | awk -F', ' -v g="${gpu}" '$1==g{print $2}')
  if [ -z "${gpu_uuid}" ]; then
    return 0
  fi
  local used
  used=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
    | awk -v u="${gpu_uuid}" '$1==u{c++} END{print c+0}')
  if [ "${used}" -eq 0 ]; then
    return 0
  fi
  return 1
}

# ===================== 启动任务函数 =====================
launch_task () {
  local gpu=$1
  local loss=$2
  local model=$3
  local hop=$4
  local lead=$5

  local tag="mlw_${loss//./_}_hop${hop}_lead${lead}"
  local log="${LOG_DIR}/${model}_${tag}_gpu${gpu}.log"

  echo "[START] GPU ${gpu} | loss=${loss} | model=${model} | hop=${hop} | lead=${lead}"

  CUDA_VISIBLE_DEVICES=${gpu} \
  HOP="${hop}" \
  LEAD_DAYS="${lead}" \
  MODEL_NAME="${model}" \
  MEAN_LOSS_WEIGHT="${loss}" \
  RUN_TAG="${tag}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  ${PYTHON} train.py > "${log}" 2>&1 &

  GPU_PID[${gpu}]=$!
}

# ===================== 初始填满 GPU（仅空闲） =====================
for gpu in "${GPUS[@]}"; do
  if [ ${task_idx} -ge ${TOTAL_TASKS} ]; then
    break
  fi
  if gpu_is_free "${gpu}"; then
    IFS="|" read -r loss model hop lead <<< "${TASKS[task_idx]}"
    launch_task "${gpu}" "${loss}" "${model}" "${hop}" "${lead}"
    task_idx=$((task_idx + 1))
  fi
done

# ===================== 轮询调度（核心） =====================
while true; do
  any_running=false

  for gpu in "${!GPU_PID[@]}"; do
    pid=${GPU_PID[$gpu]}

    # 进程还在跑
    if kill -0 "${pid}" 2>/dev/null; then
      any_running=true
      continue
    fi

    # 进程结束了
    wait "${pid}" 2>/dev/null || true
    unset GPU_PID[$gpu]

    # 如果还有任务，且 GPU 空闲，立刻补上
    if [ ${task_idx} -lt ${TOTAL_TASKS} ]; then
      if gpu_is_free "${gpu}"; then
        IFS="|" read -r loss model hop lead <<< "${TASKS[task_idx]}"
        launch_task "${gpu}" "${loss}" "${model}" "${hop}" "${lead}"
        task_idx=$((task_idx + 1))
        any_running=true
      fi
    fi
  done

  # 如果没有任务在跑，但还有任务，等待空闲 GPU 再投递
  if [ "${any_running}" = false ] && [ ${task_idx} -lt ${TOTAL_TASKS} ]; then
    for gpu in "${GPUS[@]}"; do
      if [ ${task_idx} -ge ${TOTAL_TASKS} ]; then
        break
      fi
      if [ -z "${GPU_PID[$gpu]:-}" ] && gpu_is_free "${gpu}"; then
        IFS="|" read -r loss model hop lead <<< "${TASKS[task_idx]}"
        launch_task "${gpu}" "${loss}" "${model}" "${hop}" "${lead}"
        task_idx=$((task_idx + 1))
        any_running=true
      fi
    done
  fi

  # 所有任务完成
  if [ ${task_idx} -ge ${TOTAL_TASKS} ] && [ "${#GPU_PID[@]}" -eq 0 ]; then
    break
  fi

  # 避免 busy wait
  sleep 2
done

echo
echo "✅ All FIFO tasks finished."
echo "Logs saved in: ${LOG_DIR}"
