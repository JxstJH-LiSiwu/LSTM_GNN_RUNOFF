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

LOG_DIR="checkpoints/logs/fifo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# ===================== 构建 FIFO 队列 =====================
TASKS=()
for loss in "${LOSSES[@]}"; do
  for model in "${MODELS[@]}"; do
    TASKS+=("${loss}|${model}")
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

# ===================== 启动任务函数 =====================
launch_task () {
  local gpu=$1
  local loss=$2
  local model=$3

  local tag="mlw_${loss//./_}"
  local log="${LOG_DIR}/${model}_${tag}_gpu${gpu}.log"

  echo "[START] GPU ${gpu} | loss=${loss} | model=${model}"

  CUDA_VISIBLE_DEVICES=${gpu} \
  MODEL_NAME="${model}" \
  MEAN_LOSS_WEIGHT="${loss}" \
  RUN_TAG="${tag}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  ${PYTHON} train.py > "${log}" 2>&1 &

  GPU_PID[${gpu}]=$!
}

# ===================== 初始填满 GPU =====================
for gpu in "${GPUS[@]}"; do
  if [ ${task_idx} -ge ${TOTAL_TASKS} ]; then
    break
  fi

  IFS="|" read -r loss model <<< "${TASKS[task_idx]}"
  launch_task "${gpu}" "${loss}" "${model}"
  task_idx=$((task_idx + 1))
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

    # 如果还有任务，立刻补上
    if [ ${task_idx} -lt ${TOTAL_TASKS} ]; then
      IFS="|" read -r loss model <<< "${TASKS[task_idx]}"
      launch_task "${gpu}" "${loss}" "${model}"
      task_idx=$((task_idx + 1))
      any_running=true
    fi
  done

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
