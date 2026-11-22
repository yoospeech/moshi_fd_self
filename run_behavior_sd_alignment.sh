#!/usr/bin/env bash
# SPDX-FileContributor: OpenAI
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ysw/Documents/anaconda3/envs/sdialogue/bin/python"
DATA_DIR="/home/ysw/Documents/DB/behavior-sd/validation"
LOG_DIR="${SCRIPT_DIR}/alignment_logs"
NUM_GPUS=2
NUM_SHARDS=8

mkdir -p "${LOG_DIR}"

for ((shard = 0; shard < NUM_SHARDS; shard++)); do
  gpu=$((shard % NUM_GPUS))
  echo "Starting shard ${shard}/${NUM_SHARDS} on GPU ${gpu}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/annotate_behavior_sd.py" \
    "${DATA_DIR}" \
    --model medium \
    --language en \
    --assistant-channel 0 \
    --gpu "${gpu}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard}" \
    >"${LOG_DIR}/shard_${shard}.log" 2>&1 &
done

wait
echo "All ${NUM_SHARDS} alignment shards completed."
