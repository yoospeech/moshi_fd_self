#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

exec /home/ysw/Documents/anaconda3/envs/moshi_finetune/bin/python \
  train_full_duplex.py \
  --task finetuning \
  --natural-monologue \
  --mimi-device cuda:0 \
  --folders '/home/ysw/Documents/DB/behavior-sd:en' \
  --enable-sine-conditioning \
  --sine-frequency 440 \
  --sine-amplitude 0.5 \
  --model models/LFM3-1.7B \
  --trust-remote-code \
  --disable-lfm3-triton \
  --num-codebooks 4 \
  --context 1500 \
  --assistant-text-delay-steps 0 \
  --acoustic-delay 1 \
  --depformer-dim 1024 \
  --depformer-heads 16 \
  --depformer-layers 6 \
  --depformer-ffn-dim 4096 \
  --audio-loss-weight 0.1 \
  --text-loss-weight 1.0 \
  --semantic-loss-weight 100 \
  --audio-token-balance-alpha 1.0 \
  --text-padding-weight 0.5 \
  --audio-text-mask-probability 0.3 \
  --batch-size 2 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --temporal-learning-rate 2e-5 \
  --min-duration 5 \
  --max-duration 200 \
  --min-text-tokens 5 \
  --valid-ratio 0.001 \
  --valid-every 500 \
  --valid-batches 54 \
  --log-every 50 \
  --save-every 500 \
  --log-timing \
  --print-model \
  --steps 100000 \
  --output runs/LFM3-1.7B_behavior_sd_sine_macro
