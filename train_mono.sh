#!/bin/bash

CUDA_VISIBLE_DEVICES=0
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. \
python train_full_duplex.py \
  --task mono_post_training \
  --folders '/home/ysw/Documents/DB/libriTTS/LibriTTS_R:en' \
  --alignment-mode online \
  --alignment-model medium --alignment-device cuda:0 \
  --model models/LFM3-1.7B --trust-remote-code \
  --num-codebooks 4 \
  --assistant-text-delay-steps 0 --acoustic-delay 1 \
  --depformer-dim 1024 --depformer-heads 16 \
  --depformer-layers 6 --depformer-ffn-dim 4096 \
  --audio-loss-weight 0.1 --text-loss-weight 1.0 \
  --text-padding-weight 0.5 --text-only-fraction 0.5 \
  --batch-size 4 --grad-accum 1 --steps 100000 \
  --learning-rate 2e-4 --temporal-learning-rate 2e-5 \
  --min-duration 5 --max-duration 100 --min-words 5 \
  --min-words-per-second 0.5 --max-words-per-second 2.5 \
  --valid-ratio 0.05 --valid-every 500 --valid-batches 54 \
  --log-every 10 --save-every 500 --print-model \
  --output runs/LFM3-1.7B_mono_post_training
