# Moshi Training

> Experimental research prototype.

Train a full-duplex model to predict assistant text and speech from user audio.

- [Training guide](TRAINING.md): architecture, mono post-training, stereo fine-tuning,
  online alignment, losses and DDP.
- [Inference guide](INFERENCE.md): checkpoint loading, decoding, channels and outputs.

## Setup

Run commands from the repository root with the intended Python environment active.

```bash
git clone --recurse-submodules https://github.com/yoospeech/moshi_fd_self.git
cd moshi_fd_self
python -m pip install -r requirements.txt
python -m pip install --no-deps -e thirdparty/moshi/moshi
```

The verified environment uses Python 3.11, transformers 4.57.6,
huggingface-hub 0.36.2, moshi 0.2.13 and whisper-timestamped 1.15.9.
The local interpreter used for verification is
`/data/util/anaconda3/envs/duplexomni/bin/python`.
Moshi 0.2.13 requires Hub below 1.0; transformers 5.x is incompatible with
that requirement. Prefer an isolated environment for this project.

## Train

Mono continuation post-training splits each utterance at a Whisper word
boundary and trains user audio to assistant text/audio. `LFM3-1.7B` is the
default backbone.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. \
/data/util/anaconda3/envs/duplexomni/bin/python train_full_duplex.py \
  --task mono_post_training \
  --folders '/data/db/LibriTTS_R:en' \
  --alignment-mode online \
  --alignment-model medium --alignment-device cpu \
  --model models/LFM3-1.7B --trust-remote-code \
  --audio-loss-weight 0.1 --text-loss-weight 1.0 \
  --text-padding-weight 0.5 --text-only-fraction 0.5 \
  --batch-size 4 --grad-accum 1 --steps 100000 \
  --learning-rate 2e-4 --temporal-learning-rate 2e-5 \
  --valid-ratio 0.05 --valid-every 500 --valid-batches 54 \
  --log-every 10 --save-every 500 \
  --output runs/LFM3-1.7B_mono_post_training
```

For multiple languages, pass comma-separated inputs such as
`--folders '/data/ko:ko,/data/en:en,/data/hi:hi,/data/id:id'`. See the
[training guide](TRAINING.md) for stereo fine-tuning, filtering and DDP.

## Repository layout

```text
full_duplex/model.py       model, streams, Depth Transformer and losses
full_duplex/dataset.py     stereo dialogue and mono continuation datasets
full_duplex/train.py       single-GPU/DDP training, validation and timing
full_duplex/inference.py   user-conditioned assistant text/audio generation
scripts/annotate.py        word alignment extraction
train_full_duplex.py       training CLI
infer_full_duplex.py       inference CLI
tests/                     regression tests
```
