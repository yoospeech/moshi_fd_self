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
