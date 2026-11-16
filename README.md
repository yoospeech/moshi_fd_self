# Moshi Training

> Experimental research prototype.

Train a full-duplex model to predict assistant text and speech from user audio.
User audio is conditioning only and is never an audio loss target.

- [Training guide](TRAINING.md): architecture, mono post-training, stereo fine-tuning,
  online alignment, losses, timing, DDP and checkpoints.
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

## Mono continuation post-training

Split mono speech at a word boundary into a user prefix and assistant suffix.
Pass a comma-separated list of `folder:language` inputs using Whisper codes.

```bash
PYTHONPATH=. python train_full_duplex.py \
  --task mono_post_training \
  --folders '/data/db/LibriTTS_R:en' \
  --alignment-mode online --alignment-model medium --alignment-device cpu \
  --batch-size 4 --steps 1000 --output runs/mono_post_training
```

Missing or invalid word alignments are extracted during dataset initialization.
Existing valid sidecars are reused; newly extracted alignments stay in memory.
The first batch starts after extraction completes. See
[online alignment and execution details](TRAINING.md#mono-continuation-post-training).

## Stereo dialogue fine-tuning

Read existing stereo user/assistant audio and assistant timestamp sidecars.

```bash
PYTHONPATH=. python train_full_duplex.py \
  --task finetuning --manifest /path/to/stereo_dialogue.jsonl \
  --batch-size 4 --steps 1000 --output runs/stereo_finetuning
```

Both tasks use the same batch format and online Mimi encoding.
See [stereo input requirements](TRAINING.md#stereo-dialogue-fine-tuning).

## Verification

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python train_full_duplex.py \
  --model SmolLM2-135M --smoke-test --smoke-steps 1 \
  --batch-size 1 --text-only-fraction 0 --output runs/model_smoke
```

The model smoke uses synthetic codec tokens; it does not test Whisper or Mimi.
The verified real-data path is single-GPU SmolLM2-135M with Whisper medium
alignment and Mimi encoding. LFM3 is a selectable model alias but has not
been validated end-to-end in this environment.
See [bounded real-data verification](TRAINING.md#run-mono-post-training).

## Inference

```bash
PYTHONPATH=. python infer_full_duplex.py \
  --checkpoint runs/mono_post_training/checkpoint_00001000.pt \
  --audio /path/to/user.wav --user-channel 0 \
  --device cuda --tail-seconds 10 \
  --output inference_outputs/response.json \
  --audio-output inference_outputs/response.wav
```

Outputs contain assistant text and stereo audio (assistant channel 0, user
channel 1). No annotation sidecar is needed. See [inference details](INFERENCE.md).

## Repository layout

```text
full_duplex/model.py       model, streams, Depth Transformer and losses
full_duplex/dataset.py     stereo dialogue and mono continuation datasets
full_duplex/train.py       single-GPU/DDP training, validation and timing
full_duplex/inference.py   user-conditioned assistant text/audio generation
scripts/annotate.py        word alignment extraction
train_full_duplex.py       training CLI
infer_full_duplex.py       inference CLI
tests/                    regression tests
```
