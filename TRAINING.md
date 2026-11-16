# Moshi Training Guide

Return to [README](README.md). For decoding, see [INFERENCE.md](INFERENCE.md).

Both tasks use the same batch format and predict assistant text and audio.
User audio is conditioning only and is never an audio loss target.

## Streams

With four Mimi codebooks:

```text
streams: [assistant_text, assistant_audio(q0..q3), user_audio(q0..q3)]
delays:  [0,              0,1,1,1,                0,1,1,1]
```

Semantic codebooks use the current 12.5 Hz frame; acoustic codebooks lag one
frame. Assistant timestamps become a PAD/EPAD text timeline. EPAD precedes a
word, PAD covers non-lexical frames, and user Mimi tokens provide the acoustic
prompt. Assistant audio is predicted through the Depth Transformer.

`--text-only-fraction` removes acoustic streams from a portion of each batch
while retaining assistant text targets. Use `0` for fully acoustic batches.

## Mono continuation post-training

`MonoContinuationDataset` turns one mono utterance into a synthetic turn:

```text
word-aligned prefix -> user audio
word-aligned suffix -> assistant text and audio
```

Pass comma-separated `folder:Whisper-language-code` inputs. No combined input
manifest is created.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. \
/data/util/anaconda3/envs/duplexomni/bin/python train_full_duplex.py \
  --task mono_post_training \
  --folders '/data/db/LibriTTS_R:en' \
  --alignment-mode online \
  --alignment-model medium --alignment-device cpu \
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
```

For multiple languages, use comma-separated folder and Whisper language-code
pairs, for example `--folders '/data/ko:ko,/data/en:en,/data/hi:hi,/data/id:id'`.

Online alignment runs lazily when the DataLoader first accesses each sample.
The Whisper model is loaded on first use and kept in the training process;
online alignment therefore uses `num_workers=0` and may use an
`--alignment-device` such as `cuda:0`. Valid adjacent JSON sidecars are reused;
missing or invalid alignments are extracted into memory and recomputed after
restart. Empty alignments, invalid
timestamps, unavailable word boundaries and BPE collisions are rejected.
Use `--alignment-mode sidecar` to require pre-existing JSON files.

Folder language labels are passed to Whisper and returned in
`batch["languages"]`. They do not balance languages or condition the model.
Use `--alignment-device cuda:0` only when the GPU has room for Whisper,
the backbone and Mimi together.

## Stereo dialogue fine-tuning

`StereoDialogueDataset` reads stereo user/assistant audio and assistant word
alignment sidecars.

```bash
PYTHONPATH=. python train_full_duplex.py \
  --task finetuning \
  --manifest /path/to/stereo_dialogue.jsonl \
  --batch-size 4 --steps 1000 \
  --output runs/stereo_finetuning
```

Relative WAV paths resolve from the manifest directory. Channel defaults are
assistant 0 and user 1, and can be overridden per manifest row. Filtering
checks duration, word rate, timestamps, speech coverage and BPE collisions.

## Model and distributed execution

`LFM3-1.7B` is the default model. `SmolLM2-135M` is also available as a known
alias under `models/`. Missing aliases are downloaded automatically; local model
paths and Hugging Face repository IDs are also accepted. SmolLM2-135M has been
verified end to end in this workspace; LFM3 still needs a separate runtime check.

Example stereo DDP launch:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 train_full_duplex.py \
  --model LFM3-1.7B --task finetuning \
  --manifest /path/to/stereo_dialogue.jsonl \
  --batch-size 4 --grad-accum 2 --steps 1000 \
  --output runs/stereo_ddp
```

Batch size is per rank. Do not pass `--mimi-device` under DDP. Online mono
alignment currently repeats on every rank; for multi-GPU mono training, create
sidecars first and use `--alignment-mode sidecar`.

## Losses and metrics

| Option | Default | Meaning |
|---|---:|---|
| `--text-loss-weight` | 1.0 | Assistant text loss |
| `--audio-loss-weight` | 0.1 | Assistant audio loss |
| `--semantic-loss-weight` | 100.0 | Semantic q0 weight within audio loss |
| `--text-padding-weight` | 0.5 | PAD weight relative to other text targets |

`assistant_np_accuracy` excludes PAD but includes EPAD;
`assistant_all_accuracy` includes PAD. Audio accuracy is logged per Mimi
codebook as `assistant_rq_accuracy_1` through `assistant_rq_accuracy_4`.
