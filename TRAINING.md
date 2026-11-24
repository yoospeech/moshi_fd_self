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

## Mono post-training

Duration limits apply to the recording. Mono post-training and stereo fine-tuning
both use `--min-text-tokens` (default 5), requiring at least that many lexical
tokenizer pieces in the assistant target (the complete utterance for mono),
using the same word-by-word encoding as the text targets, excluding PAD/EPAD.
Set it to 0 to disable the token minimum. Valid word boundaries and actual BPE
frame collision checks are still required,
including after online alignment. Rejected lazy samples are replaced by the next
valid sample.

`MonoPostTrainingDataset` uses the complete mono utterance as assistant audio and
text, with an absent user-audio stream:

```text
silence -> user audio
full word-aligned utterance -> assistant text and audio
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
  --min-duration 5 --max-duration 100 --min-text-tokens 5 \
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
timestamps and unavailable word boundaries are rejected. Overlapping BPE
pieces are shifted to the next free text frame instead of rejecting the sample.
Audio and source timestamps remain unchanged. Later gaps absorb the shift;
text extending beyond the audio is retained with audio targets masked out.
There is currently no maximum shift limit, so dense speech can accumulate delay.
Use `--alignment-mode sidecar` to require pre-existing JSON files.

Skipped online samples report their failure stage and original error. The stages
are `transcription_failed`, `empty_alignment`, `timestamp_validation_failed`,
and `text_tokens_failed`. CUDA out-of-memory errors remain fatal rather than
being treated as invalid samples. A final word ending no more than 100 ms past
the WAV boundary is clamped to the actual duration; larger overruns and
out-of-range words before the final word are rejected.

Folder language labels are passed to Whisper and returned in
`batch["languages"]`. They do not balance languages or condition the model.
Use `--alignment-device cuda:0` only when the GPU has room for Whisper,
the backbone and Mimi together.

## Stereo dialogue fine-tuning

`StereoDialogueDataset` reads stereo user/assistant audio and assistant word
alignment sidecars.

The same recursive folder loader used by mono post-training can read native
stereo WAVs. It uses assistant channel 0 and user channel 1 and preserves the
complete stereo timeline. With online alignment, each sample is transcribed
lazily the first time its DDP rank loads it; no manifest or sidecar is required:

```bash
PYTHONPATH=. python train_full_duplex.py \
  --task finetuning \
  --folders '/path/to/behavior-sd/validation:en' \
  --alignment-mode online \
  --alignment-model turbo --alignment-device local \
  --model models/SmolLM2-135M \
  --batch-size 4 --steps 1000 \
  --output runs/stereo_finetuning
```

```bash
PYTHONPATH=. python train_full_duplex.py \
  --task finetuning \
  --manifest /path/to/stereo_dialogue.jsonl \
  --batch-size 4 --steps 1000 \
  --output runs/stereo_finetuning
```

Relative WAV paths resolve from the manifest directory. Channel defaults are
assistant 0 and user 1, and can be overridden per manifest row. Filtering
checks duration, minimum assistant BPE tokens, timestamps and speech coverage.
BPE frame overlaps are resolved by shifting text forward.

## Model and distributed execution

`LFM3-1.7B` is the default model. `SmolLM2-135M` is also available as a known
alias under `models/`. Missing aliases are downloaded automatically; local model
paths and Hugging Face repository IDs are also accepted. SmolLM2-135M has been
verified end to end in this workspace; LFM3 still needs a separate runtime check.

Example stereo DDP launch:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 train_full_duplex.py \
  --model LFM3-1.7B --task finetuning \
  --folders '/path/to/behavior-sd/validation:en' \
  --alignment-mode online \
  --alignment-model turbo --alignment-device local \
  --batch-size 4 --grad-accum 2 --steps 1000 \
  --output runs/stereo_ddp
```

Batch size is per rank, so the effective batch above is
`2 GPUs * 4 samples * 2 accumulation = 16`. Do not pass `--mimi-device` under
DDP. `--alignment-device local` resolves to `cuda:LOCAL_RANK`, so every rank
loads its own Whisper model on the same GPU as its backbone and Mimi. Use a
smaller batch or CPU alignment if those three models do not fit together.

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
