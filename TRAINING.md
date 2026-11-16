# Moshi Training Guide

Return to [README](README.md). For decoding, see [INFERENCE.md](INFERENCE.md).

Both tasks optimize assistant text and assistant audio. User speech is
conditioning only; a manifest's `user_hyp` is not a target.
`StereoDialogueDataset` and `MonoContinuationDataset` share
`collate_duplex`: `waveforms`, `assistant_waveforms`,
`assistant_text_tokens`, frame lengths, IDs and language metadata.

## Streams and timing

```text
streams: [assistant_text, assistant_audio(q0..q3), user_audio(q0..q3)]
delays:  [0,              0,1,1,1,          0,1,1,1]
```

Assistant text and both semantic audio codebooks are placed at the current
12.5 Hz frame. The three acoustic codebooks in each audio stream lag by one
frame (80 ms), following Moshi's semantic/acoustic interleaving. Assistant
audio is teacher-forced only through its causal past; its current frame is the
Depth prediction target. User audio is prompt conditioning and has no loss.

Assistant word timestamps are converted to the original Moshi PAD/EPAD
timeline. EPAD is placed immediately before a word unless that would overwrite
a lexical token. No speaker BEGIN/END tokens are used.

### Causal learning rhythm: PAD, audio context, then lexical text

A useful way to understand the aligned sequence is `PAD -> audio context ->
EPAD -> non-PAD text`. This is a recurring causal rhythm on the shared audio
clock, not four separately scheduled training stages:

```text
time frame       t-2          t-1          t            t+1
assistant text   PAD          EPAD         lexical      PAD / next token
audio context    Mimi tokens  Mimi tokens  Mimi tokens  Mimi tokens
```

During silent or listening frames, PAD keeps the assistant text stream aligned
with the continuously advancing Mimi audio streams. The Temporal Transformer
therefore learns the acoustic state while the text target remains PAD. EPAD at
the frame immediately before a word is a learned boundary signal that says a
lexical token is about to begin. The following non-PAD token is then predicted
from the causal history containing the user prompt, past assistant audio, PAD
timing, and EPAD boundary. Assistant audio is predicted in parallel through the
Depth Transformer and is fed back only as past context at later frames.

Thus PAD is not merely ignored blank space: it teaches when not to emit text,
EPAD teaches when speech/text emission is about to start, and non-PAD tokens
teach what to say. User audio participates only in that conditioning history;
it is never an optimization target.

Training treats user Mimi tokens as an acoustic prompt and optimizes assistant
text cross-entropy plus assistant Mimi codebook cross-entropy. User Mimi tokens
are never audio targets. Offline inference force-feeds user Mimi tokens,
autoregressively samples assistant text and audio, feeds generated assistant
audio back into later Temporal frames, and decodes it to a WAV file.

Each training batch mixes acoustic-conditioned and text-only examples. With
the documented `--batch-size 4 --text-only-fraction 0.5`, every rank receives
two examples with the user-audio prompt and two examples with the acoustic
streams removed. Their positions are sampled randomly in every batch; both
groups retain the assistant text target. The text-only half is replay from the
paired assistant-text dataset, not a separate external text corpus, and helps
keep the pretrained text model active while adapting it to acoustic prompts.


## Stereo dialogue fine-tuning

Use `--task finetuning` to select `StereoDialogueDataset`. It reads existing
user/assistant audio channels and assistant word alignments.

```bash
PYTHONPATH=. python3 train_full_duplex.py \
  --task finetuning \
  --manifest /path/to/stereo_dialogue.jsonl \
  --output runs/stereo_finetuning
```

Use the original manifest:

```text
/data/db/daily-talk-contiguous/dailytalk.jsonl
```

For a row whose audio path is `data_stereo/406.wav`, user audio is read from
WAV channel 1 by default and assistant response timestamps are read from
`data_stereo/406.json`. `user_hyp`, `user_406.json`, and Whisper transcripts
are not used. Relative audio paths are resolved from the manifest directory.

Default filtering keeps samples with duration 5–200 seconds, at least 5
assistant words, 0.5–2.5 assistant words/second, valid timestamps, 5–80%
assistant timestamp coverage, and no lexical BPE collision at 12.5 Hz.

## Mono continuation post-training

Use `--task mono_post_training` to select `MonoContinuationDataset`.
Its manifest rows need `path` and `duration`; the loader reads Whisper word
timestamps from an adjacent sidecar by default (`utterance.wav` ->
`utterance.json`). It chooses a word-end timestamp as the split point, so it
never divides a word between the synthetic user prefix and assistant suffix.
Rows without valid, nonempty word alignment are rejected. Assistant text
timeline collisions are filtered before training.

Pass folders and their languages directly:

```bash
# mono WAV -> user prefix + assistant suffix text/audio
PYTHONPATH=. /data/util/anaconda3/envs/duplexomni/bin/python \
  train_full_duplex.py \
  --task mono_post_training \
  --folders '/data/db/LibriTTS_R:en,/path/to/korean_corpus:ko'

```

For mono post-training, the loader recursively discovers mono WAV files directly
from `--folders`; each folder is paired with its language using
`folder:language`. No combined input manifest is created. By default, missing,
empty or invalid word alignments are extracted using `annotate.process_one`
during dataset initialization and kept in memory (no JSON writes). Existing
valid sidecars are reused. Use `--alignment-mode sidecar` to disable extraction.
Configure Whisper using `--alignment-model medium --alignment-device cuda:0`
(the default device is CPU). Extraction completes before the first training
batch; each distributed rank initializes its own alignments, and results without
sidecars are recomputed on restart. The loader produces the same
`waveforms`, `assistant_waveforms`, and `assistant_text_tokens` batch fields as
fine-tuning, and Mimi encoding remains online and batched in the trainer.

`StereoDialogueDataset` loads stereo conversations for `finetuning`;
`MonoContinuationDataset` constructs synthetic turns for `mono_post_training`.
Both use the shared `collate_duplex` batch format. DailyTalk is one supported
source corpus, not a requirement of the dataset class.

### Run mono post-training

The verified environment uses Python 3.11, transformers 4.57.6,
huggingface-hub 0.36.2, moshi 0.2.13 and whisper-timestamped 1.15.9.
Moshi 0.2.13 requires huggingface-hub below 1.0; do not install transformers
5.x into this environment. A separate environment is recommended when other
projects require newer Hub versions.

```bash
PYTHONPATH=. /data/util/anaconda3/envs/duplexomni/bin/python \
  train_full_duplex.py \
  --model SmolLM2-135M \
  --task mono_post_training \
  --folders '/data/db/LibriTTS_R:en' \
  --alignment-mode online --alignment-model medium --alignment-device cpu \
  --batch-size 4 --steps 1000 --output runs/mono_post_training
```

Add languages as `/data/korean:ko,/data/english:en,/data/hindi:hi`;
Indonesian is `id`. Folder labels are supplied to Whisper and returned as
`batch['languages']`; they do not implement language-balanced sampling or
model language conditioning. Use the correct language for each folder.
The GPU can be used for alignment with `--alignment-device cuda:0`, provided
there is space alongside the already-loaded backbone and Mimi.

This is initialization-time extraction, not per-batch Whisper inference.
All selected files are scanned and aligned before the first batch. Existing
sidecars are reused, while newly extracted words stay in RAM. DDP ranks each
perform their own initialization; multi-rank online extraction is not yet
coordinated. Single-GPU execution is the verified path. LFM3 is a selectable
model alias, but the current end-to-end smoke was run with SmolLM2-135M;
the alias alone does not establish LFM3 runtime compatibility.

For a bounded real-data check, select enough files to retain at least two
valid samples after filtering (one train and one validation):

```bash
PYTHONPATH=. /data/util/anaconda3/envs/duplexomni/bin/python \
  train_full_duplex.py --task mono_post_training \
  --folders '/data/db/LibriTTS_R:en' --max-files 10 \
  --alignment-model medium --alignment-device cpu \
  --batch-size 1 --text-only-fraction 0 --steps 1 \
  --valid-ratio 0.2 --log-every 1 --output runs/mono_check
```

`--max-files` limits discovered candidates before duration/alignment filtering.
`--valid-ratio` must be strictly between 0 and 1. Batch size 1 requires
`--text-only-fraction 0` (the default mixes text-only and acoustic samples).
For a model-only test with synthetic codec tokens, use `--smoke-test
--smoke-steps 1`; this does not exercise Whisper or Mimi.

Timing appears in the console and TensorBoard under `timing/`. Startup
reports tokenizer/backbone/Mimi loading; dataset timing reports discovery,
Whisper loading, successful/failed extraction and filtering. `data` is batch
loading, `mimi` includes encoding and stream preparation, and `train` includes
forward/backward plus optimizer updates on accumulation boundaries. The step
`total` excludes data loading, logging, validation and checkpoint saving.
CUDA synchronization at stage boundaries makes the times interpretable but
adds profiling overhead. First-step timings include warmup; use subsequent
steps to estimate sustained throughput.

```bash
PYTHONPATH=. /data/util/anaconda3/envs/duplexomni/bin/python \
  -m unittest discover -s tests -v
tensorboard --logdir runs/mono_post_training/tensorboard
```

### Optional offline annotation and export

For offline workflows, create a single manifest from one folder or a comma-separated list of folders.
Quote the list if it contains spaces. Repeated or overlapping folder entries
are deduplicated by resolved WAV path; all folders must exist.

```bash
python3 scripts/build_libritts_manifest.py \
  '/data/db/LibriTTS_R,/path/to/another_corpus' \
  data/combined_manifest.jsonl
```

The supplied Whisper annotator is included at
[`scripts/annotate.py`](scripts/annotate.py). Run it once over the manifest to
create those sidecars, then export or train from the same manifest:

```bash
PYTHONPATH=. python3 scripts/annotate.py \
  data/LibriTTS_R/libritts_manifest.jsonl \
  --local --lang en --whisper_model medium
```

To materialize a few synthetic turns for listening or inspection, export them
as stereo WAVs. Channel 0 is the assistant suffix and channel 1 is the user
prefix; each WAV has a JSON sidecar with the selected word boundary and the
shifted assistant word timestamps.

```bash
PYTHONPATH=. python3 scripts/export_mono_continuation_stereo.py \
  --manifest /path/to/libritts_manifest.jsonl \
  --output-dir samples/mono_continuation \
  --max-items 10
```

### Folder-based speech preprocessing

No input manifest needs to be created. The following entry point scans the
comma-separated folders, reuses existing word sidecars or calls the supplied
`annotate.process_one`, splits mono speech at word boundaries, and encodes the
user/assistant waveforms with Mimi. This is an optional artifact exporter;
folder-based training does not require it. It uses one `--lang` for all input
folders and does not apply the training tokenizer's BPE collision filter.

```bash
PYTHONPATH=. /data/util/anaconda3/envs/duplexomni/bin/python \
  scripts/prepare_speech_pretraining.py \
  --folders '/data/db/LibriTTS_R,/path/to/another_corpus' \
  --output-dir data/prepared_speech \
  --max-files 10
```

Omit `--max-files` to process all eligible files (default duration 3–45 seconds).
Requires CUDA, the Moshi package, and the annotation dependencies. Outputs are
`annotations/` (word timestamps), `audio/` (assistant/user stereo WAVs and
assistant text sidecars), `tokens/` (user and assistant Mimi tensors), plus an
automatically generated `prepared_manifest.jsonl`, `summary.json`, and
`errors.json`. Input files are not modified. Annotation results under the same
output directory are reused on rerun; codec/audio exports are regenerated.
The current trainer still performs online Mimi encoding; it does not yet read
the exported `.pt` token artifacts.

### DuplexChat subset

DuplexChat distributes manifests rather than podcast audio. To prepare a
deterministic 1,000-hour maximum subset (keeping its English/Japanese source
ratio), run:

```bash
python3 scripts/sample_duplexchat_manifest.py \
  --source-dir data/DuplexChat \
  --output data/DuplexChat/subsets/duplexchat_1000h.jsonl \
  --target-hours 1000
```

The output contains complete dialogue clips whose total duration never exceeds
1,000 hours; the adjacent `.summary.json` records the exact result. Download
only the source episodes referenced by this smaller manifest. Podcast audio is
not included in DuplexChat and remains subject to the original right holders.


## Model selection

`--model SmolLM2-135M` and `--model LFM3-1.7B` resolve under `models/`.
If a known model is absent, rank 0 downloads its repository and other ranks
wait. A local directory or Hugging Face repository ID can also be supplied.
LFM3 enables remote model code automatically. Keep output directories separate.
Only SmolLM2-135M has completed the real-data end-to-end smoke in this environment;
LFM3 runtime compatibility must be verified separately.

## Distributed training

For stereo data with prepared sidecars, an example two-GPU launch is:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 train_full_duplex.py \
  --model SmolLM2-135M --task finetuning \
  --manifest /path/to/stereo_dialogue.jsonl \
  --batch-size 4 --grad-accum 2 --steps 1000 --output runs/stereo_ddp
```

Use the torchrun executable from the same environment as Python.
DDP creates a model and frozen Mimi per rank. Do not pass `--mimi-device`
under torchrun. Batch size is per rank; `--steps` counts microbatches.
Effective batch size is world size × batch size × gradient accumulation.
Online mono alignment is independently performed by each rank; shared extraction
and a persistent cache are not implemented. For predictable multi-rank mono
runs, prepare sidecars first and use `--alignment-mode sidecar`.

## Losses and metrics

Current CLI defaults are:

| Option | Default | Meaning |
|---|---:|---|
| `--text-loss-weight` | 1.0 | Assistant text loss multiplier |
| `--audio-loss-weight` | 0.1 | Assistant audio loss multiplier |
| `--semantic-loss-weight` | 100.0 | q0 weight in normalized audio codebook loss |
| `--text-padding-weight` | 0.5 | PAD weight relative to other text targets |
| `--min-duration` | 5.0 | Minimum sample seconds |
| `--max-duration` | 200.0 | Maximum sample seconds |

The direct mono dataset/export APIs may have different duration defaults;
pass explicit limits when comparing pipelines.

`assistant_np_accuracy` excludes PAD but includes EPAD, so it is not a
lexical-only score. `assistant_all_accuracy` includes PAD.
Per-codebook accuracy is logged as `assistant_rq_accuracy_1` through
`assistant_rq_accuracy_4` for the default four codebooks.
The trainer prints weighted total loss and the text/audio components.
The default text-only mixture masks acoustic conditioning and audio targets
for selected batch items. Use `--text-only-fraction 0` for fully acoustic batches.

## Checkpoints and logs

Rank 0 writes `checkpoint_XXXXXXXX.pt` and `config.json` to `--output`.
Checkpoints contain model and optimizer states; the final checkpoint is saved
even if the save interval has not been reached.
Deterministic split indices are written to `split.json`. Use distinct output
directories for distinct runs. TensorBoard defaults to `OUTPUT/tensorboard`.

The CLI does not currently expose an optimizer-resume option. Saving optimizer
state does not mean training automatically resumes from an existing checkpoint.

## Validation scope

Regression tests cover stream/loss behavior, inference parity, word boundaries,
annotation filtering, collisions and online extraction wiring (mocked Whisper).
A separate actual run verified Whisper medium → mono split → Mimi →
SmolLM2-135M forward/backward for one step using a tiny English sample.
This verifies execution, not convergence or multilingual quality.
