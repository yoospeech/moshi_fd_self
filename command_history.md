# Command history

Behavior-SD stereo 데이터를 내려받고, alignment를 확인하고, full-duplex
fine-tuning을 실행할 때 사용하는 명령 모음이다. 아래 명령은 프로젝트 루트에서
실행한다.

```bash
cd /home/speech/Documents/workspace/moshi_fd_self
```

## Behavior-SD validation 다운로드

```bash
hf download yhytoto12/behavior-sd \
  --repo-type dataset \
  --include "validation/*" \
  --local-dir /home/speech/Documents/DB/behavior-sd
```

다운로드된 WAV 개수와 stereo 채널을 확인한다.

```bash
find /home/speech/Documents/DB/behavior-sd/validation \
  -type f -name '*.wav' | wc -l

ffprobe -v error \
  -show_entries stream=channels,sample_rate:format=duration \
  -of default=noprint_wrappers=1 \
  /home/speech/Documents/DB/behavior-sd/validation/0000000005.wav
```

## 선택 사항: alignment 파일 미리 생성

온라인 alignment 학습에는 필요하지 않다. 별도 JSON을 먼저 만들고 싶을 때만
사용한다. `--vad auto`는 원본 `scripts/annotate.py`와 같이 10초를 초과하는
음성에 `auditok` VAD를 적용하고, 발화 경계의 silence를 1초 보존한다.

```bash
PYTHONPATH=. /data/util/anaconda3/envs/duplexomni/bin/python \
  annotate_behavior_sd.py \
  /home/speech/Documents/DB/behavior-sd/validation \
  --model turbo \
  --language en \
  --assistant-channel 0 \
  --vad auto \
  --keep-silence-in-segments 1 \
  --device cuda \
  --gpu 0 \
  --limit 2 \
  --verbose
```

전체 데이터는 `--limit 2`를 제거하고 shard별 프로세스를 실행한다.

```bash
PYTHONPATH=. /data/util/anaconda3/envs/duplexomni/bin/python \
  annotate_behavior_sd.py \
  /home/speech/Documents/DB/behavior-sd/validation \
  --model turbo --language en --assistant-channel 0 \
  --vad auto --device cuda --gpu 0 \
  --num-shards 2 --shard-index 0

PYTHONPATH=. /data/util/anaconda3/envs/duplexomni/bin/python \
  annotate_behavior_sd.py \
  /home/speech/Documents/DB/behavior-sd/validation \
  --model turbo --language en --assistant-channel 0 \
  --vad auto --device cuda --gpu 1 \
  --num-shards 2 --shard-index 1
```

## SmolLM 단일 GPU 스모크 테스트

WAV를 하위 경로까지 재귀 검색하고, stereo 전체 타임라인을 학습 입력으로
사용한다. Whisper alignment는 assistant 채널 0에 대해 온라인으로 수행된다.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. \
/data/util/anaconda3/envs/duplexomni/bin/python train_full_duplex.py \
  --task finetuning \
  --folders '/home/speech/Documents/DB/behavior-sd/validation:en' \
  --alignment-mode online \
  --alignment-model turbo \
  --alignment-device cuda:0 \
  --max-files 4 \
  --model models/SmolLM2-135M \
  --num-codebooks 4 \
  --context 1024 \
  --depformer-dim 128 --depformer-heads 4 \
  --depformer-layers 1 --depformer-ffn-dim 512 \
  --batch-size 1 --steps 1 \
  --min-duration 5 --max-duration 200 --min-text-tokens 1 \
  --valid-ratio 0.5 --valid-every 10 --valid-batches 1 \
  --log-every 1 --save-every 10 --log-timing \
  --output runs/behavior_sd_smollm_smoke
```

## LFM3 2-GPU DDP 학습

`--alignment-device local`은 각 DDP rank의 Whisper를 해당
`cuda:LOCAL_RANK`에 배치한다. `batch-size`는 rank당 크기이므로 아래 명령의
유효 batch size는 `2 GPUs * 1 * grad-accum 4 = 8`이다.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. \
/data/util/anaconda3/envs/duplexomni/bin/torchrun \
  --standalone --nproc_per_node=2 train_full_duplex.py \
  --task finetuning \
  --folders '/home/speech/Documents/DB/behavior-sd/validation:en' \
  --alignment-mode online \
  --alignment-model turbo \
  --alignment-device local \
  --model models/LFM3-1.7B \
  --min-duration 5 \
  --max-duration 200 \
  --min-text-tokens 5 \
  --max-files 0 \
  --valid-ratio 0.05 \
  --valid-every 500 \
  --valid-batches 20 \
  --batch-size 1 \
  --grad-accum 4 \
  --steps 1000 \
  --context 3000 \
  --num-codebooks 4 \
  --learning-rate 2e-4 \
  --temporal-learning-rate 2e-5 \
  --save-every 500 \
  --log-every 10 \
  --log-timing \
  --output runs/behavior_sd_lfm3_online_ddp
```

VRAM이 부족하면 먼저 `--context 2048`로 낮춘다. Online alignment는
`auditok` VAD를 이미 사용하지만 `beam_size=5`, `best_of=5` word alignment가
필요하므로 첫 epoch의 각 샘플 로딩은 오래 걸릴 수 있다.

## 상태 확인

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
  --format=csv

git status --short
```

