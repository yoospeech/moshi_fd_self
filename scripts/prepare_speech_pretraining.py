"""Folders -> word annotation -> full-utterance stereo/text -> Mimi tokens.

Run with PYTHONPATH=.:thirdparty/moshi/moshi. Existing training is unchanged;
the exported Mimi tokens are artifacts, not yet a training-cache interface.
"""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folders', required=True, help='comma-separated mono WAV folders')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-files', type=int, default=0, help='0: all files; use 10 for a small run')
    parser.add_argument('--whisper-model', default='medium')
    parser.add_argument('--lang', default='en')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mimi-checkpoint')
    parser.add_argument('--num-codebooks', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-duration', type=float, default=3.0)
    parser.add_argument('--max-duration', type=float, default=45.0)
    args = parser.parse_args()
    roots = [Path(p.strip()).expanduser().resolve() for p in args.folders.split(',') if p.strip()]
    if not roots or any(not p.is_dir() for p in roots):
        parser.error('all input folders must exist')
    if args.max_files < 0 or args.num_codebooks < 1:
        parser.error('max-files must be nonnegative and num-codebooks positive')

    import soundfile as sf
    import torch
    from moshi.models import loaders
    from full_duplex.dataset import MonoPostTrainingDataset
    # Reuse the supplied annotation procedure without changing annotate.py.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import annotate

    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('the supplied annotate.py requires a CUDA device')
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    for name in ('annotations', 'audio', 'tokens'):
        (out / name).mkdir(parents=True, exist_ok=True)
    paths = sorted({p.resolve() for root in roots for p in root.rglob('*.wav')
                    if not p.resolve().is_relative_to(out)})
    rows, failures = [], []
    for path in paths:
        try:
            info = sf.info(path)
            duration = info.frames / info.samplerate
            if info.channels != 1 or not args.min_duration <= duration <= args.max_duration:
                continue
            key = hashlib.sha256(str(path).encode()).hexdigest()[:24]
            rows.append({'path': str(path), 'duration': duration,
                         'alignment_path': str(out / 'annotations' / (key + '.json'))})
            if args.max_files and len(rows) >= args.max_files:
                break
        except Exception as exc:
            failures.append({'path': str(path), 'error': str(exc)})
    if not rows:
        raise ValueError('no eligible mono WAV files')
    print(f'Selected {len(rows)} WAVs', flush=True)
    whisper_model = None
    annotated = []
    for index, row in enumerate(rows):
        path, alignment = Path(row['path']), Path(row['alignment_path'])
        try:
            if not alignment.exists():
                source_sidecar = path.with_suffix('.json')
                if source_sidecar.exists():
                    write_json(alignment, json.loads(source_sidecar.read_text()))
                else:
                    if whisper_model is None:
                        whisper_model = annotate.whisper.load_model(args.whisper_model, device=device)
                    annotate.process_one(path, alignment, args.lang, whisper_model,
                                         SimpleNamespace(keep_silence_in_segments=1.0))
            words = json.loads(alignment.read_text())['alignments']
            if not words:
                raise ValueError('empty word annotation')
            annotated.append(row)
            print(f'Annotation {index + 1}/{len(rows)}', flush=True)
        except Exception as exc:
            failures.append({'path': str(path), 'error': str(exc)})
    del whisper_model
    gc.collect()
    torch.cuda.empty_cache()
    write_json(out / 'errors.json', failures)
    if not annotated:
        raise ValueError('no successful annotations; see errors.json')

    checkpoint = args.mimi_checkpoint or str(loaders.hf_get(loaders.MIMI_NAME, loaders.DEFAULT_REPO))
    mimi = loaders.get_mimi(checkpoint, device=device, num_codebooks=args.num_codebooks)
    mimi.eval().requires_grad_(False)
    dataset = MonoPostTrainingDataset(
        None, None, entries=annotated, text_padding_id=0, text_epad_id=1,
        sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate,
        min_duration=args.min_duration, max_duration=args.max_duration,
    )
    results = []
    for index, entry in enumerate(dataset.entries):
        try:
            user, assistant, alignments = dataset.render_stereo(index)
            if not alignments:
                raise ValueError('no assistant suffix words')
            key = Path(entry['alignment_path']).stem
            stereo_path = out / 'audio' / (key + '.wav')
            tokens_path = out / 'tokens' / (key + '.pt')
            with torch.inference_mode():
                codes = mimi.encode(torch.stack([user, assistant]).unsqueeze(1).to(device)).cpu()
            tmp = tokens_path.with_suffix('.pt.tmp')
            torch.save({'user_audio_codes': codes[0], 'assistant_audio_codes': codes[1],
                        'sample_rate': mimi.sample_rate, 'frame_rate': mimi.frame_rate,
                        'num_samples': user.numel()}, tmp)
            tmp.replace(tokens_path)
            sf.write(stereo_path, torch.stack([assistant, user], dim=1).numpy(),
                     mimi.sample_rate, subtype='PCM_16')
            text = ' '.join(w.strip() for w, _, _ in alignments)
            write_json(stereo_path.with_suffix('.json'), {'alignments': alignments,
                       'text': text, 'source_path': entry['path']})
            results.append({'path': str(stereo_path), 'duration': user.numel() / mimi.sample_rate,
                            'user_channel': 1, 'assistant_channel': 0,
                            'assistant_text': text, 'mimi_path': str(tokens_path),
                            'source_path': entry['path']})
            print(f'Mimi/export {index + 1}/{len(dataset)}', flush=True)
        except Exception as exc:
            failures.append({'path': entry['path'], 'error': str(exc)})
    (out / 'prepared_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
    write_json(out / 'errors.json', failures)
    write_json(out / 'summary.json', {'selected': len(rows), 'annotated': len(annotated),
               'exported': len(results), 'dataset_rejected': dataset.rejected,
               'errors': len(failures), 'options': {k: str(v) if isinstance(v, Path) else v
                                                 for k, v in vars(args).items()}})
    if not results:
        raise RuntimeError('no exported samples; see errors.json')
    print(f'Finished: {len(results)} samples in {out}', flush=True)


if __name__ == '__main__':
    main()
