"""Real visual-only corpus evaluation with per-clip candidates and input diagnostics."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
from services.inference.decoding import DecodeOptions
from services.inference.evaluation import summarize


def read_frames(path, maximum=12):
    import av
    import numpy as np
    if Path(path).suffix == '.npz':
        array = np.load(path)['mouth']
        return list(array), len(array) / 25
    from services.inference.stream import FrameBuffer
    buffer = FrameBuffer(seconds=maximum + 1)
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.time is None: raise ValueError('Video has no timestamps')
            timestamp = float(frame.time) * 1000
            if buffer.frames and timestamp - buffer.frames[0][0] > maximum * 1000:
                raise ValueError('Clip exceeds maximum; split it with aligned references first')
            buffer.push(timestamp, frame.to_ndarray(format='rgb24'))
    sampled = buffer.sample(.1)
    if sampled is None: raise ValueError('Too few video frames')
    frames, start, end = sampled
    return frames, (end - start) / 1000


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path); p.add_argument('--split', choices=['dev', 'eval'], required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='auto')
    p.add_argument('--beam', type=int, default=4); p.add_argument('--options', type=Path)
    p.add_argument('--streaming', action='store_true'); p.add_argument('--overlap', type=float, default=0, choices=[0,.5,1])
    p.add_argument('--lm', action='store_true'); p.add_argument('--limit', type=int)
    p.add_argument('--variant', choices=['raw', 'jpeg', 'frozen'], default='raw')
    p.add_argument('--model', choices=['auto_avsr', 'avhubert', 'ensemble', 'vallr', 'vallr_stream'], default='auto_avsr')
    args = p.parse_args()
    import cv2
    records = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    dev = {x['group'] for x in records if x['split'] == 'dev'}
    test = {x['group'] for x in records if x['split'] == 'eval'}
    if dev & test: raise ValueError('Development and evaluation groups overlap')
    records = [r for r in records if r['split'] == args.split][:args.limit]
    if not records: raise ValueError('No clips selected')
    options = DecodeOptions.load(args.options) if args.options else DecodeOptions(beam_size=args.beam, lm_weight=.01 if args.lm else 0)
    if args.model == 'auto_avsr':
        from services.inference.model import VisualRecognizer
        recognizer = VisualRecognizer(args.device, options)
        recognizer.diagnostics = True
    elif args.model == 'ensemble':
        from services.inference.ensemble import ProposalRecognizer
        recognizer = ProposalRecognizer(args.device, options)
        recognizer.diagnostics = True
    elif args.model == 'avhubert':
        from services.inference.avhubert import AVHubertRecognizer
        recognizer = AVHubertRecognizer(args.device, options)
    else:
        from services.inference.vallr import VALLRRecognizer
        recognizer = VALLRRecognizer(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    try:
        with args.output.open('w') as out:
            for record in records:
                frames, seconds = read_frames(record['video'])
                if args.variant == 'jpeg':
                    frames = [cv2.imdecode(cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 92])[1], cv2.IMREAD_UNCHANGED) for f in frames]
                elif args.variant == 'frozen': frames = [frames[len(frames)//2]] * len(frames)
                if args.streaming:
                    if args.model != 'auto_avsr': raise ValueError('Streaming simulation currently supports Auto-AVSR')
                    from services.inference.windowing import decode_windows
                    result = decode_windows(recognizer, frames, record.get('input_kind') == 'prealigned_mouth', args.overlap)
                elif record.get('input_kind') == 'prealigned_mouth':
                    if args.model not in {'auto_avsr','avhubert','ensemble'}: raise ValueError('VALLR requires uncropped RGB faces; do not resize mouth crops into faces.')
                    result = recognizer.recognize_mouth(frames)
                else:
                    result = recognizer.recognize_stream(frames) if args.model == 'vallr_stream' else recognizer.recognize(frames)
                preview = result.pop('mouth_preview', None)
                if preview:
                    import base64
                    (args.output.parent / f'{record["id"]}-{args.variant}-mouth.png').write_bytes(base64.b64decode(preview.split(',')[1]))
                row = {**record, **result, 'reference': '' if args.variant == 'frozen' else record['reference'],
                       'video_seconds': seconds, 'variant': args.variant, 'streaming': args.streaming, 'overlap': args.overlap, 'options': options.asdict()}
                out.write(json.dumps(row, allow_nan=False) + '\n'); out.flush(); outputs.append(row)
                print(json.dumps({'id': record['id'], 'reference': row['reference'], 'text': row['text'], 'elapsed': row['elapsed']}), flush=True)
    finally: recognizer.close()
    summary = summarize(outputs)
    summary.update(model=args.model, split=args.split, variant=args.variant, options=options.asdict(),
                   note='Dataset transcripts; no human reverification. Decode latency excludes window collection and screen capture.')
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__': main()
