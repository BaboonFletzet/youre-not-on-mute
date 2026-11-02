"""Run real visual-only inference on a local video; never decode its audio track."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', nargs='?')
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
    parser.add_argument('--seconds', type=float, default=4)
    parser.add_argument('--decoder', choices=['attention_beam', 'attention', 'ctc', 'hybrid'], default='attention_beam')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    from services.inference.model import VisualRecognizer
    import av
    import numpy as np

    start = time.perf_counter()
    recognizer = VisualRecognizer(args.device)
    print(json.dumps({'event': 'loaded', 'device': recognizer.device, 'seconds': round(time.perf_counter() - start, 3)}), flush=True)
    if not args.video:
        print('Model loaded. Pass a video path to measure real recognition.', flush=True)
        recognizer.close()
        return
    frames, next_time = [], 0.0
    with av.open(args.video) as container:
        for frame in container.decode(video=0):
            timestamp = float(frame.time or 0)
            if timestamp > args.seconds:
                break
            if timestamp + 0.001 >= next_time:
                frames.append(frame.to_ndarray(format='rgb24'))
                next_time += 1 / 25
    if len(frames) < 13:
        raise SystemExit('Need at least half a second of video.')
    for repeat in range(args.repeat):
        result = recognizer.recognize(frames, decoder=args.decoder)
        result.update(video_seconds=round(len(frames) / 25, 3), frames=len(frames), repeat=repeat + 1, decoder=args.decoder)
        result['real_time_factor'] = round(result['elapsed'] / result['video_seconds'], 3)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    recognizer.close()


if __name__ == '__main__':
    main()
