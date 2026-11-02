"""Prepare reproducible evaluation manifests; never uses model predictions as labels."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path)
    parser.add_argument('--grid', type=Path)
    parser.add_argument('--exclude-manifest',type=Path)
    parser.add_argument('--confirmation',action='store_true',help='Create evaluation-only clips, excluding previous manifests')
    parser.add_argument('--per-split', type=int, default=60)
    parser.add_argument('--output', type=Path, default=ROOT / '.cache/accuracy/manifest.jsonl')
    args = parser.parse_args()
    items = []
    if args.parquet:
        import pyarrow.parquet as pq
        import numpy as np
        parquet = pq.ParquetFile(args.parquet)
        target = args.output.parent / 'mouths'; target.mkdir(parents=True, exist_ok=True)
        # This source has no speaker IDs: only claim clip-disjoint evaluation.
        indices = [int(x) for batch in parquet.iter_batches(columns=['idx']) for x in batch.column('idx').to_pylist()]
        excluded = {int(json.loads(line)['id'].removeprefix('lrs3-')) for line in args.exclude_manifest.read_text().splitlines()} if args.exclude_manifest else set()
        ordered = sorted((i for i in indices if i not in excluded), key=lambda i: hashlib.sha256(str(i).encode()).hexdigest())
        chosen = set(ordered[:args.per_split if args.confirmation else 2 * args.per_split])
        dev_ids = set() if args.confirmation else set(ordered[:args.per_split])
        for batch in parquet.iter_batches(batch_size=1, columns=['idx', 'video', 'label']):
            index = int(batch.column('idx')[0].as_py())
            if index not in chosen: continue
            array = batch.column('video')[0].values
            shape = (len(array), len(array[0].values), len(array[0].values[0].values))
            pixels = array.flatten().flatten().to_numpy().reshape(shape)
            filename = target / f'lrs3-{index}.npz'
            np.savez_compressed(filename, mouth=pixels)
            items.append({'id': f'lrs3-{index}', 'video': str(filename.resolve()),
                          'reference': batch.column('label')[0].as_py().strip(),
                          'group': f'clip-{index}', 'source': 'mattymchen/lrs3-test',
                          'input_kind': 'prealigned_mouth', 'speaker_disjoint': False,
                          'label_status': 'dataset_transcript', 'split': 'dev' if index in dev_ids else 'eval'})
    elif args.grid:
        for speaker in ('s1', 's2', 's3', 's4'):
            for path in sorted((args.grid / speaker).glob('*.mpg')):
                align = args.grid / 'align' / (path.stem + '.align')
                if not align.is_file(): continue
                text = ' '.join(line.split()[2] for line in align.read_text().splitlines()
                                if len(line.split()) == 3 and line.split()[2] not in {'sil', 'sp'})
                items.append({'id': speaker + '-' + path.stem, 'video': str(path.resolve()),
                              'reference': text, 'group': speaker, 'source': 'GRID', 'label_status': 'dataset_transcript',
                              'split': 'dev' if speaker in {'s1', 's2'} else 'eval'})
    else:
        parser.error('Provide --parquet or --grid')
    groups = sorted({x['group'] for x in items}, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    development = set(groups[::2])
    result = []
    for split in ('dev', 'eval'):
        candidates = [x for x in items if x.get('split', 'dev' if x['group'] in development else 'eval') == split]
        # Hash ordering avoids taking only the first speaker in an archive.
        candidates.sort(key=lambda x: hashlib.sha256(x['id'].encode()).hexdigest())
        for item in candidates[:args.per_split]: result.append({**item, 'split': split})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(x) + '\n' for x in result))
    print(f'Wrote {len(result)} clips to {args.output}; dev/eval groups are disjoint (clip groups only for LRS3; speaker identities unavailable). Dataset labels are not manually reverified.')

if __name__ == '__main__': main()
