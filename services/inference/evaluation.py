"""Per-utterance edit accounting; empty output counts as deletions, not success."""
from __future__ import annotations
from services.inference.calibration import words


def edits(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    # tuples carry total cost, substitutions, deletions, insertions.
    row = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, expected in enumerate(ref, 1):
        nxt = [(i, 0, i, 0)]
        for j, actual in enumerate(hyp, 1):
            cost, s, d, ins = row[j-1]
            diagonal = (cost + (expected != actual), s + (expected != actual), d, ins)
            cost, s, d, ins = row[j]
            deletion = (cost + 1, s, d + 1, ins)
            cost, s, d, ins = nxt[j-1]
            insertion = (cost + 1, s, d, ins + 1)
            nxt.append(min(diagonal, deletion, insertion))
        row = nxt
    _, s, d, i = row[-1]
    return {'substitutions': s, 'deletions': d, 'insertions': i, 'reference_words': len(ref)}


def summarize(rows):
    import numpy as np
    totals = {key: 0 for key in ('substitutions', 'deletions', 'insertions', 'reference_words')}
    for row in rows:
        for key, count in edits(row['reference'], row.get('text', '')).items(): totals[key] += count
    silent = [r for r in rows if not words(r['reference'])]
    silent_seconds = sum(r['video_seconds'] for r in silent)
    speech = [r for r in rows if words(r['reference'])]
    return {**totals, 'clips': len(rows),
            'wer': (totals['substitutions'] + totals['deletions'] + totals['insertions']) / max(1, totals['reference_words']),
            'utterance_coverage': sum(bool(words(r.get('text', ''))) for r in speech) / max(1, len(speech)),
            'false_words_per_silent_minute': sum(len(words(r.get('text', ''))) for r in silent) * 60 / silent_seconds if silent_seconds else None,
            'decode_p95_seconds': float(np.percentile([r['elapsed'] for r in rows], 95)) if rows else None,
            'real_time_factor': sum(r['elapsed'] for r in rows) / max(.001, sum(r['video_seconds'] for r in rows))}
