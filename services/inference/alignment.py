"""CTC forced alignment of an existing hypothesis, without reference transcripts."""
from __future__ import annotations
import numpy as np


def token_spans(log_probs, ids):
    """Return best-path frame spans, or None when the candidate cannot align."""
    if not ids: return []
    lp = np.asarray(log_probs)
    labels = np.zeros(2 * len(ids) + 1, dtype=np.int64); labels[1::2] = ids
    steps, states = lp.shape[0], len(labels)
    score = np.full(states, -np.inf); score[0] = lp[0, 0]; score[1] = lp[0, ids[0]]
    back = np.zeros((steps, states), dtype=np.int8)
    skip = np.zeros(states, dtype=bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    for time in range(1, steps):
        advance = np.r_[-np.inf, score[:-1]]
        jump = np.r_[-np.inf, -np.inf, score[:-2]]; jump[~skip] = -np.inf
        choices = np.stack((score, advance, jump))
        back[time] = choices.argmax(0)
        score = choices.max(0) + lp[time, labels]
    state = states - 1 if score[-1] >= score[-2] else states - 2
    if not np.isfinite(score[state]): return None
    path = np.zeros(steps, dtype=np.int64)
    for time in range(steps - 1, -1, -1):
        path[time] = state
        state -= int(back[time, state])
    result = []
    for index in range(len(ids)):
        times = np.flatnonzero(path == 2 * index + 1)
        if not len(times): return None
        result.append((int(times[0]), int(times[-1]) + 1))
    return result


def word_spans(log_probs, ids, tokens):
    spans = token_spans(log_probs, ids)
    if spans is None: return None
    words = []
    for token_id, (start, end) in zip(ids, spans):
        token = tokens[token_id]
        if token.startswith('▁') or not words:
            words.append({'text': token.lstrip('▁'), 'start': start / len(log_probs), 'end': end / len(log_probs)})
        else:
            words[-1]['text'] += token
            words[-1]['end'] = end / len(log_probs)
    return [w for w in words if w['text']]


class WindowCommitter:
    """Assign aligned words to successive intervals while retaining overlap context."""
    def __init__(self): self.boundary = -1.0

    def select(self, words, start_ms, end_ms, final, hold_ms=0):
        cutoff = end_ms - hold_ms if final else end_ms
        result = []
        for word in words:
            midpoint = start_ms + (end_ms - start_ms) * (word['start'] + word['end']) / 2
            if self.boundary <= midpoint < cutoff:
                result.append(word['text'])
        if final: self.boundary = cutoff
        return ' '.join(result)
