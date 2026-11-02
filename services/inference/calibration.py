"""Small held-out calibration helpers for the existing recognizer."""
from __future__ import annotations

import re


DECODERS = ('attention_beam', 'attention', 'ctc')


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold())


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    row = list(range(len(hyp) + 1))
    for i, expected in enumerate(ref, 1):
        next_row = [i]
        for j, actual in enumerate(hyp, 1):
            next_row.append(min(
                next_row[-1] + 1,
                row[j] + 1,
                row[j - 1] + (expected != actual),
            ))
        row = next_row
    return row[-1] / len(ref)


def select_decoder(practice_reference: str, validation_reference: str,
                   practice_hypotheses: dict[str, list[str]],
                   validation_hypotheses: dict[str, list[str]]) -> tuple[str, dict]:
    """Select on practice text; accept a change only if held-out WER improves."""
    practice_scores = {
        key: word_error_rate(practice_reference, ' '.join(practice_hypotheses.get(key, [])))
        for key in DECODERS
    }
    validation_scores = {
        key: word_error_rate(validation_reference, ' '.join(validation_hypotheses.get(key, [])))
        for key in DECODERS
    }
    candidate = min(DECODERS, key=lambda key: (practice_scores[key], key != 'attention_beam'))
    selected = candidate
    # Small samples can favor a decoder by chance. Require a clear held-out gain.
    if candidate != 'attention_beam' and validation_scores[candidate] > validation_scores['attention_beam'] - 0.02:
        selected = 'attention_beam'
    return selected, {
        'practice_wer': {key: round(value, 3) for key, value in practice_scores.items()},
        'validation_wer': {key: round(value, 3) for key, value in validation_scores.items()},
        'candidate': candidate,
        'accepted': selected == candidate,
    }
