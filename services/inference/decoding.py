"""Candidate scoring. Language scores only rank hypotheses supplied by vision."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path


@dataclass
class DecodeOptions:
    beam_size: int = 4
    ctc_weight: float = 0.0
    lm_weight: float = 0.0
    length_bonus: float = 0.0
    uncertainty_margin: float = 0.0

    def __post_init__(self):
        if self.beam_size not in (2, 4, 8):
            raise ValueError('Beam size must be 2, 4, or 8.')
        for name in ('ctc_weight', 'lm_weight', 'length_bonus', 'uncertainty_margin'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite.')
        if not 0 <= self.ctc_weight <= 1 or not 0 <= self.lm_weight <= 1:
            raise ValueError('Score weights must be between zero and one.')
        if abs(self.length_bonus) > 2 or not 0 <= self.uncertainty_margin <= 10:
            raise ValueError('Decoder penalty out of range.')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())['options'])

    def asdict(self):
        return asdict(self)


def rank_candidates(candidates, options):
    ranked = []
    for candidate in candidates:
        score = (1 - options.ctc_weight) * candidate['attention_score']
        if options.ctc_weight:
            score += options.ctc_weight * candidate['ctc_score']
        if options.lm_weight:
            score += options.lm_weight * candidate['lm_score']
        score += options.length_bonus * len(candidate['ids'])
        ranked.append({**candidate, 'score': score})
    return sorted(ranked, key=lambda c: c['score'], reverse=True)


class LocalLanguageScorer:
    def __init__(self, path):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Small batches on CPU leave unified GPU memory available to the visual model.
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True).eval()

    def score(self, texts):
        torch = self.torch
        prefixed = [self.tokenizer.eos_token + text.lower() for text in texts]
        batch = self.tokenizer(prefixed, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.inference_mode():
            logits = self.model(**batch).logits[:, :-1].log_softmax(-1)
            targets = batch['input_ids'][:, 1:]
            selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            return (selected * batch['attention_mask'][:, 1:]).sum(-1).tolist()
