"""Pronunciation-constrained CTC beam decoding. Experimental, no free-form generation."""
from __future__ import annotations
import re
import numpy as np


class PronunciationDecoder:
    def __init__(self, phonemes, pronunciations=None, beam_size=64):
        if pronunciations is None:
            import cmudict
            pronunciations = cmudict.dict()
        self.beam_size = beam_size
        self.children = [{}]; self.terminals = [[]]
        ids = {p:i for i,p in enumerate(phonemes)}
        for word, variants in pronunciations.items():
            if not re.fullmatch("[a-z]+(?:'[a-z]+)?",word): continue
            for variant in variants:
                phones = [ids.get(re.sub(r'\d','',p)) for p in variant]
                if not phones or any(p is None for p in phones): continue
                node = 0
                for token in phones:
                    if token not in self.children[node]:
                        self.children[node][token] = len(self.children)
                        self.children.append({}); self.terminals.append([])
                    node = self.children[node][token]
                if word not in self.terminals[node]: self.terminals[node].append(word)

    def decode(self, log_probs, word_penalty=1.0):
        # State = lexicon node, previous CTC token, completed words.
        beam = {(0,0,()):0.0}
        for row in np.asarray(log_probs):
            tokens = set(np.argsort(row)[-12:].tolist()) | {0}
            nxt = {}
            def add(key, value):
                nxt[key] = float(np.logaddexp(nxt.get(key,-np.inf),value))
            for (node, previous, words), score in beam.items():
                for token in tokens:
                    value = score + float(row[token])
                    if token == 0 or token == previous:
                        add((node,token,words),value)
                        continue
                    child = self.children[node].get(token)
                    if child is None: continue
                    if self.children[child]: add((child,token,words),value)
                    for word in self.terminals[child]:
                        add((0,token,words+(word,)),value-word_penalty)
            beam = dict(sorted(nxt.items(),key=lambda x:x[1],reverse=True)[:self.beam_size])
            if not beam: break
        candidates = {}
        for (node,_,words),score in beam.items():
            if node == 0 and words:
                text=' '.join(words);candidates[text]=max(score,candidates.get(text,-np.inf))
        return [{'text':text,'score':score} for text,score in sorted(candidates.items(),key=lambda x:x[1],reverse=True)[:8]]
