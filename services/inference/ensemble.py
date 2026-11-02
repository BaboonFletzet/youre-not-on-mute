"""AV-HuBERT proposals scored by Auto-AVSR vision, CTC, and an optional text prior."""
from .model import VisualRecognizer,ROOT
from .decoding import rank_candidates,LocalLanguageScorer


class ProposalRecognizer(VisualRecognizer):
    def __init__(self,device='auto',options=None):
        super().__init__(device,options)
        from .avhubert import AVHubertRecognizer
        self.proposal=AVHubertRecognizer(self.device,options)

    def candidate_details(self,encoded):
        result=self.proposal.recognize_mouth(list(self.last_mouth*255))
        texts=[c['text'] for c in result['candidates']]
        # A cheap greedy proposal supplies a second model's hypothesis without a second beam search.
        texts.append(self._decode_text(encoded, None, 'attention'))
        candidates=self.score_text_candidates(encoded,texts)
        if self.options.lm_weight and candidates:
            if self.language_scorer is None:self.language_scorer=LocalLanguageScorer(ROOT/'.cache/models/gpt2')
            for c,score in zip(candidates,self.language_scorer.score([c['text'] for c in candidates])):c['lm_score']=score
        return rank_candidates(candidates,self.options)

    def close(self):
        super().close()
        self.proposal.close()
