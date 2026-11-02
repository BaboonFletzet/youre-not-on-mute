"""Visual-only Auto-AVSR adapter. No audio is opened or passed to the model."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '.cache' / 'auto-avsr'
CHECKPOINT = ROOT / '.cache' / 'models' / 'autoavsr-en.pth'


class VisualRecognizer:
    def __init__(self, device: str = 'auto', options=None):
        import torch
        if not CHECKPOINT.is_file() or not SOURCE.is_dir():
            raise RuntimeError('Model assets missing. Run: .venv/bin/python scripts/setup_model.py')
        sys.path.insert(0, str(SOURCE))
        from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
        from preparation.detectors.mediapipe.video_process import VideoProcess
        import mediapipe as mp

        from services.inference.decoding import DecodeOptions
        config = os.environ.get("RML_DECODING_CONFIG", str(ROOT / "config/decoder.json"))
        self.options = options or (DecodeOptions.load(config) if config else DecodeOptions())
        self.configuration_note = None
        if self.options.lm_weight and not (ROOT / '.cache/models/gpt2/model.safetensors').is_file():
            self.options = DecodeOptions.load(ROOT / 'config/fast-decoder.json')
            self.configuration_note = 'Using fast decoding; install local rescoring assets with scripts/setup_accuracy_assets.py lm.'
            print(self.configuration_note, flush=True)
        self.language_scorer = None
        self.diagnostics = False
        self.prealigned = False
        self.last_mouth = None
        self.torch = torch
        torch.set_num_threads(4)
        self.device = ('mps' if torch.backends.mps.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Metal unavailable in this process. Run the app from a normal macOS terminal.')
        units = (SOURCE / 'spm/unigram/unigram5000_units.txt').read_text().splitlines()
        self.tokens = ['<blank>'] + [line.split()[0] for line in units] + ['<eos>']
        self.model = E2E(len(self.tokens), 'video')
        weights = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        del weights
        if self.device == 'mps':
            from services.inference.mps import SpatialMaxPool3d
            # Temporal kernel = 1, so framewise max_pool2d is exactly equivalent.
            pool = self.model.frontend.frontend3D[3]
            if tuple(pool.kernel_size) != (1, 3, 3) or tuple(pool.stride) != (1, 2, 2) or tuple(pool.padding) != (0, 1, 1):
                raise RuntimeError('Unexpected reference pooling configuration; refusing an unverified replacement.')
            self.model.frontend.frontend3D[3] = SpatialMaxPool3d()
        self.model.eval().to(self.device)
        self.decoder_profile = 'attention_beam'
        self.detectors = [mp.solutions.face_detection.FaceDetection(model_selection=i, min_detection_confidence=0.5) for i in (0, 1)]
        self.cropper = VideoProcess(convert_gray=False)

    def landmarks(self, frames):
        """Use the author's four-keypoint alignment; reject ambiguous face selection."""
        points, valid, eye_distances = [], 0, []
        for frame in frames:
            detections = []
            for detector in self.detectors:
                detections = detector.process(frame).detections or []
                if detections:
                    break
            if len(detections) != 1:
                points.append(None)
                continue
            h, w = frame.shape[:2]
            keypoints = detections[0].location_data.relative_keypoints
            points.append(np.array([[p.x * w, p.y * h] for p in keypoints[:4]], dtype=np.float32))
            eye_distances.append(float(np.linalg.norm(points[-1][0] - points[-1][1])))
            valid += 1
        eye_pixels = float(np.median(eye_distances)) if eye_distances else 0
        self.input_quality = {'inter_eye_pixels': round(eye_pixels, 1),
                              'small_face': bool(eye_distances and eye_pixels < 20),
                              'note': 'Source resolution heuristic, not a confidence score'}
        return points, valid / max(1, len(frames))

    def set_decoder_profile(self, decoder: str):
        if decoder not in {'hybrid', 'attention_beam', 'attention', 'ctc'}:
            raise ValueError(f'Unsupported decoder profile: {decoder}')
        self.decoder_profile = decoder

    def _encode(self, frames: list[np.ndarray], started: float):
        torch = self.torch
        if self.prealigned:
            gray = np.stack(frames).astype(np.float32) / 255.0
            if gray.ndim != 3 or gray.shape[1:] not in ((96, 96), (88, 88)):
                raise ValueError('Prealigned mouth input must be T x 96 x 96 or T x 88 x 88.')
            if gray.shape[1] == 96:
                gray = gray[:, 4:92, 4:92]
            coverage = 1.0
            self.input_quality = {'prealigned': True}
        else:
            points, coverage = self.landmarks(frames)
            if coverage < 0.8:
                return None, None, coverage, 0.0, started, 'no_face'
            # The reference cropper already smooths landmarks over 12 frames.
            crops = self.cropper(np.stack(frames), points)
            crops = crops[:, 4:92, 4:92, :].astype(np.float32) / 255.0
            gray = crops @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        sample = torch.from_numpy((gray - 0.421) / 0.165).unsqueeze(1).unsqueeze(0).to(self.device)
        self.last_mouth = gray
        prepared = time.perf_counter()
        with torch.inference_mode():
            encoded = self.model.frontend(sample)
            encoded = self.model.proj_encoder(encoded)
            encoded, _ = self.model.encoder(encoded, None)
            ctc_logits = self.model.ctc.ctc_lo(encoded)
            self.last_ctc_log_probs = ctc_logits.log_softmax(-1)[0].float().cpu()
            ctc_ids = ctc_logits.argmax(dim=-1)[0]
            ctc_activity = float((ctc_ids != 0).float().mean().item())
        if ctc_activity < 0.02:
            return None, ctc_ids, coverage, ctc_activity, prepared, 'no_speech'
        return encoded, ctc_ids, coverage, ctc_activity, prepared, None

    def _decode_ctc(self, ctc_ids):
        ids = ctc_ids.cpu().tolist()
        collapsed, last = [], -1
        for token in ids:
            if token != last and token != 0:
                collapsed.append(token)
            last = token
        return collapsed

    def _decode_attention(self, encoded):
        torch = self.torch
        ys = torch.tensor([self.model.sos], device=self.device, dtype=torch.long)
        state, ids = None, []
        for _ in range(min(100, encoded.shape[1])):
            scores, state = self.model.decoder.score(ys, state, encoded[0])
            token = int(scores.argmax().item())
            if token == self.model.eos:
                break
            ids.append(token)
            ys = torch.cat((ys, ys.new_tensor([token])))
        return ids

    def _decode_hybrid(self, encoded):
        # Match the reference inference recipe: combine autoregressive attention
        # and CTC prefix scores, without an external language model.
        from espnet.nets.batch_beam_search import BatchBeamSearch
        scorers = self.model.scorers()
        if self.device == 'mps':
            torch = self.torch
            from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
            from espnet.nets.scorers.ctc import CTCPrefixScorer

            class MPSCTCPrefixScorer(CTCPrefixScorer):
                def batch_init_state(scorer, x):
                    logp = scorer.ctc.log_softmax(x.unsqueeze(0))
                    xlen = torch.tensor([logp.size(1)], device=logp.device)
                    scorer.impl = CTCPrefixScoreTH(logp, xlen, 0, scorer.eos)
                    # Upstream detects CUDA explicitly and otherwise defaults to
                    # CPU. MPS needs the prefix tensors and indices on Metal too.
                    scorer.impl.device = logp.device
                    scorer.impl.end_frames = xlen - 1
                    scorer.impl.idx_b = torch.arange(scorer.impl.batch, device=logp.device)
                    scorer.impl.idx_bo = (scorer.impl.idx_b * scorer.impl.odim).unsqueeze(1)
                    return None

            scorers['ctc'] = MPSCTCPrefixScorer(self.model.ctc, self.model.eos)
        from espnet.nets.scorers.length_bonus import LengthBonus
        beam = BatchBeamSearch(
            beam_size=10,
            vocab_size=len(self.tokens),
            weights={'decoder': 0.9, 'ctc': 0.1, 'lm': 0.0, 'length_bonus': 0.0},
            scorers={**scorers, 'lm': None,
                     'length_bonus': LengthBonus(len(self.tokens))},
            sos=self.model.odim - 1,
            eos=self.model.odim - 1,
            token_list=self.tokens,
            pre_beam_score_key='decoder',
        ).to(self.device)
        return [int(i) for i in beam(encoded[0])[0].yseq[1:].tolist()]

    def _decode_attention_nbest(self, encoded):
        from espnet.nets.batch_beam_search import BatchBeamSearch
        beam = BatchBeamSearch(
            beam_size=self.options.beam_size,
            vocab_size=len(self.tokens),
            weights={'decoder': 1.0},
            scorers={'decoder': self.model.decoder},
            sos=self.model.odim - 1,
            eos=self.model.odim - 1,
            token_list=self.tokens,
            pre_beam_score_key=None,
        ).to(self.device)
        return beam(encoded[0])[:self.options.beam_size]

    def _decode_attention_beam(self, encoded):
        return [int(i) for i in self._decode_attention_nbest(encoded)[0].yseq[1:].tolist()]

    def candidate_details(self, encoded, score_language=True):
        from services.inference.decoding import rank_candidates, LocalLanguageScorer
        torch = self.torch
        candidates, seen = [], set()
        for hypothesis in self._decode_attention_nbest(encoded):
            ids = [int(i) for i in hypothesis.yseq[1:].tolist() if int(i) not in (0, self.model.eos)]
            text = ''.join(self.tokens[i] for i in ids).replace('▁', ' ').strip()
            if not text or text in seen:
                continue
            seen.add(text)
            logp = self.last_ctc_log_probs.unsqueeze(1)
            target = torch.tensor(ids, dtype=torch.long)
            loss = torch.nn.functional.ctc_loss(logp, target,
                torch.tensor([logp.shape[0]]), torch.tensor([len(ids)]), reduction='sum')
            candidates.append({'ids': ids, 'text': text, 'attention_score': float(hypothesis.score),
                               'ctc_score': -float(loss) if self.torch.isfinite(loss) else -1e9, 'lm_score': 0.0})
        if self.options.lm_weight and candidates and score_language:
            if self.language_scorer is None:
                self.language_scorer = LocalLanguageScorer(ROOT / '.cache/models/gpt2')
            for item, score in zip(candidates, self.language_scorer.score([c['text'] for c in candidates])):
                item['lm_score'] = score
        return rank_candidates(candidates, self.options)

    def score_text_candidates(self, encoded, texts):
        """Teacher-forced visual likelihoods for proposals from another recognizer."""
        import sentencepiece as spm
        from services.inference.calibration import words
        torch=self.torch
        if not hasattr(self,'sentencepiece'):
            self.sentencepiece=spm.SentencePieceProcessor(model_file=str(SOURCE/'spm/unigram/unigram5000.model'))
            self.token_ids={token:i for i,token in enumerate(self.tokens)}
        ids_list=[];unique=[]
        for text in texts:
            text=' '.join(words(text)).upper()
            if not text or text in unique:continue
            unique.append(text)
            ids_list.append([self.token_ids.get(piece,1) for piece in self.sentencepiece.encode(text,out_type=str)])
        if not ids_list:return []
        lengths=torch.tensor([len(ids)+1 for ids in ids_list],device=self.device)
        size=int(lengths.max());batch=len(ids_list)
        inputs=torch.zeros((batch,size),dtype=torch.long,device=self.device)
        targets=torch.zeros_like(inputs)
        for i,ids in enumerate(ids_list):
            inputs[i,:len(ids)+1]=inputs.new_tensor([self.model.sos]+ids)
            targets[i,:len(ids)+1]=inputs.new_tensor(ids+[self.model.eos])
        valid=torch.arange(size,device=self.device)[None,:]<lengths[:,None]
        mask=valid[:,None,:] & torch.ones(size,size,device=self.device,dtype=torch.bool).tril()[None,:,:]
        with torch.inference_mode():
            scores,_=self.model.decoder(inputs,mask,encoded.expand(batch,-1,-1),None)
            selected=scores.log_softmax(-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
            attention=(selected*valid).sum(-1).cpu().tolist()
        candidates=[]
        for ids,text,score in zip(ids_list,unique,attention):
            logp=self.last_ctc_log_probs.unsqueeze(1)
            loss=torch.nn.functional.ctc_loss(logp,torch.tensor(ids),torch.tensor([len(logp)]),torch.tensor([len(ids)]),reduction='sum')
            candidates.append({'text':text,'ids':ids,'attention_score':score,
                               'ctc_score':-float(loss) if torch.isfinite(loss) else -1e9,'lm_score':0.})
        return candidates

    def mouth_preview(self):
        import cv2, base64
        frames = self.last_mouth
        if frames is None:
            return None
        indices = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
        sheet = np.concatenate([(frames[i] * 255).clip(0, 255).astype(np.uint8) for i in indices], axis=1)
        ok, png = cv2.imencode('.png', sheet)
        return 'data:image/png;base64,' + base64.b64encode(png).decode() if ok else None

    def _decode_text(self, encoded, ctc_ids, decoder):
        if decoder == 'ctc':
            ids = self._decode_ctc(ctc_ids)
        elif decoder == 'attention':
            ids = self._decode_attention(encoded)
        elif decoder == 'attention_beam':
            ids = self._decode_attention_beam(encoded)
        elif decoder == 'hybrid':
            ids = self._decode_hybrid(encoded)
        else:
            raise ValueError(f'Unsupported decoder: {decoder}')
        return ''.join(self.tokens[i] for i in ids if i not in (0, self.model.eos)).replace('▁', ' ').strip()

    def recognize_candidates(self, frames: list[np.ndarray]) -> dict[str, str]:
        """Decode one encoder pass three ways for held-out transcript calibration."""
        started = time.perf_counter()
        encoded, ctc_ids, _, _, _, quality = self._encode(frames, started)
        if quality:
            return {key: '' for key in ('attention_beam', 'attention', 'ctc')}
        with self.torch.inference_mode():
            ranked = self.candidate_details(encoded)
            candidates = {'attention_beam': ranked[0]['text'] if ranked else '',
                          'attention': self._decode_text(encoded, ctc_ids, 'attention'),
                          'ctc': self._decode_text(encoded, ctc_ids, 'ctc')}
        if self.device == 'mps':
            self.torch.mps.synchronize()
        return candidates

    def recognize(self, frames: list[np.ndarray], decoder: str | None = None) -> dict:
        decoder = decoder or self.decoder_profile
        started = time.perf_counter()
        self.last_mouth = None
        encoded, ctc_ids, coverage, ctc_activity, prepared, quality = self._encode(frames, started)
        if quality:
            return {'text': '', 'quality': quality, 'coverage': round(coverage, 2),
                    'ctc_activity': round(ctc_activity, 3),
                    **({'mouth_preview': self.mouth_preview()} if self.diagnostics else {}),
                    'elapsed': round(time.perf_counter() - started, 3), 'device': self.device}
        candidates, margin = [], None
        with self.torch.inference_mode():
            if decoder == 'attention_beam':
                candidates = self.candidate_details(encoded)
                text = candidates[0]['text'] if candidates else ''
                if len(candidates) > 1:
                    margin = (candidates[0]['score'] - candidates[1]['score']) / max(1, len(candidates[0]['ids']))
            else:
                text = self._decode_text(encoded, ctc_ids, decoder)
        aligned_words = None
        if candidates:
            from services.inference.alignment import word_spans
            aligned_words = word_spans(self.last_ctc_log_probs.numpy(), candidates[0]['ids'], self.tokens)
        uncertain = not text or (margin is not None and margin < self.options.uncertainty_margin)
        if uncertain:
            text = ''
        if self.device == 'mps':
            self.torch.mps.synchronize()
        return {'text': text, 'quality': 'uncertain' if uncertain else 'readable',
                'score_margin': margin, 'candidates': candidates, 'word_spans': aligned_words,
                'input_quality': self.input_quality,
                **({'mouth_preview': self.mouth_preview()} if self.diagnostics else {}), 'coverage': round(coverage, 2),
                'ctc_activity': round(ctc_activity, 3), 'decoder': decoder,
                'elapsed': round(time.perf_counter() - started, 3),
                'preprocess_seconds': round(prepared - started, 3), 'device': self.device}

    def recognize_mouth(self, frames, decoder=None):
        """Offline evaluation of reference-aligned mouth crops; never detect faces here."""
        self.prealigned = True
        try:
            return self.recognize(frames, decoder)
        finally:
            self.prealigned = False

    def close(self):
        for detector in self.detectors:
            detector.close()
