"""Local VALLR visual-to-phoneme recognizer (experimental, not sentence ASR)."""
from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '.cache' / 'vallr'
CHECKPOINT = ROOT / '.cache' / 'models' / 'vallr.path'

PHONEMES = [
    '<blank>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N',
    'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W',
    'Y', 'Z', 'ZH',
]


class VALLRRecognizer:
    def __init__(self, device: str = 'auto'):
        import torch
        import mediapipe as mp
        if not SOURCE.is_dir() or not CHECKPOINT.is_file():
            raise RuntimeError('VALLR assets missing. Run: .venv/bin/python scripts/setup_vallr.py')
        self.torch = torch
        self.device = ('mps' if torch.backends.mps.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Metal unavailable in this process.')
        torch.set_num_threads(4)
        sys.path.insert(0, str(SOURCE))
        from transformers import VideoMAEConfig, Wav2Vec2Config
        from Models.VALLR import VALLR

        phoneme_config = Wav2Vec2Config()
        phoneme_config.vocab_size = len(PHONEMES)
        self.model = VALLR(VideoMAEConfig(), phoneme_config, adapter_dim=256)
        weights = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        del weights
        self.model.eval().to(self.device)
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

    def _crop_face(self, frames: list[np.ndarray]):
        crops, valid = [], 0
        for frame in frames:
            detections = self.detector.process(frame).detections or []
            if len(detections) != 1:
                crops.append(None)
                continue
            h, w = frame.shape[:2]
            box = detections[0].location_data.relative_bounding_box
            cx, cy = (box.xmin + box.width / 2) * w, (box.ymin + box.height / 2) * h
            side = max(box.width * w, box.height * h) * 1.45
            x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
            x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
            # Pad at the image boundary instead of stretching a partial face crop.
            crop = np.zeros((max(1, y1 - y0), max(1, x1 - x0), 3), dtype=np.uint8)
            sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if sx1 <= sx0 or sy1 <= sy0:
                crops.append(None)
                continue
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
            crops.append(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR))
            valid += 1
        return crops, valid / max(1, len(frames))

    def recognize(self, frames: list[np.ndarray]) -> dict:
        started = time.perf_counter()
        if len(frames) < 16:
            return {'text': '', 'quality': 'too_short', 'output_kind': 'phonemes',
                    'elapsed': round(time.perf_counter() - started, 3), 'device': self.device}
        indices = np.linspace(0, len(frames) - 1, 16).astype(int)
        sampled = [frames[i] for i in indices]
        face_crops, coverage = self._crop_face(sampled)
        if coverage < 0.8:
            return {'text': '', 'quality': 'no_face', 'coverage': round(coverage, 2),
                    'output_kind': 'phonemes', 'elapsed': round(time.perf_counter() - started, 3),
                    'device': self.device}
        # Interpolate isolated misses so one detector dropout does not erase a frame.
        for i, crop in enumerate(face_crops):
            if crop is None:
                nearest = min((j for j, item in enumerate(face_crops) if item is not None),
                              key=lambda j: abs(j - i))
                face_crops[i] = face_crops[nearest]
        video = self.torch.from_numpy(np.stack(face_crops)).float()
        video = video.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        with self.torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            logits, _ = self.model(video)
        if self.device == 'mps':
            self.torch.mps.synchronize()
        ids = logits.argmax(-1)[0].cpu().tolist()
        decoded, previous = [], -1
        for token in ids:
            if token == 0:
                previous = 0
                continue
            if token != previous:
                decoded.append(PHONEMES[token])
            previous = token
        text = ' '.join(decoded)
        return {'text': text, 'quality': 'readable' if decoded else 'no_speech',
                'output_kind': 'phonemes', 'coverage': round(coverage, 2),
                'decoder': 'phoneme_ctc_greedy', 'elapsed': round(time.perf_counter() - started, 3),
                'device': self.device}

    def recognize_stream(self, frames, crop=True, scale='raw'):
        """Research: preserve native 25 fps in consecutive 16-frame blocks."""
        from services.inference.phoneme_words import PronunciationDecoder
        started = time.perf_counter()
        if crop:
            crops, coverage = self._crop_face(frames)
        else:
            crops = [cv2.resize(f, (224,224)) for f in frames]
            coverage = 1.0
        if coverage < .8:
            return {'text':'','quality':'no_face','elapsed':time.perf_counter()-started}
        valid = [i for i,c in enumerate(crops) if c is not None]
        for i,c in enumerate(crops):
            if c is None: crops[i]=crops[min(valid,key=lambda j:abs(i-j))]
        all_logits=[]
        with self.torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            for offset in range(0,len(crops),16):
                block=crops[offset:offset+16]
                count=len(block)
                block += [block[-1]]*(16-count)
                video=self.torch.from_numpy(np.stack(block)).float().permute(0,3,1,2).unsqueeze(0).to(self.device)
                if scale == 'unit': video = video / 255
                elif scale == 'symmetric': video = video / 127.5 - 1
                elif scale != 'raw': raise ValueError('Unknown pixel scale')
                logits,_=self.model(video)
                keep=max(1,round(logits.shape[1]*count/16))
                all_logits.append(logits[0,:keep].log_softmax(-1).cpu().numpy())
        logp=np.concatenate(all_logits)
        if not hasattr(self,'word_decoder'):self.word_decoder=PronunciationDecoder(PHONEMES)
        candidates=self.word_decoder.decode(logp)
        ids=logp.argmax(-1).tolist();phones=[];previous=-1
        for token in ids:
            if token and token != previous:phones.append(PHONEMES[token])
            previous=token
        return {'text':candidates[0]['text'] if candidates else '', 'quality':'experimental',
                'phonemes':' '.join(phones),'output_steps':len(logp),'candidates':candidates,
                'coverage':coverage,'elapsed':round(time.perf_counter()-started,3),'device':self.device}

    def close(self):
        self.detector.close()
