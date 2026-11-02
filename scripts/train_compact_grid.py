"""Tiny masked-video pretraining + word-CTC pilot on the research-only GRID corpus.

Expected layout after extracting the official normal-quality video and word-alignment
archives: ROOT/s1/*.mpg ... ROOT/s4/*.mpg and ROOT/align/*.align.
This experiment is deliberately separate from the app's production recognizer.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.inference.calibration import word_error_rate


class GridDataset(Dataset):
    def __init__(self, root: Path, speakers: tuple[int, ...], vocab: dict[str, int],
                 limit: int | None = None, pretrain: bool = False):
        self.root, self.vocab, self.pretrain = root, vocab, pretrain
        items = []
        for speaker in speakers:
            videos = sorted((root / f's{speaker}').glob('*.mpg'))
            if limit:
                videos = videos[:limit]
            for video in videos:
                alignment = root / 'align' / f'{video.stem}.align'
                if not pretrain and alignment.is_file():
                    words = [line.split()[2] for line in alignment.read_text().splitlines()
                             if len(line.split()) == 3 and line.split()[2] != 'sil']
                    if words:
                        items.append((video, [vocab[word] for word in words if word in vocab]))
                elif pretrain:
                    items.append((video, []))
        self.items = items

    @staticmethod
    def read_video(path: Path, fixed_frames: int | None = None, augment: bool = False):
        capture = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            # GRID is a fixed frontal camera shot. Retain the lower central face,
            # including mouth, chin, and adjacent cheek motion.
            roi = frame[int(0.51 * h):int(0.83 * h), int(0.25 * w):int(0.75 * w)]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (64, 48), interpolation=cv2.INTER_AREA)
            frames.append(roi)
        capture.release()
        if not frames:
            raise ValueError(f'No frames decoded from {path}')
        if fixed_frames and len(frames) != fixed_frames:
            indices = np.linspace(0, len(frames) - 1, fixed_frames).round().astype(int)
            frames = [frames[i] for i in indices]
        if augment:
            stack = np.stack(frames)
            angle = random.uniform(-8.0, 8.0)
            matrix = cv2.getRotationMatrix2D((32, 24), angle, 1.0)
            stack = np.stack([cv2.warpAffine(frame, matrix, (64, 48), borderMode=cv2.BORDER_REPLICATE)
                              for frame in stack])
            if random.random() < 0.5:
                stack = stack[:, :, ::-1].copy()
            gain, bias = random.uniform(0.85, 1.15), random.uniform(-10, 10)
            stack = np.clip(stack.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
            frames = list(stack)
        return torch.from_numpy(np.stack(frames).copy()).float().div_(255.0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        video, target = self.items[index]
        frames = self.read_video(video, fixed_frames=32 if self.pretrain else None,
                                 augment=not self.pretrain)
        return frames, torch.tensor(target, dtype=torch.long)


def collate_clips(batch):
    videos, labels = zip(*batch)
    lengths = torch.tensor([len(v) for v in videos], dtype=torch.long)
    max_frames = int(lengths.max())
    padded = torch.zeros(len(videos), max_frames, 48, 64, dtype=torch.float32)
    for i, video in enumerate(videos):
        padded[i, :len(video)] = video
    return padded, torch.cat(labels), lengths, torch.tensor([len(x) for x in labels], dtype=torch.long)


class LipEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=(1, 2, 2), padding=1), nn.BatchNorm3d(16), nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.BatchNorm3d(32), nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.GELU(),
        )

    def forward(self, video):
        # B,C,T,H,W -> B,T,C after spatial pooling.
        features = self.layers(video).mean(dim=(-1, -2))
        return features.transpose(1, 2)


class MaskedLipAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LipEncoder()
        self.decoder = nn.Sequential(
            RepeatUpsample3d((2, 2, 2)),
            nn.Conv3d(64, 32, 3, padding=1), nn.GELU(),
            RepeatUpsample3d((2, 2, 2)),
            nn.Conv3d(32, 16, 3, padding=1), nn.GELU(),
            RepeatUpsample3d((1, 2, 2)),
            nn.Conv3d(16, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, video):
        return self.decoder(self.encoder.layers(video))


class RepeatUpsample3d(nn.Module):
    """Nearest-neighbor 3D upsample expressed with MPS-supported tensor ops."""
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        for dimension, factor in zip((2, 3, 4), self.scale):
            if factor > 1:
                x = x.repeat_interleave(factor, dim=dimension)
        return x


class LipWordCTC(nn.Module):
    def __init__(self, num_tokens: int):
        super().__init__()
        self.encoder = LipEncoder()
        self.temporal = nn.GRU(64, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(128, num_tokens)

    def forward(self, video):
        features = self.encoder(video)
        features, _ = self.temporal(features)
        return self.classifier(features)


def make_mask(batch, device):
    # Random 8x8 space-time tubes. Reconstruct only the missing image regions.
    coarse = torch.rand(batch, 1, 4, 6, 8, device=device) < 0.45
    return coarse.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3).repeat_interleave(8, dim=4)[:, :, :32, :48, :64]


def train_pretrain(root, train_speakers, limit, epochs, batch_size, device, output):
    dataset = GridDataset(root, train_speakers, {}, limit=limit, pretrain=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = MaskedLipAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        started, total, steps = time.perf_counter(), 0.0, 0
        for frames, _ in loader:
            video = frames.unsqueeze(1).to(device)
            mask = make_mask(len(frames), device)
            prediction = model(video * (~mask).to(video.dtype))
            loss = ((prediction - video).square() * mask).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            steps += 1
            if steps % 25 == 0:
                print(f'pretrain epoch={epoch + 1} batch={steps}/{len(loader)} '
                      f'masked_mse={total / steps:.5f}', flush=True)
        print(f'pretrain epoch={epoch + 1} masked_mse={total / max(steps, 1):.5f} '
              f'seconds={time.perf_counter() - started:.1f}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), output)
    print(f'encoder_saved={output} pretrain_videos={len(dataset)}', flush=True)


def greedy_decode(logits, length, id_to_word):
    ids = logits[:length].argmax(-1).detach().cpu().tolist()
    output, previous = [], -1
    for token in ids:
        if token != previous and token != 0:
            output.append(id_to_word[token])
        previous = token
    return ' '.join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / '.cache/datasets/grid/processed')
    parser.add_argument('--train-speakers', default='1,2,3')
    parser.add_argument('--test-speaker', type=int, default=4)
    parser.add_argument('--limit-per-speaker', type=int, default=1000)
    parser.add_argument('--pretrain-epochs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-limit', type=int, default=250)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=ROOT / '.cache/models/grid-compact-ctc.pth')
    args = parser.parse_args()
    train_speakers = tuple(int(x) for x in args.train_speakers.split(','))
    if args.test_speaker in train_speakers:
        raise SystemExit('Held-out speaker must not appear in --train-speakers.')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS is unavailable in this process. Run from a normal macOS terminal.')
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    device = torch.device(args.device)

    all_words = set()
    for path in args.root.joinpath('align').glob('*.align'):
        all_words.update(line.split()[2] for line in path.read_text().splitlines()
                         if len(line.split()) == 3 and line.split()[2] != 'sil')
    words = ['<blank>'] + sorted(all_words)
    vocab = {word: index for index, word in enumerate(words)}
    train_set = GridDataset(args.root, train_speakers, vocab, limit=args.limit_per_speaker)
    test_set = GridDataset(args.root, (args.test_speaker,), vocab, limit=args.eval_limit)
    print(f'device={device} train_speakers={train_speakers} train_videos={len(train_set)} '
          f'heldout_speaker={args.test_speaker} heldout_videos={len(test_set)} vocab={len(words)-1}', flush=True)

    encoder_path = args.output.with_name('grid-masked-encoder.pth')
    if args.pretrain_epochs > 0:
        train_pretrain(args.root, train_speakers, args.limit_per_speaker,
                       args.pretrain_epochs, args.batch_size, device, encoder_path)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_clips)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=collate_clips)
    model = LipWordCTC(len(words)).to(device)
    if args.pretrain_epochs > 0 and encoder_path.is_file():
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    for epoch in range(args.epochs):
        model.train(); started = time.perf_counter(); total = 0.0; steps = 0
        for frames, labels, frame_lengths, label_lengths in train_loader:
            video = frames.unsqueeze(1).to(device)
            logits = model(video)
            output_lengths = ((frame_lengths + 3) // 4).to('cpu')
            # MPS does not implement CTCLoss natively; evaluate this small loss on
            # CPU while autograd carries its gradient back to the MPS model.
            log_probs = logits.float().log_softmax(-1).transpose(0, 1).to('cpu')
            loss = ctc(log_probs, labels, output_lengths, label_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()); steps += 1
            if steps % 25 == 0:
                print(f'ctc epoch={epoch + 1} batch={steps}/{len(train_loader)} '
                      f'loss={total / steps:.4f}', flush=True)
        print(f'ctc epoch={epoch + 1} loss={total / max(steps, 1):.4f} '
              f'seconds={time.perf_counter() - started:.1f}', flush=True)

    model.eval(); references, hypotheses = [], []
    with torch.inference_mode():
        for frames, labels, frame_lengths, label_lengths in test_loader:
            logits = model(frames.unsqueeze(1).to(device))[0]
            output_length = int((int(frame_lengths[0]) + 3) // 4)
            reference = ' '.join(words[int(i)] for i in labels[:int(label_lengths[0])])
            hypothesis = greedy_decode(logits, output_length, words)
            references.append(reference); hypotheses.append(hypothesis)
    wer = word_error_rate(' '.join(references), ' '.join(hypotheses))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'words': words,
                'train_speakers': train_speakers, 'test_speaker': args.test_speaker}, args.output)
    print(f'heldout_word_error_rate={wer:.4f} evaluated_videos={len(test_set)} '
          f'checkpoint={args.output}', flush=True)


if __name__ == '__main__':
    main()
