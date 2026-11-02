"""Bounded timestamp resampling and conservative overlap reconciliation."""
from __future__ import annotations
from collections import deque
import numpy as np


class FrameBuffer:
    def __init__(self, seconds=6.0, fps=25):
        self.seconds, self.fps = seconds, fps
        self.frames = deque(maxlen=int(seconds * 60))
        self.last_timestamp = -1.0

    def push(self, timestamp: float, frame: np.ndarray):
        if not np.isfinite(timestamp) or timestamp <= self.last_timestamp:
            raise ValueError('Frame timestamps must increase.')
        gap = self.last_timestamp >= 0 and timestamp - self.last_timestamp > 500
        if gap:
            self.frames.clear()
        self.last_timestamp = timestamp
        self.frames.append((timestamp, frame))
        while self.frames and timestamp - self.frames[0][0] > self.seconds * 1000:
            self.frames.popleft()
        return gap

    def retain(self, seconds):
        """Keep actual frames at the tail; do not reset monotonic timestamp checks."""
        if not self.frames: return
        cutoff = self.frames[-1][0] - seconds * 1000
        while self.frames and self.frames[0][0] < cutoff:
            self.frames.popleft()

    def sample(self, minimum_seconds=2):
        if len(self.frames) < 2:
            return None
        items = list(self.frames)
        start, end = items[0][0], items[-1][0]
        if end - start < minimum_seconds * 1000:
            return None
        times = np.array([x[0] for x in items])
        grid = np.arange(start, end + 0.01, 1000 / self.fps)
        indices = np.searchsorted(times, grid).clip(0, len(items) - 1)
        # Nearest-neighbor timestamp resampling; avoid inventing motion.
        prev = (indices - 1).clip(0)
        indices = np.where(abs(times[prev] - grid) < abs(times[indices] - grid), prev, indices)
        return [items[i][1] for i in indices], start, end


def reconcile(previous: str, current: str) -> tuple[str, str]:
    """Only repeated prefixes become stable; returned remainder stays provisional."""
    a, b = previous.split(), current.split()
    shared = 0
    for x, y in zip(a, b):
        if x.casefold() != y.casefold():
            break
        shared += 1
    return ' '.join(b[:shared]), ' '.join(b[shared:])
