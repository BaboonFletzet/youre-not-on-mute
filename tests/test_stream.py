import numpy as np
import pytest
from services.inference.stream import FrameBuffer, reconcile


def test_timestamp_resampling_uses_nearest_frames():
    buffer = FrameBuffer(fps=25)
    for timestamp in [0, 37, 83, 121]:
        buffer.push(timestamp, np.array([timestamp]))
    frames, start, end = buffer.sample(minimum_seconds=0.1)
    assert [int(f[0]) for f in frames] == [0, 37, 83, 121]
    assert (start, end) == (0, 121)


def test_buffer_bounds_memory_and_context():
    buffer = FrameBuffer(seconds=2)
    for i in range(1000):
        buffer.push(i * 20, np.array([i]))
    assert len(buffer.frames) <= 101
    assert buffer.frames[-1][0] - buffer.frames[0][0] <= 2000


def test_gap_discards_previous_speaker_context():
    buffer = FrameBuffer()
    buffer.push(0, np.array([0]))
    assert buffer.push(900, np.array([1])) is True
    assert len(buffer.frames) == 1
    assert buffer.sample() is None


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf')])
def test_rejects_invalid_or_reordered_timestamps(bad):
    buffer = FrameBuffer()
    buffer.push(0, np.array([0]))
    with pytest.raises(ValueError):
        buffer.push(bad, np.array([0]))


def test_shared_prefix_does_not_commit_revised_words():
    stable, partial = reconcile('we can leave today', 'we can leave tomorrow')
    assert stable == 'we can leave'
    assert partial == 'tomorrow'
