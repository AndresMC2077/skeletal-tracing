"""
utils.py
========
Small, dependency-free geometry helpers shared by the gesture classifier and
the pose backbone. Keeping these pure functions (no OpenCV/torch imports)
makes them trivial to unit test.
"""

from __future__ import annotations
import math
from collections import deque
from typing import Deque, Optional, Sequence, Tuple

Point = Tuple[float, float]


def angle_deg(a: Point, b: Point, c: Point) -> float:
    """Angle ABC (at vertex b), in degrees, in [0, 180]."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_theta))


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


class EMAFilter:
    """Exponential moving-average filter for a single scalar or point.

    Cheap jitter reduction for keypoints -- much less compute than a Kalman
    filter or One-Euro filter, which matters on the Orin Nano's shared
    CPU/GPU memory bus. alpha closer to 1.0 = less smoothing (more responsive).
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._state: Optional[Tuple[float, ...]] = None

    def update(self, value: Sequence[float]) -> Tuple[float, ...]:
        if self._state is None:
            self._state = tuple(value)
        else:
            self._state = tuple(
                self.alpha * v + (1 - self.alpha) * s
                for v, s in zip(value, self._state)
            )
        return self._state

    def reset(self):
        self._state = None


class RollingBuffer:
    """Fixed-size FIFO of (timestamp, value) samples, used for the swipe
    gesture's temporal displacement tracking."""

    def __init__(self, maxlen: int):
        self._buf: Deque[Tuple[float, Sequence[float]]] = deque(maxlen=maxlen)

    def push(self, t: float, value: Sequence[float]):
        self._buf.append((t, value))

    def __len__(self):
        return len(self._buf)

    def full(self) -> bool:
        return len(self._buf) == self._buf.maxlen

    def oldest(self):
        return self._buf[0] if self._buf else None

    def newest(self):
        return self._buf[-1] if self._buf else None

    def clear(self):
        self._buf.clear()
