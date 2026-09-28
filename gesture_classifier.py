"""
gesture_classifier.py
======================
Two independent detectors sharing the same per-frame observation:

  StopDetector   - STATIC. Open palm, facing the camera, arm raised /
                   extended. Evaluated on angles (elbow bend, finger
                   extension) and pixel distances (fingertip spread),
                   normalized by hand/torso scale so it's distance-invariant.

  SwipeRightDetector - DYNAMIC. Tracks the wrist's x-position over a rolling
                   time window and fires when horizontal displacement
                   exceeds a threshold while vertical drift stays small and
                   the hand is NOT in a "Stop" pose (avoids double-firing
                   while someone holds their palm up and their whole body
                   drifts sideways).

Both are intentionally simple, threshold-based, explainable classifiers
(no extra ML model) -- appropriate given the brief asks for angle/distance
math specifically, and it keeps the whole thing cheap enough to run on the
Orin Nano's CPU with zero extra inference cost.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time

from config import DeviceProfile
from hand_landmarker import HandResult, FINGER_TIPS, FINGER_PIPS, WRIST
from pose_estimator import PersonPose
from utils import angle_deg, euclidean, RollingBuffer


class Gesture(Enum):
    NONE = "none"
    STOP = "stop"
    SWIPE_RIGHT = "swipe_right"


@dataclass
class GestureEvent:
    gesture: Gesture
    confidence: float
    timestamp: float


class StopDetector:
    """Open palm facing the camera, arm raised."""

    # Tunables -- expressed as ratios/angles so they're resolution-independent.
    MIN_FINGER_EXTENSION_RATIO = 1.15   # tip-to-wrist dist must exceed pip-to-wrist by this factor
    MIN_SPREAD_RATIO = 0.55             # avg fingertip pairwise spread / hand span
    MIN_ELBOW_ANGLE = 60.0              # arm doesn't have to be fully straight
    MIN_EXTENDED_FINGERS = 4            # thumb is unreliable for "facing camera"; require 4/5

    def evaluate(self, person: PersonPose, hand: Optional[HandResult]) -> float:
        """Returns a confidence in [0, 1] that this frame shows a Stop pose."""
        if hand is None:
            return 0.0

        pts = hand.landmarks_xy
        wrist = pts[WRIST]

        # Hand scale reference: wrist -> middle-finger MCP (landmark 9) distance.
        hand_span = euclidean(wrist, pts[9])
        if hand_span < 1e-3:
            return 0.0

        extended = 0
        tip_dists = []
        for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):  # skip thumb (idx 0)
            d_tip = euclidean(pts[tip_idx], wrist)
            d_pip = euclidean(pts[pip_idx], wrist)
            tip_dists.append(pts[tip_idx])
            if d_pip > 1e-3 and (d_tip / d_pip) > self.MIN_FINGER_EXTENSION_RATIO:
                extended += 1

        if extended < self.MIN_EXTENDED_FINGERS:
            return 0.0

        # Fingertip spread: average pairwise pixel distance among the four
        # (non-thumb) fingertips, normalized by hand_span. A closed fist or
        # a fist-with-thumb-up collapses this value.
        n = len(tip_dists)
        pairwise = [
            euclidean(tip_dists[i], tip_dists[j])
            for i in range(n) for j in range(i + 1, n)
        ]
        avg_spread = sum(pairwise) / len(pairwise)
        spread_ratio = avg_spread / hand_span
        if spread_ratio < self.MIN_SPREAD_RATIO:
            return 0.0

        # Arm should be raised/extended, not resting at the hip: use the
        # elbow angle (shoulder-elbow-wrist) as a coarse "arm engaged" check.
        elbow_angle = self._elbow_angle_for_wrist(person, hand.handedness)
        if elbow_angle is not None and elbow_angle < self.MIN_ELBOW_ANGLE:
            return 0.0

        confidence = min(1.0, 0.5 + 0.5 * (spread_ratio / self.MIN_SPREAD_RATIO - 1.0) + 0.1 * extended / 4)
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _elbow_angle_for_wrist(person: PersonPose, handedness: str) -> Optional[float]:
        side = "left" if handedness == "Left" else "right"
        try:
            shoulder = person.kp(f"{side}_shoulder")
            elbow = person.kp(f"{side}_elbow")
            wrist = person.kp(f"{side}_wrist")
        except KeyError:
            return None
        if person.kp_conf(f"{side}_elbow") < 0.3 or person.kp_conf(f"{side}_wrist") < 0.3:
            return None
        return angle_deg(tuple(shoulder), tuple(elbow), tuple(wrist))


class SwipeRightDetector:
    """Dynamic gesture: wrist travels left-to-right within a short window."""

    def __init__(self, profile: DeviceProfile):
        self.profile = profile
        self.buffer = RollingBuffer(maxlen=profile.swipe_window_frames)

    def update(self, person: PersonPose, is_stop_pose: bool, timestamp: float) -> float:
        """Feed one frame's wrist position; returns confidence in [0, 1]."""
        # Prefer whichever wrist has higher detection confidence this frame.
        r_conf, l_conf = person.kp_conf("right_wrist"), person.kp_conf("left_wrist")
        wrist = person.kp("right_wrist") if r_conf >= l_conf else person.kp("left_wrist")
        if max(r_conf, l_conf) < 0.3 or is_stop_pose:
            # Low-confidence tracking or the hand is doing a static Stop:
            # don't let noise / held poses masquerade as a swipe.
            self.buffer.clear()
            return 0.0

        shoulder_l, shoulder_r = person.kp("left_shoulder"), person.kp("right_shoulder")
        shoulder_width = max(euclidean(tuple(shoulder_l), tuple(shoulder_r)), 1e-3)

        self.buffer.push(timestamp, (float(wrist[0]), float(wrist[1])))
        if not self.buffer.full():
            return 0.0

        (t0, p0), (t1, p1) = self.buffer.oldest(), self.buffer.newest()
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        norm_dx = dx / shoulder_width

        if norm_dx < self.profile.swipe_min_norm_disp:
            return 0.0
        if abs(dy) > self.profile.swipe_max_vertical_ratio * abs(dx):
            return 0.0

        confidence = min(1.0, norm_dx / (self.profile.swipe_min_norm_disp * 1.5))
        return confidence


class GestureEngine:
    """Owns both detectors, applies a shared cooldown so a single physical
    gesture doesn't spam repeated events across consecutive frames."""

    def __init__(self, profile: DeviceProfile):
        self.profile = profile
        self.stop_detector = StopDetector()
        self.swipe_detector = SwipeRightDetector(profile)
        self._last_fire_time: float = 0.0

    def update(
        self,
        person: Optional[PersonPose],
        hand: Optional[HandResult],
        timestamp: Optional[float] = None,
    ) -> GestureEvent:
        timestamp = timestamp if timestamp is not None else time.time()

        if person is None:
            self.swipe_detector.buffer.clear()
            return GestureEvent(Gesture.NONE, 0.0, timestamp)

        stop_conf = self.stop_detector.evaluate(person, hand) if hand is not None else 0.0
        is_stop = stop_conf > 0.5

        swipe_conf = self.swipe_detector.update(person, is_stop, timestamp)

        in_cooldown = (timestamp - self._last_fire_time) < self.profile.gesture_cooldown_s

        if is_stop and not in_cooldown:
            self._last_fire_time = timestamp
            return GestureEvent(Gesture.STOP, stop_conf, timestamp)

        if swipe_conf > 0.5 and not in_cooldown:
            self._last_fire_time = timestamp
            self.swipe_detector.buffer.clear()
            return GestureEvent(Gesture.SWIPE_RIGHT, swipe_conf, timestamp)

        return GestureEvent(Gesture.NONE, max(stop_conf, swipe_conf), timestamp)
