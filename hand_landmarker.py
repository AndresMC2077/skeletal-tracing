"""
hand_landmarker.py
===================
YOLOv8-Pose's 17 COCO keypoints give us shoulder/elbow/wrist but NOT
fingertips -- there simply aren't finger points in that skeleton. Since the
gesture spec requires fingertip geometry (open-palm "Stop"), we run
MediaPipe Hands as a second, cheap stage, but only on a small crop around
each detected wrist rather than the full frame. This "detect coarse, refine
locally" pattern is what keeps this pipeline real-time on the Orin Nano:
MediaPipe Hands on a 200x200 crop is far cheaper than on a 1280x720 frame.

MediaPipe Hands runs on CPU (with XNNPACK/NEON acceleration) here, in
parallel with the GPU running YOLO -- so the two stages don't compete for
the same compute resource. That's a deliberate hardware-mapping choice for
the Jetson's shared-memory architecture.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2
import mediapipe as mp

from config import DeviceProfile

# MediaPipe Hands landmark indices we care about
WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [3, 6, 10, 14, 18]   # joint just below each tip, used for "extended" test


@dataclass
class HandResult:
    landmarks_xy: np.ndarray   # (21, 2) in FULL-FRAME pixel coordinates
    handedness: str            # "Left" / "Right" as reported by MediaPipe


class HandLandmarker:
    def __init__(self, profile: DeviceProfile):
        self.profile = profile
        self._mp_hands = mp.solutions.hands
        self.detector = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,                       # one crop = one hand of interest
            model_complexity=profile.hand_model_complexity,
            min_detection_confidence=profile.hand_min_det_conf,
            min_tracking_confidence=profile.hand_min_track_conf,
        )

    def infer_on_wrist(
        self,
        frame_bgr: np.ndarray,
        wrist_xy: Tuple[float, float],
        crop_radius: int,
    ) -> Optional[HandResult]:
        h, w = frame_bgr.shape[:2]
        cx, cy = int(wrist_xy[0]), int(wrist_xy[1])

        x0, y0 = max(cx - crop_radius, 0), max(cy - crop_radius, 0)
        x1, y1 = min(cx + crop_radius, w), min(cy + crop_radius, h)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None

        crop = frame_bgr[y0:y1, x0:x1]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.detector.process(crop_rgb)
        if not result.multi_hand_landmarks:
            return None

        hand_lms = result.multi_hand_landmarks[0]
        ch, cw = crop.shape[:2]
        pts = np.array(
            [[lm.x * cw + x0, lm.y * ch + y0] for lm in hand_lms.landmark],
            dtype=np.float32,
        )
        handedness = "Unknown"
        if result.multi_handedness:
            handedness = result.multi_handedness[0].classification[0].label

        return HandResult(landmarks_xy=pts, handedness=handedness)

    def close(self):
        self.detector.close()
