"""
visualizer.py
=============
Pure drawing code. Nothing here touches the model, the camera, or gesture
logic -- it just takes already-computed results and paints them onto a
frame. Keeping this separate means you can swap the inference backend (or
run headless on the Jetson for a deployed product) without touching this
file at all.
"""

from __future__ import annotations
from typing import List, Optional
import cv2
import numpy as np

from pose_estimator import PersonPose
from hand_landmarker import HandResult
from gesture_classifier import GestureEvent, Gesture
from config import COCO_KPT

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
]

GESTURE_COLORS = {
    Gesture.STOP: (0, 0, 255),
    Gesture.SWIPE_RIGHT: (0, 200, 255),
    Gesture.NONE: (0, 255, 0),
}


def draw_skeleton(frame: np.ndarray, person: PersonPose, conf_thresh: float = 0.3) -> None:
    for a, b in SKELETON_EDGES:
        ca, cb = person.kp_conf(a), person.kp_conf(b)
        if ca < conf_thresh or cb < conf_thresh:
            continue
        pa = tuple(person.kp(a).astype(int))
        pb = tuple(person.kp(b).astype(int))
        cv2.line(frame, pa, pb, (255, 200, 0), 2, cv2.LINE_AA)

    for name in COCO_KPT:
        if person.kp_conf(name) < conf_thresh:
            continue
        p = tuple(person.kp(name).astype(int))
        cv2.circle(frame, p, 3, (0, 165, 255), -1, cv2.LINE_AA)

    x0, y0, x1, y1 = person.box_xyxy.astype(int)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)


def draw_hand(frame: np.ndarray, hand: HandResult) -> None:
    pts = hand.landmarks_xy.astype(int)
    for a, b in HAND_EDGES:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p), 2, (0, 255, 255), -1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    gesture_event: Optional[GestureEvent],
    profile_name: str,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.putText(frame, f"FPS: {fps:5.1f}  |  profile: {profile_name}",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if gesture_event is not None and gesture_event.gesture != Gesture.NONE:
        color = GESTURE_COLORS[gesture_event.gesture]
        label = f"{gesture_event.gesture.value.upper()}  ({gesture_event.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cv2.rectangle(frame, (w // 2 - tw // 2 - 12, 50), (w // 2 + tw // 2 + 12, 50 + th + 24), color, -1)
        cv2.putText(frame, label, (w // 2 - tw // 2, 50 + th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)


def render_frame(
    frame: np.ndarray,
    people: List[PersonPose],
    hand: Optional[HandResult],
    gesture_event: Optional[GestureEvent],
    fps: float,
    profile_name: str,
) -> np.ndarray:
    out = frame.copy()
    for p in people:
        draw_skeleton(out, p)
    if hand is not None:
        draw_hand(out, hand)
    draw_hud(out, fps, gesture_event, profile_name)
    return out
