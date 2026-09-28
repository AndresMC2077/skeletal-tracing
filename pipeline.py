"""
pipeline.py
===========
Main entry point. Deliberately structured as three decoupled stages so each
can be profiled, swapped, or moved to a different thread/process without
touching the others:

    capture (video_stream.py) -> INFERENCE (this file's `run_inference_step`)
                                       -> DRAWING/DISPLAY (visualizer.py)

Usage:
    GESTURE_PROFILE=dev_rtx3060      python pipeline.py     # RTX 3060 dev box
    GESTURE_PROFILE=deploy_orin_nano python pipeline.py     # Jetson, after export_tensorrt.py
"""

from __future__ import annotations
import argparse
import time
from typing import List, Optional, Tuple

import cv2

from config import get_profile, DeviceProfile
from video_stream import ThreadedVideoStream
from pose_estimator import PoseEstimator, PersonPose
from hand_landmarker import HandLandmarker, HandResult
from gesture_classifier import GestureEngine, GestureEvent, Gesture
from visualizer import render_frame


def pick_primary_person(people: List[PersonPose]) -> Optional[PersonPose]:
    """Choose the largest bounding box (closest / most prominent person)."""
    if not people:
        return None
    def area(p: PersonPose) -> float:
        x0, y0, x1, y1 = p.box_xyxy
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return max(people, key=area)


def run_inference_step(
    frame_bgr,
    pose_estimator: PoseEstimator,
    hand_landmarker: HandLandmarker,
    gesture_engine: GestureEngine,
    profile: DeviceProfile,
    timestamp: float,
) -> Tuple[List[PersonPose], Optional[HandResult], GestureEvent]:
    """Pure inference: no drawing, no cv2.imshow. Returns everything the
    display stage needs to render."""

    people = pose_estimator.infer(frame_bgr)
    primary = pick_primary_person(people)

    hand: Optional[HandResult] = None
    if primary is not None:
        r_conf, l_conf = primary.kp_conf("right_wrist"), primary.kp_conf("left_wrist")
        if max(r_conf, l_conf) >= 0.3:
            wrist = primary.kp("right_wrist") if r_conf >= l_conf else primary.kp("left_wrist")
            # Crop radius scales with the detected person's box size so it
            # still works whether they're close to or far from the camera.
            x0, y0, x1, y1 = primary.box_xyxy
            box_h = max(1.0, y1 - y0)
            crop_radius = int(max(60, 0.35 * box_h))
            hand = hand_landmarker.infer_on_wrist(frame_bgr, tuple(wrist), crop_radius)

    gesture_event = gesture_engine.update(primary, hand, timestamp)
    return ([primary] if primary is not None else []), hand, gesture_event


def main():
    parser = argparse.ArgumentParser(description="2D skeletal tracking + gesture recognition")
    parser.add_argument("--profile", default=None, help="dev_rtx3060 | deploy_orin_nano")
    parser.add_argument("--source", type=int, default=0, help="webcam index")
    parser.add_argument("--headless", action="store_true", help="skip cv2.imshow (Jetson deployment)")
    args = parser.parse_args()

    profile = get_profile(args.profile)
    print(f"[pipeline] using profile '{profile.name}' -> weights={profile.pose_weights}")

    stream = ThreadedVideoStream(
        src=args.source,
        width=profile.capture_width,
        height=profile.capture_height,
        fps=profile.capture_fps_request,
    )
    pose_estimator = PoseEstimator(profile)
    hand_landmarker = HandLandmarker(profile)
    gesture_engine = GestureEngine(profile)

    fps_ema = 0.0
    last_t = time.time()

    try:
        while True:
            frame, ts = stream.read()
            if frame is None:
                continue

            people, hand, gesture_event = run_inference_step(
                frame, pose_estimator, hand_landmarker, gesture_engine, profile, ts
            )

            if gesture_event.gesture != Gesture.NONE:
                print(f"[gesture] {gesture_event.gesture.value} conf={gesture_event.confidence:.2f}")

            now = time.time()
            inst_fps = 1.0 / max(now - last_t, 1e-6)
            fps_ema = inst_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst_fps
            last_t = now

            if not args.headless:
                out = render_frame(frame, people, hand, gesture_event, fps_ema, profile.name)
                cv2.imshow("Skeletal Tracking + Gesture Recognition", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        stream.stop()
        hand_landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
