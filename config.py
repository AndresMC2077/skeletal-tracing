"""
config.py
=========
Central place for every hardware-dependent knob. The idea is: you develop and
iterate on the RTX 3060 box using the "dev_rtx3060" profile (plain .pt/.onnx
weights, CUDA, fp16), then flip PROFILE to "deploy_orin_nano" on the Jetson
after exporting a TensorRT engine with export_tensorrt.py. Nothing else in the
codebase needs to change.
"""

from dataclasses import dataclass, field
from typing import Tuple
import os


@dataclass
class DeviceProfile:
    name: str
    # --- Pose backbone (YOLOv8-Pose via Ultralytics/PyTorch) ---
    pose_weights: str          # .pt (dev) or .engine (Jetson, TensorRT)
    pose_device: str           # "cuda:0" or "0" for ultralytics
    pose_imgsz: int
    pose_half: bool            # fp16 inference
    pose_conf: float
    pose_max_det: int

    # --- MediaPipe Hands (fingertip detail) ---
    hand_model_complexity: int     # 0 = lite, 1 = full
    hand_min_det_conf: float
    hand_min_track_conf: float
    hand_num_threads: int          # CPU threads MediaPipe's XNNPACK delegate may use

    # --- Capture / display ---
    capture_width: int
    capture_height: int
    capture_fps_request: int
    display_scale: float

    # --- Gesture temporal parameters (frame-rate dependent, tuned per device) ---
    swipe_window_frames: int
    swipe_min_norm_disp: float     # min horizontal displacement, normalized by shoulder width
    swipe_max_vertical_ratio: float  # vertical drift allowed relative to horizontal displacement
    gesture_cooldown_s: float


PROFILES = {
    # --------------------------------------------------------------------
    # Development / training box: RTX 3060 (12 GB), plenty of headroom.
    # Use the raw PyTorch checkpoint or an ONNX export; TensorRT engines
    # built on an x86 GPU are NOT portable to the Orin Nano's Xavier/Ampere
    # iGPU + different TensorRT/CUDA/cuDNN minor versions, so we don't
    # bother building a .engine here -- that step happens on-device (Jetson).
    # --------------------------------------------------------------------
    "dev_rtx3060": DeviceProfile(
        name="dev_rtx3060",
        pose_weights=os.environ.get("POSE_WEIGHTS", "yolov8n-pose.pt"),
        pose_device="cuda:0",
        pose_imgsz=640,
        pose_half=True,
        pose_conf=0.5,
        pose_max_det=4,
        hand_model_complexity=1,   # full model, GPU has plenty of headroom for CPU work in parallel
        hand_min_det_conf=0.6,
        hand_min_track_conf=0.5,
        hand_num_threads=4,
        capture_width=1280,
        capture_height=720,
        capture_fps_request=30,
        display_scale=1.0,
        swipe_window_frames=12,
        swipe_min_norm_disp=1.4,
        swipe_max_vertical_ratio=0.6,
        gesture_cooldown_s=0.8,
    ),

    # --------------------------------------------------------------------
    # Deployment target: Jetson Orin Nano (8 GB), power-constrained.
    # pose_weights MUST point to a .engine file built ON THIS DEVICE via
    # export_tensorrt.py (TensorRT engines are architecture + TRT-version
    # locked -- see that script's docstring).
    # --------------------------------------------------------------------
    "deploy_orin_nano": DeviceProfile(
        name="deploy_orin_nano",
        pose_weights=os.environ.get("POSE_WEIGHTS", "yolov8n-pose.engine"),
        pose_device="0",
        pose_imgsz=416,             # smaller input -> big FPS win on the Orin Nano GPU
        pose_half=True,             # fp16 engine
        pose_conf=0.5,
        pose_max_det=2,             # we only care about 1-2 people in frame -> less NMS work
        hand_model_complexity=0,    # lite model: MediaPipe hands runs on CPU/NEON here, keep it cheap
        hand_min_det_conf=0.6,
        hand_min_track_conf=0.5,
        hand_num_threads=2,         # leave cores for capture + display threads
        capture_width=640,
        capture_height=480,
        capture_fps_request=30,
        display_scale=1.0,
        swipe_window_frames=10,     # fewer frames needed since we still target ~30fps but lower res
        swipe_min_norm_disp=1.4,
        swipe_max_vertical_ratio=0.6,
        gesture_cooldown_s=0.8,
    ),
}


def get_profile(name: str = None) -> DeviceProfile:
    name = name or os.environ.get("GESTURE_PROFILE", "dev_rtx3060")
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Choices: {list(PROFILES)}")
    return PROFILES[name]


# COCO-17 keypoint indices returned by YOLOv8-Pose (Ultralytics ordering)
COCO_KPT = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}
