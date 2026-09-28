"""
pose_estimator.py
==================
Thin wrapper around Ultralytics YOLOv8-Pose. This is the ONLY module that
touches the pose backbone, so swapping .pt -> .onnx -> .engine (TensorRT)
is a config change, not a code change: Ultralytics' `YOLO()` loader inspects
the file extension and picks the right runtime automatically.

Why YOLOv8-Pose instead of MediaPipe Pose here: it exports cleanly to
TensorRT for the Jetson (Section 3 of the brief), gives us proper
multi-person boxes + COCO-17 keypoints in one forward pass, and the same
checkpoint format runs unmodified on the RTX 3060 in fp16/fp32.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from ultralytics import YOLO

from config import DeviceProfile, COCO_KPT


@dataclass
class PersonPose:
    box_xyxy: np.ndarray          # (4,) pixel coords
    keypoints_xy: np.ndarray      # (17, 2) pixel coords
    keypoints_conf: np.ndarray    # (17,)
    track_id: Optional[int] = None

    def kp(self, name: str) -> np.ndarray:
        return self.keypoints_xy[COCO_KPT[name]]

    def kp_conf(self, name: str) -> float:
        return float(self.keypoints_conf[COCO_KPT[name]])


class PoseEstimator:
    def __init__(self, profile: DeviceProfile):
        self.profile = profile
        # Ultralytics auto-detects backend from the file suffix:
        #   .pt     -> PyTorch (eager)
        #   .onnx   -> ONNX Runtime
        #   .engine -> TensorRT (built via export_tensorrt.py, see that file)
        self.model = YOLO(profile.pose_weights, task="pose")
        # Warm up so the first real frame isn't penalized by lazy CUDA/TRT
        # context creation -- important for FPS measurements on the Orin Nano.
        dummy = np.zeros((profile.pose_imgsz, profile.pose_imgsz, 3), dtype=np.uint8)
        self.model.predict(
            dummy,
            device=profile.pose_device,
            imgsz=profile.pose_imgsz,
            half=profile.pose_half,
            verbose=False,
        )

    def infer(self, frame_bgr: np.ndarray) -> List[PersonPose]:
        """Run one forward pass and return per-person pose results."""
        results = self.model.predict(
            frame_bgr,
            device=self.profile.pose_device,
            imgsz=self.profile.pose_imgsz,
            half=self.profile.pose_half,
            conf=self.profile.pose_conf,
            max_det=self.profile.pose_max_det,
            verbose=False,
        )[0]

        people: List[PersonPose] = []
        if results.keypoints is None or results.boxes is None:
            return people

        boxes = results.boxes.xyxy.cpu().numpy()
        kpts_xy = results.keypoints.xy.cpu().numpy()      # (N, 17, 2)
        kpts_conf = (
            results.keypoints.conf.cpu().numpy()
            if results.keypoints.conf is not None
            else np.ones(kpts_xy.shape[:2])
        )

        for i in range(kpts_xy.shape[0]):
            people.append(
                PersonPose(
                    box_xyxy=boxes[i],
                    keypoints_xy=kpts_xy[i],
                    keypoints_conf=kpts_conf[i],
                )
            )
        return people
