# 2D Skeletal Tracking + Gesture Recognition

Real-time webcam pipeline: YOLOv8-Pose (body skeleton, PyTorch/TensorRT) +
MediaPipe Hands (fingertip detail, run only on a wrist-cropped ROI) feeding
a lightweight angle/distance-based gesture classifier for **Stop** (static
open palm) and **Swipe Right** (dynamic wrist displacement).

## Why this split (YOLO-Pose + MediaPipe Hands)?

The brief needs elbow/wrist/fingertip geometry. YOLOv8-Pose's COCO-17
skeleton has shoulder/elbow/wrist but no finger joints; MediaPipe Hands has
21 finger landmarks but no elbow. Running full-frame MediaPipe Hands *and*
a body model every frame is wasteful, so:

1. YOLOv8-Pose runs once on the full frame (GPU, TensorRT on Jetson) →
   gives elbow/wrist and a person bounding box.
2. MediaPipe Hands runs only on a small crop centered on the detected wrist
   (CPU, XNNPACK) → gives the 5 fingertips.

This keeps the two stages on different compute resources (GPU vs CPU),
which matters a lot on the Jetson's shared-memory architecture, and avoids
ever running hand detection on a full 720p frame.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Hardware profiles (`dev_rtx3060`, `deploy_orin_nano`) — weights path, imgsz, precision, capture res, gesture timing. |
| `video_stream.py` | Threaded webcam grab, decoupled from inference. |
| `pose_estimator.py` | YOLOv8-Pose wrapper (auto-picks `.pt`/`.onnx`/`.engine` backend). |
| `hand_landmarker.py` | MediaPipe Hands on a wrist-centered ROI. |
| `gesture_classifier.py` | `StopDetector`, `SwipeRightDetector`, `GestureEngine` (angles + distances, no extra ML model). |
| `visualizer.py` | All drawing/HUD code — zero inference logic. |
| `pipeline.py` | Orchestrates capture → inference → display; `run_inference_step()` is pure and unit-testable. |
| `export_tensorrt.py` | Builds a TensorRT `.engine` **on the target device**. |

## Setup — RTX 3060 (development)

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Ultralytics auto-downloads yolov8n-pose.pt on first run, or fetch it yourself:
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

GESTURE_PROFILE=dev_rtx3060 python pipeline.py
```

## Deployment — Jetson Orin Nano

```bash
# 1. Flash JetPack 6.x, which ships CUDA/cuDNN/TensorRT for the Orin Nano.
# 2. Install NVIDIA's prebuilt torch/torchvision wheel for your JetPack version
#    (PyPI torch has no Jetson/Tegra CUDA support).
pip install ultralytics --no-deps
pip install onnx onnxsim opencv-python mediapipe numpy

# 3. Copy your trained/verified yolov8n-pose.pt from the RTX 3060 box here, then
#    build the engine ON THIS DEVICE (engines are not portable across GPUs):
python export_tensorrt.py --weights yolov8n-pose.pt --imgsz 416 --half

# 4. Run with the Jetson profile (expects yolov8n-pose.engine):
GESTURE_PROFILE=deploy_orin_nano python pipeline.py --headless
```

`--headless` skips `cv2.imshow` for a true edge deployment (e.g. running as
a systemd service over SSH with no display attached); gesture events still
print to stdout / can be piped into whatever downstream system consumes them.

## Tuning gestures

All thresholds live in `config.py` (`swipe_min_norm_disp`,
`swipe_window_frames`, etc.) and `gesture_classifier.py`
(`StopDetector.MIN_*`). They're expressed as ratios/angles rather than raw
pixels so they hold up across resolutions and distances from the camera —
but webcam FOV, mounting height, and lighting will still shift them, so
plan to calibrate once against your actual camera before locking values in.

## Performance notes

- `pose_imgsz` is the single biggest FPS lever: 640 on the RTX 3060 for
  accuracy headroom, 416 on the Orin Nano for throughput.
- `pose_max_det` capped low (2) on Jetson: fewer boxes → less NMS/keypoint
  decode work, and this use case only ever expects 1-2 people in frame.
- MediaPipe's `model_complexity=0` ("lite") on Jetson keeps the CPU-side
  hand stage cheap enough to not starve the capture/display threads.
- `ThreadedVideoStream` + MJPG capture avoids `cap.read()` blocking the
  main loop and gets better throughput than raw YUYV on most USB webcams.
