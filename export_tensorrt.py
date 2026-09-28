"""
export_tensorrt.py
===================
Builds a TensorRT engine from the YOLOv8-Pose PyTorch checkpoint.

*** IMPORTANT: run this ON the machine that will actually run inference. ***
TensorRT engines are serialized, hardware- and version-locked artifacts --
an .engine file built on the RTX 3060 (Ampere, desktop TensorRT/driver
stack) will NOT load on the Jetson Orin Nano (also Ampere-class iGPU, but a
different TensorRT minor version, different memory layout / JetPack CUDA
build) and vice versa. So the workflow is:

  1. Iterate on the RTX 3060 using the raw .pt checkpoint (config.py's
     "dev_rtx3060" profile) -- fast Python dev loop, no export needed.
  2. Once the model/weights are final, copy the .pt file to the Jetson and
     run THIS script there to produce yolov8n-pose.engine.
  3. Point config.py's "deploy_orin_nano".pose_weights at that .engine file.

On the Jetson, prefer running this via `jetson-containers`' l4t-pytorch /
ultralytics image, or after `pip install ultralytics` inside JetPack's
system Python where the NVIDIA-provided TensorRT/pycuda wheels are already
present -- building TensorRT/pycuda from source is painful and unnecessary.

Usage:
    python export_tensorrt.py --weights yolov8n-pose.pt --imgsz 416 --half
    python export_tensorrt.py --weights yolov8n-pose.pt --imgsz 640 --dynamic   # RTX 3060, if you DO want a local engine
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=416,
                         help="Lower (e.g. 416) trades a little accuracy for a lot of FPS on Orin Nano.")
    parser.add_argument("--half", action="store_true", default=True,
                         help="fp16 engine -- essentially free accuracy loss, big speed win on Jetson.")
    parser.add_argument("--int8", action="store_true",
                         help="int8 quantization -- needs a calibration dataset, max speed, more accuracy risk.")
    parser.add_argument("--dynamic", action="store_true",
                         help="Support variable input sizes (slightly slower, more flexible).")
    parser.add_argument("--workspace", type=int, default=4,
                         help="TensorRT builder workspace in GB. Keep modest on the 8GB Orin Nano.")
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    model = YOLO(args.weights, task="pose")

    print(f"Exporting {args.weights} -> TensorRT engine "
          f"(imgsz={args.imgsz}, half={args.half}, int8={args.int8}, dynamic={args.dynamic})")

    engine_path = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half and not args.int8,
        int8=args.int8,
        dynamic=args.dynamic,
        workspace=args.workspace,
        device=args.device,
        simplify=True,   # ONNX-simplify pass before TRT build, fewer op quirks
    )

    print(f"Done. Engine written to: {engine_path}")
    print("Point config.py's pose_weights at this file for the corresponding profile.")


if __name__ == "__main__":
    main()
