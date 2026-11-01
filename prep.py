"""Run ONCE before the event (allowed: "installing tools, downloading model weights").

Downloads the MediaPipe hand model and YOLO-World weights, then bakes our factory
vocabulary into a small model file so runs need no internet and no CLIP model.
"""
import urllib.request
from pathlib import Path

from egohand.config import Config

HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task")


def main():
    cfg = Config()
    Path("models").mkdir(exist_ok=True)
    if not Path(cfg.hand_model).exists():
        print("Downloading MediaPipe hand model ...")
        urllib.request.urlretrieve(HAND_URL, cfg.hand_model)
    print("MediaPipe model OK:", cfg.hand_model)
    try:
        from ultralytics import YOLO, YOLOWorld
    except ImportError:
        print("ultralytics not installed -> pip install ultralytics  (or use --objects none)")
        return
    YOLO("yolov8n.pt")                                # small COCO fallback (--objects coco)
    print("COCO fallback model OK: yolov8n.pt")
    try:
        from egohand.objects import ObjectDetector, model_path_for
        ObjectDetector(cfg)                           # builds + saves the model for this vocabulary
        print("YOLO-World with factory vocabulary saved:", model_path_for(cfg))
    except Exception as e:
        print("YOLO-World setup FAILED (use --objects coco until fixed):", repr(e))


if __name__ == "__main__":
    main()
