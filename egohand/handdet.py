"""MediaPipe Hand Landmarker wrapper.

Two landmarker instances:
  * `video`  - VIDEO mode on full frames (uses MediaPipe's own frame-to-frame tracking)
  * `image`  - IMAGE mode, only used for recovery on zoomed crops / enhanced frames
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RawHand:
    kp: np.ndarray                 # (21, 2) pixel coords in the full frame
    label: str                     # MediaPipe's label, as returned ("left"/"right")
    score: float                   # handedness confidence
    src: str = "mp"                # mp | roi | enh  (how it was found)
    kp3d: np.ndarray = None        # (21, 3) metres, around the hand centre

    @property
    def bbox(self):
        x1, y1 = self.kp.min(0)
        x2, y2 = self.kp.max(0)
        return np.array([x1, y1, x2, y2], dtype=float)

    @property
    def size(self):
        b = self.bbox
        return float(np.hypot(b[2] - b[0], b[3] - b[1])) + 1e-6

    @property
    def center(self):
        b = self.bbox
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def _to_hands(res, w, h, ox=0.0, oy=0.0, sx=1.0, sy=1.0, src="mp"):
    """Convert a MediaPipe result into RawHands in full-frame pixel coords.
    (ox, oy) = crop offset, (sx, sy) = crop-pixels -> frame-pixels scale."""
    out = []
    worlds = res.hand_world_landmarks or [None] * len(res.hand_landmarks)
    for pts, hd, wpts in zip(res.hand_landmarks, res.handedness, worlds):
        kp = np.array([[p.x * w * sx + ox, p.y * h * sy + oy] for p in pts], dtype=float)
        kp3d = None if wpts is None else np.array([[p.x, p.y, p.z] for p in wpts], dtype=float)
        out.append(RawHand(kp=kp, label=hd[0].category_name.lower(),
                           score=float(hd[0].score), src=src, kp3d=kp3d))
    return out


class MediaPipeHands:
    def __init__(self, cfg):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        model = Path(cfg.hand_model)
        if not model.exists():
            raise FileNotFoundError(
                f"{model} not found. Run:  python prep.py   (downloads the MediaPipe model)")
        self.mp = mp

        def make(mode, n):
            return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(model)),
                running_mode=mode, num_hands=n,
                min_hand_detection_confidence=cfg.det_conf,
                min_hand_presence_confidence=cfg.presence_conf,
                min_tracking_confidence=cfg.track_conf))

        self.video = make(vision.RunningMode.VIDEO, cfg.num_hands)
        self.image = make(vision.RunningMode.IMAGE, 2)   # crop may hold both hands; tracker picks
        self.last_ts = -1

    def _img(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

    def detect(self, bgr, t_ms):
        t_ms = max(int(t_ms), self.last_ts + 1)       # VIDEO mode needs increasing timestamps
        self.last_ts = t_ms
        h, w = bgr.shape[:2]
        return _to_hands(self.video.detect_for_video(self._img(bgr), t_ms), w, h)

    def detect_image(self, bgr, src="enh"):
        h, w = bgr.shape[:2]
        return _to_hands(self.image.detect(self._img(bgr)), w, h, src=src)

    def detect_crop(self, bgr, box, size):
        """Run the landmarker on a square crop around `box`, zoomed to `size` px."""
        H, W = bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return []
        crop = cv2.resize(bgr[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_LINEAR)
        res = self.image.detect(self._img(crop))
        return _to_hands(res, size, size, ox=x1, oy=y1,
                         sx=(x2 - x1) / size, sy=(y2 - y1) / size, src="roi")

    def close(self):
        self.video.close()
        self.image.close()
