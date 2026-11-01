"""Object detection (YOLO-World, open vocabulary, CPU) + a small IoU tracker.

The detector runs only every `object_every` processed frames (it is the slowest
step on a CPU); boxes are interpolated in between.
"""
from collections import Counter
from pathlib import Path

import numpy as np

from .tracker import _iou

COCO_PORTABLE = {"bottle", "cup", "scissors", "knife", "cell phone", "book", "remote",
                 "keyboard", "mouse", "spoon", "fork", "backpack", "handbag", "suitcase",
                 "laptop", "toothbrush", "hair drier", "vase", "sports ball"}
COCO_FIXED = {"dining table", "chair", "bench", "refrigerator", "oven", "sink", "microwave",
              "tv", "toilet", "bed", "couch"}


def model_path_for(cfg):
    """One saved model per vocabulary, so editing the word lists rebuilds it automatically."""
    import hashlib
    from .config import vocabulary
    h = hashlib.md5("|".join(vocabulary(cfg)).encode()).hexdigest()[:8]
    return str(Path(cfg.object_model).with_name(f"yoloworld_{h}.pt"))


def _nms(dets, iou_thr=0.5):
    keep = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if all(d["name"] != k["name"] or _iou(d["bbox"], k["bbox"]) < iou_thr for k in keep):
            keep.append(d)
    return keep


class ObjectDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = cfg.objects
        if self.backend == "none":
            return
        try:
            from ultralytics import YOLO, YOLOWorld
        except ImportError as e:
            raise ImportError("pip install ultralytics   (or run with --objects none)") from e
        if self.backend == "world":
            from .config import vocabulary
            path = model_path_for(cfg)
            if Path(path).exists():                       # built before with this exact vocabulary
                self.model = YOLO(path)
            else:                                         # needs the CLIP text model once (cached by prep.py)
                self.model = YOLOWorld("yolov8s-worldv2.pt")
                self.model.set_classes(vocabulary(cfg))
                Path(path).parent.mkdir(exist_ok=True)
                self.model.save(path)
        else:
            self.model = YOLO("yolov8n.pt")

    def _run(self, img, imgsz, ox=0.0, oy=0.0, sx=1.0, sy=1.0):
        r = self.model.predict(img, conf=self.cfg.object_conf, imgsz=imgsz, verbose=False, device="cpu")[0]
        out = []
        for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy().astype(int)):
            name = r.names[cls]
            if name == "person":
                continue
            b = np.array([box[0] * sx + ox, box[1] * sy + oy, box[2] * sx + ox, box[3] * sy + oy], float)
            out.append({"bbox": b, "name": name, "conf": float(conf)})
        return out

    def detect(self, bgr, hand_boxes=()):
        """Full-frame detection, plus (hand_zoom) a zoomed crop around each hand:
        held objects are small and half-hidden by fingers, the zoom finds far more of them."""
        if self.backend == "none":
            return []
        H, W = bgr.shape[:2]
        dets = self._run(bgr, self.cfg.object_imgsz)
        if self.cfg.hand_zoom:
            for hb in hand_boxes:
                cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
                half = max(hb[2] - hb[0], hb[3] - hb[1]) * self.cfg.hand_zoom_scale / 2
                x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
                x2, y2 = int(min(W, cx + half)), int(min(H, cy + half))
                if x2 - x1 < 32 or y2 - y1 < 32:
                    continue
                crop = bgr[y1:y2, x1:x2]
                for d in self._run(crop, self.cfg.hand_zoom_imgsz, ox=x1, oy=y1):
                    d["zoom"] = True
                    dets.append(d)
        frame_area = W * H
        dets = [d for d in dets if not (self.kind(d["name"]) == "portable" and
                                        _area(d["bbox"]) > self.cfg.portable_max_area * frame_area)]
        return _nms(dets)

    def kind(self, name):
        if name in self.cfg.vocab_fixed or name in COCO_FIXED:
            return "fixed"
        return "portable"


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


class ObjectTracker:
    """Gives each object a stable id so tool-use counts are right."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}            # id -> dict(names Counter, obs {k: bbox}, last_k)
        self.next_id = 0

    def update(self, dets, k):
        live = [t for t in self.tracks.values() if k - t["last_k"] <= self.cfg.obj_ttl]
        pairs = []
        for di, d in enumerate(dets):
            for t in live:
                tb = t["obs"][t["last_k"]]
                iou = _iou(d["bbox"], tb)
                same = d["name"] == t["names"].most_common(1)[0][0]
                if iou >= self.cfg.obj_iou_match:
                    pairs.append((iou + (0.2 if same else 0.0), di, t["id"]))
                elif same:
                    # moving object missed for a few runs: match by distance, same name only
                    diag = np.hypot(tb[2] - tb[0], tb[3] - tb[1]) + 1e-6
                    dist = np.linalg.norm((d["bbox"][:2] + d["bbox"][2:]) / 2 - (tb[:2] + tb[2:]) / 2)
                    if dist < 1.5 * diag:
                        pairs.append((0.25 * (1 - dist / (1.5 * diag)), di, t["id"]))
        used_d, used_t = set(), set()
        for _, di, tid in sorted(pairs, reverse=True):
            if di in used_d or tid in used_t:
                continue
            used_d.add(di), used_t.add(tid)
            t = self.tracks[tid]
            t["obs"][k] = dets[di]["bbox"]
            t["names"][dets[di]["name"]] += dets[di]["conf"]
            t["last_k"] = k
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks[self.next_id] = {"id": self.next_id, "obs": {k: d["bbox"]},
                                             "names": Counter({d["name"]: d["conf"]}), "last_k": k}
                self.next_id += 1

    def name(self, tid):
        return self.tracks[tid]["names"].most_common(1)[0][0]

    def boxes_per_frame(self, n_proc, hold):
        """For every processed frame: {track_id: bbox}. Linear interpolation between
        detector runs, held for `hold` frames after the last sighting."""
        frames = [dict() for _ in range(n_proc)]
        for tid, t in self.tracks.items():
            ks = sorted(t["obs"])
            if len(ks) == 1 and t["names"].total() < 0.25:
                continue                                  # single weak sighting = noise
            for a, b in zip(ks, ks[1:]):
                for k in range(a, b):
                    if b - a > hold * 2:                  # long gap: don't invent a path
                        if k - a <= hold:
                            frames[k][tid] = t["obs"][a]
                        continue
                    w = (k - a) / (b - a)
                    frames[k][tid] = (1 - w) * t["obs"][a] + w * t["obs"][b]
            for k in range(ks[-1], min(n_proc, ks[-1] + hold + 1)):
                frames[k][tid] = t["obs"][ks[-1]]
        return frames
