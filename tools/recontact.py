"""Re-run ONLY the contact stage on an existing <clip>.results.json.

Hands and object boxes are reused from the file, so tuning contact settings takes
seconds instead of re-running MediaPipe + YOLO. Head motion is re-estimated from the clip.

    python -m tools.recontact outputs/X/X.results.json egohack-data/clips/X.mp4 [--write]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egohand import contact as C  # noqa: E402
from egohand.camera import CameraMotion  # noqa: E402
from egohand.config import Config  # noqa: E402
from egohand.export import SRC  # noqa: E402
from egohand.tracker import SIDES, Tracked  # noqa: E402
from egohand.video import frames  # noqa: E402


def load(path, clip, cfg):
    d = json.load(open(path))
    meta = d["meta"]
    W, H = meta["process_width"], int(round(meta["height"] * meta["process_width"] / meta["width"]))
    scale = meta["width"] / W
    src_names = list(SRC)
    hist = []
    for i in d["proc_idx"]:
        fr = {}
        for n, side in enumerate(SIDES):
            r = d["frames"][i][n]
            fr[side] = None if r is None else Tracked(
                kp=np.array(r[3:], float).reshape(21, 2) / scale, score=0.9, src=src_names[r[0]],
                raw_label="", track_id=0)
        hist.append(fr)
    obj_frames = [{o[0]: np.array(o[1:5], float) / scale for o in row if not o[5]} for row in d["objects"]]
    info = {int(k): v for k, v in d["object_info"].items()}
    names = {k: v["name"] for k, v in info.items()}
    kinds = {k: ("fixed" if v["name"] in cfg.vocab_fixed or v["kind"] == "fixed" else "portable")
             for k, v in info.items()}
    for row in obj_frames:                          # same "giant portable box" rule as the detector
        for t in [t for t, b in row.items() if kinds.get(t) == "portable" and
                  (b[2] - b[0]) * (b[3] - b[1]) > cfg.portable_max_area * W * H]:
            del row[t]
    times = [i / meta["fps"] for i in d["proc_idx"]]
    cam = CameraMotion(W, H)
    for k, (_, _, img) in enumerate(frames(clip, meta["step"], W)):
        if k >= len(hist):
            break
        cam.update(img, [h.kp.min(0).tolist() + h.kp.max(0).tolist() for h in hist[k - 1].values() if h]
                   if k else [])
    return d, hist, obj_frames, names, kinds, times, cam.to_scene[:len(hist)], W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("clip")
    ap.add_argument("--write", action="store_true", help="overwrite segments in the results file")
    a = ap.parse_args()
    cfg = Config()
    d, hist, obj_frames, names, kinds, times, scene, W, H = load(a.results, a.clip, cfg)
    dt = d["meta"]["step"] / d["meta"]["fps"]
    cand, ghosts = C.frame_candidates(hist, obj_frames, kinds, times, cfg, W, H, scene)
    states = C.hysteresis(cand, cfg)
    segs = C.refine_kinds(C.segments(states, times, dt, names, cfg), obj_frames, scene, cfg)
    contacts = [s for s in segs if s["state"] in ("portable", "fixed")]
    print(f"{len(contacts)} contacts:")
    for s in contacts:
        print(f'  {s["t_start"]:6.2f}-{s["t_end"]:6.2f}  {s["hand"]:5s} {s["state"]:8s} {s["object_name"]}')
    if a.write:
        d["segments"] = [{k: v for k, v in s.items() if k != "key"} for s in segs]
        d["tools"] = C.tool_usage(segs)
        json.dump(d, open(a.results, "w"), separators=(",", ":"))
    return segs


if __name__ == "__main__":
    main()
