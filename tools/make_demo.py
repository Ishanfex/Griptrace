"""Cut one demo video (<= 2 minutes total, the organisers' limit) from the rendered
outputs/<clip>/<clip>_demo.mp4 files: the busiest window of each clip (most grabs and
releases, both hands visible), with a 1.5 s title card before each.

    python -m tools.make_demo                 # all clips in outputs/
    python -m tools.make_demo --seconds 110   # total length
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def busiest_window(res, win_s):
    fps = res["meta"]["fps"]
    n = len(res["frames"])
    score = np.zeros(n)
    for i, fr in enumerate(res["frames"]):
        score[i] += 0.02 * sum(h is not None for h in fr)          # hands visible
    for s in res["segments"]:
        if s["state"] in ("portable", "fixed"):
            for t in (s["t_start"], s["t_end"]):                    # grab and release moments
                i = int(t * fps)
                if 0 <= i < n:
                    score[i] += 1.0
    w = max(1, int(win_s * fps))
    if n <= w:
        return 0, n
    csum = np.concatenate([[0], np.cumsum(score)])
    sums = csum[w:] - csum[:-w]
    start = int(np.argmax(sums))
    return start, start + w


def title_card(text, sub, size):
    W, H = size
    img = np.full((H, W, 3), (28, 23, 19), np.uint8)
    cv2.putText(img, text, (40, H // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (232, 239, 242), 2, cv2.LINE_AA)
    cv2.putText(img, sub, (40, H // 2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (58, 181, 232), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--seconds", type=float, default=110.0, help="total length (limit is 120)")
    ap.add_argument("--out", default="outputs/demo_2min.mp4")
    a = ap.parse_args()
    clips = sorted(p for p in Path(a.outputs).iterdir() if p.is_dir() and list(p.glob("*_demo.mp4")))
    if not clips:
        sys.exit("No rendered demos in outputs/. Run run.py first.")
    card_s = 1.5
    per = (a.seconds - card_s * len(clips)) / len(clips)
    writer = None
    total = 0.0
    for d in clips:
        demo = next(d.glob("*_demo.mp4"))
        res = json.load(open(next(d.glob("*.results.json"))))
        cap = cv2.VideoCapture(str(demo))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        size = (int(cap.get(3)), int(cap.get(4)))
        if writer is None:
            for cc in ("avc1", "mp4v"):
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*cc), fps, size)
                if writer.isOpened():
                    break
            out_size = size
        i0, i1 = busiest_window(res, per)
        grabs = sum(s["state"] == "portable" and i0 / fps <= s["t_start"] < i1 / fps for s in res["segments"])
        card = title_card(d.name, f"{i0 / fps:.0f}-{i1 / fps:.0f} s  |  {grabs} grabs in this window", out_size)
        for _ in range(int(card_s * fps)):
            writer.write(card)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
        for _ in range(i1 - i0):
            ok, img = cap.read()
            if not ok:
                break
            if (img.shape[1], img.shape[0]) != out_size:
                img = cv2.resize(img, out_size)
            writer.write(img)
        total += card_s + (i1 - i0) / fps
        print(f"{d.name}: {i0 / fps:.1f}-{i1 / fps:.1f} s, {grabs} grabs")
        cap.release()
    writer.release()
    print(f"-> {a.out}  ({total:.1f} s total)")


if __name__ == "__main__":
    main()
