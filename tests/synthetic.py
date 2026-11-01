"""Synthetic stress test with KNOWN ground truth.

Real clips have no labels, so to prove each fix works we generate a clip where we
know everything, and a fake detector that makes MediaPipe's typical mistakes:
jitter, missed frames (incl. blur bursts), mirrored + flickering left/right labels,
and a co-worker's hand in view. The object detector misses objects, more often
when a hand covers them.

    python -m tests.synthetic            # prints baseline vs ours vs ground truth
"""
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from egohand.handdet import RawHand

FPS, W, H, SECONDS = 30, 640, 360, 14
N = FPS * SECONDS
rng = np.random.default_rng(7)

# 21-point right hand, wrist at origin, fingers up, ~unit size
_OPEN = np.array([[0, 0], [-.25, -.15], [-.42, -.32], [-.55, -.48], [-.65, -.62],
                  [-.2, -.55], [-.22, -.8], [-.23, -.98], [-.24, -1.12],
                  [0, -.6], [0, -.88], [0, -1.08], [0, -1.24],
                  [.18, -.55], [.2, -.8], [.21, -.98], [.22, -1.1],
                  [.33, -.45], [.38, -.65], [.4, -.8], [.42, -.92]])
_GRIP = _OPEN.copy()
for tip, base in ((4, 2), (8, 6), (12, 10), (16, 14), (20, 18)):
    _GRIP[tip] = _OPEN[base] * 0.85 + np.array([0, .12])     # fingertips curl in


def hand_pts(side, wrist, size, grip=0.0):
    p = (1 - grip) * _OPEN + grip * _GRIP
    if side == "left":
        p = p * np.array([-1, 1])
    return wrist + p * size


# ---------------- scene (in "world" pixels; camera views a moving window) ----------------
BENCH = np.array([60, 250, 900, 520.0])              # fixed surface
WRENCH0 = np.array([520, 255, 600, 290.0])            # portable object resting on bench


def camera_offset(f):
    t = f / FPS
    return np.array([60 + 40 * math.sin(t * 0.7) + 3 * math.sin(t * 9),
                     30 + 15 * math.sin(t * 0.5 + 1) + 2 * math.sin(t * 11)])


def gt_frame(f):
    """Ground truth for frame f in image coords."""
    t = f / FPS
    off = camera_offset(f)
    size = 95.0
    # right hand: reaches the wrench at 3s, lifts and moves it 3.5-6.5s, puts back, leaves
    r_wrist = np.array([470 + 45 * math.sin(t * 1.4), 355.0 - 15 * math.sin(t * 2.1)])
    grip_r, wrench = 0.0, WRENCH0.copy() - np.r_[off, off]
    r_state = "none"
    if 2.5 <= t < 3.2:
        a = (t - 2.5) / 0.7
        target = np.array([(wrench[0] + wrench[2]) / 2 + 5, wrench[3] + 55])
        r_wrist = (1 - a) * r_wrist + a * target
    elif 3.2 <= t < 7.0:
        lift = math.sin((t - 3.2) / 3.8 * math.pi) * 90
        shift = np.array([-lift * 0.8, -lift])
        wrench = wrench + np.r_[shift, shift]
        r_wrist = np.array([(wrench[0] + wrench[2]) / 2 + 5, wrench[3] + 55])
        grip_r, r_state = 1.0, "portable"
    elif 7.0 <= t < 7.7:
        a = (t - 7.0) / 0.7
        start = np.array([(wrench[0] + wrench[2]) / 2 + 5, wrench[3] + 55])
        r_wrist = (1 - a) * start + a * np.array([470, 355.0])
    # KNOWN HARD CASE: 12.0-13.0s the right hand hovers perfectly still over the bench
    # without touching it. From 2D video alone this looks like resting; we expect to fail.
    if 11.8 <= t < 13.0:                                  # glides in (0.4 s), then holds still
        hover = np.array([460.0, 360.0]) - off
        r_wrist = r_wrist + min(1.0, (t - 11.8) / 0.4) * (hover - r_wrist)
    # right hand out of view 10.5-11.5s (re-entry test)
    r_vis = not (10.5 <= t < 11.5)
    # left hand: presses on the bench 8-10s, otherwise idle on the left
    l_wrist = np.array([170 + 60 * math.sin(t * 1.6), 350 - 20 * math.cos(t * 1.1)])
    l_state = "none"
    if 8.0 <= t < 10.0:                                   # resting on the bench: still in the SCENE
        l_wrist = np.array([240.0, 380.0]) - off
        l_state = "fixed"
    hands = {"left": hand_pts("left", l_wrist, size),
             "right": hand_pts("right", r_wrist, size, grip_r) if r_vis else None}
    states = {"left": l_state, "right": r_state if r_vis else "not_visible"}
    bench = BENCH - np.r_[off, off]
    return hands, states, {"wrench": wrench, "workbench": bench}


def make_video(path):
    world = np.zeros((700, 1100, 3), np.uint8)
    world[:] = (70, 75, 80)
    for _ in range(400):                                  # texture for camera tracking
        x, y = rng.integers(0, 1100), rng.integers(0, 700)
        c = tuple(int(v) for v in rng.integers(40, 200, 3))
        cv2.circle(world, (int(x), int(y)), int(rng.integers(2, 9)), c, -1)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(N):
        off = camera_offset(f).astype(int)
        img = world[off[1]:off[1] + H, off[0]:off[0] + W].copy()
        hands, _, objs = gt_frame(f)
        b = objs["workbench"].astype(int)
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (60, 110, 150), -1)
        b = objs["wrench"].astype(int)
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (180, 180, 200), -1)
        for kp in hands.values():
            if kp is not None:
                for p in kp.astype(int):
                    cv2.circle(img, tuple(p), 6, (120, 160, 210), -1)
        if 5.0 <= f / FPS < 6.0:                          # a dark stretch
            img = (img * 0.25).astype(np.uint8)
        wr.write(img)
    wr.release()


class FakeHands:
    """Behaves like MediaPipe, including its typical mistakes."""

    def __init__(self, fps=FPS):
        self.fps = fps
        self.frame = 0
        self.burst = 0

    def _noisy(self, kp, sigma=2.5):
        return kp + rng.normal(0, sigma, kp.shape)

    def detect(self, bgr, t_ms):
        f = int(round(t_ms / 1000 * self.fps))
        self.frame = f
        hands, _, _ = gt_frame(f)
        if self.burst > 0:
            self.burst -= 1
        elif rng.random() < 0.02:
            self.burst = int(rng.integers(3, 8))          # motion-blur burst: nothing found
        out = []
        for side, kp in hands.items():
            if kp is None or self.burst > 0 or rng.random() < 0.12:
                continue
            mirrored = "right" if side == "left" else "left"   # MediaPipe's selfie assumption
            label = mirrored if rng.random() > 0.12 else side  # plus random flicker
            out.append(RawHand(kp=self._noisy(kp), label=label, score=0.9))
        if rng.random() < 0.15:                           # co-worker's small hand far away
            out.append(RawHand(kp=hand_pts("right", np.array([320, 80.0]), 18) + rng.normal(0, 1, (21, 2)),
                               label="left", score=0.7))
        return out

    def detect_crop(self, bgr, box, size):
        hands, _, _ = gt_frame(self.frame)
        for side, kp in hands.items():
            if kp is None:
                continue
            c = kp.mean(0)
            if box[0] < c[0] < box[2] and box[1] < c[1] < box[3] and rng.random() < 0.6:
                return [RawHand(kp=self._noisy(kp, 3.0), label="left" if side == "right" else "right",
                                score=0.8, src="roi")]
        return []

    def detect_image(self, bgr, src="enh"):
        return []


class FakeObjects:
    backend = "world"

    def __init__(self, hands):
        self.h = hands

    def detect(self, bgr, hand_boxes=()):
        _, states, objs = gt_frame(self.h.frame)
        out = []
        for name, b in objs.items():
            miss = 0.5 if (name == "wrench" and states["right"] == "portable") else 0.15
            if rng.random() < miss:
                continue
            out.append({"bbox": b + rng.normal(0, 3, 4), "name": name, "conf": 0.6})
        return out

    def kind(self, name):
        return "fixed" if name == "workbench" else "portable"


# ------------------------------------------------------------------ evaluation
def evaluate(res, step):
    per_frame, states, raw = res["per_frame"], res["states"], res["raw"]
    n = len(per_frame)
    vis = side_ok = present = err_n = 0
    err_ours, err_base, base_side_ok, base_present = [], [], 0, 0
    st_ok = st_n = 0
    for f in range(n):
        hands, gts, _ = gt_frame(f)
        for side, kp in hands.items():
            if kp is None:
                continue
            vis += 1
            h = per_frame[f][side]
            if h is not None:
                present += 1
                err_ours.append(np.linalg.norm(h.kp - kp, axis=1).mean())
                other = hands["left" if side == "right" else "right"]
            # side correctness: is the hand drawn under this side really this hand?
            if h is not None and np.linalg.norm(h.kp.mean(0) - kp.mean(0)) < 40:
                side_ok += 1
            if f % step == 0:                              # baseline exists on processed frames
                rr = raw[f // step]
                match = [r for r in rr if np.linalg.norm(r.kp.mean(0) - kp.mean(0)) < 40]
                if match:
                    base_present += 1
                    err_base.append(np.linalg.norm(match[0].kp - kp, axis=1).mean())
                    base_side_ok += match[0].label == side
            if gts[side] in ("portable", "fixed", "none"):
                st_n += 1
                st_ok += states[f][side][0] == gts[side]
    n_proc_vis = sum(1 for f in range(0, n, step) for kp in gt_frame(f)[0].values() if kp is not None)
    return {
        "hand_found_%": (round(100 * base_present / n_proc_vis, 1), round(100 * present / vis, 1)),
        "side_correct_%": (round(100 * base_side_ok / n_proc_vis, 1), round(100 * side_ok / vis, 1)),
        "kp_error_px": (round(float(np.mean(err_base)), 2), round(float(np.mean(err_ours)), 2)),
        "contact_state_acc_%": (None, round(100 * st_ok / st_n, 1)),
    }


def run_once(cfg=None, out=None, render=False):
    from egohand.config import Config
    from egohand.pipeline import run
    cfg = cfg or Config(render=render)
    cfg.render = render
    tmp = Path(out or tempfile.mkdtemp())
    clip = tmp / "synthetic.mp4"
    if not clip.exists():
        make_video(clip)
    hands = FakeHands()
    res = run(clip, tmp / "out", cfg, hand_detector=hands, object_detector=FakeObjects(hands),
              log=lambda *a: None)
    return res, evaluate(res, cfg.step), tmp


if __name__ == "__main__":
    res, ev, tmp = run_once(out="/tmp/egosyn" if False else None, render=True)
    print("                      baseline   ours")
    for k, (b, o) in ev.items():
        print(f"  {k:22s} {str(b):>8}   {o}")
    print("segments:")
    for s in res["segments"]:
        if s["state"] != "none":
            print("  ", s["hand"], s["state"], s["object_name"], s["t_start"], s["t_end"], s.get("moved"))
    print("metrics:", {k: v for k, v in res["metrics"].items() if k != "ms_per_frame"})
    print("output in", tmp)
