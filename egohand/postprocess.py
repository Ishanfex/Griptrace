"""Offline clean-up after tracking (we process recorded clips, so we may look ahead).

1. fix_sides   - give each whole track the side most of its evidence supports
2. fill_gaps   - interpolate short drop-outs (blur, occlusion by a tool)
3. smooth      - One Euro filter run forwards AND backwards (no lag, no jitter)
4. expand      - fill frames we skipped (step=2/3) by interpolating neighbours
"""
from collections import defaultdict

import numpy as np

from .tracker import OTHER, SIDES, Tracked, forearm_entry


# ---------------------------------------------------------------- 1. sides
def fix_sides(history, calibrated, width, height):
    """Vote per track: MediaPipe's (calibrated) label on every frame, plus a weak
    vote from where the hand sits in the image. Move a track to the other side
    only where that side is free (or its own track also wants to move)."""
    votes = defaultdict(float)
    for fr in history:
        for side, h in fr.items():
            if h is None:
                continue
            if h.src == "mp":
                votes[h.track_id] += 1.0 if calibrated(h.raw_label) == side else -1.0
            e = forearm_entry(h.kp, width, height)
            if e is not None and abs(e - 0.5) > 0.15:          # clear forearm direction
                votes[h.track_id] += 0.6 if (e < 0.5) == (side == "left") else -0.6
            nx = h.kp[:, 0].mean() / width
            if nx < 0.4 or nx > 0.6:
                pos_side = "left" if nx < 0.4 else "right"
                votes[h.track_id] += 0.3 if pos_side == side else -0.3
    counts = defaultdict(int)
    for fr in history:
        for h in fr.values():
            if h is not None:
                counts[h.track_id] += 1
    flip = {t for t, v in votes.items() if v < -max(3.0, 0.3 * counts[t])}
    n_fixed = 0
    for fr in history:
        new = {s: None for s in SIDES}
        for side, h in fr.items():
            if h is None:
                continue
            want = OTHER[side] if h.track_id in flip else side
            if new[want] is None:
                new[want] = h
                n_fixed += want != side
            else:                                      # both want the same side: keep original
                new[side] = h
        fr.update(new)
    return n_fixed


# ---------------------------------------------------------------- 2. gaps
def fill_gaps(history, max_gap, max_jump):
    n_filled = 0
    for side in SIDES:
        last = None
        for k, fr in enumerate(history):
            h = fr[side]
            if h is None:
                continue
            if last is not None and 1 < k - last <= max_gap + 1:
                a, b = history[last][side], h
                size = np.ptp(a.kp, axis=0).max() + 1e-6
                if np.linalg.norm(a.kp.mean(0) - b.kp.mean(0)) / size <= max_jump * (k - last):
                    for j in range(last + 1, k):
                        w = (j - last) / (k - last)
                        history[j][side] = Tracked(kp=(1 - w) * a.kp + w * b.kp,
                                                   score=min(a.score, b.score), src="interp",
                                                   raw_label="", track_id=a.track_id)
                        n_filled += 1
            last = k
    return n_filled


# ---------------------------------------------------------------- 3. smoothing
class OneEuro:
    """One Euro filter (Casiez et al. 2012) on whole arrays at once.
    Slow hand -> strong smoothing (kills jitter). Fast hand -> light smoothing (no lag)."""

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(abs(t - self.t), 1e-6)
        self.t = t
        self.dx += self._alpha(self.dc, dt) * ((x - self.x) / dt - self.dx)
        cutoff = self.mc + self.beta * np.abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        return self.x.copy()


def _runs(history, side):
    """Consecutive frames where this side's hand exists (same track)."""
    run, prev_tid = [], None
    for k, fr in enumerate(history):
        h = fr[side]
        if h is None or (run and (k != run[-1] + 1 or h.track_id != prev_tid)):
            if run:
                yield run
            run = []
        if h is not None:
            run.append(k)
            prev_tid = h.track_id
    if run:
        yield run


def smooth(history, times, cfg, diag):
    """Replace every hand's keypoints with smoothed ones; keeps the raw copy in .kp_raw."""
    for side in SIDES:
        for run in _runs(history, side):
            xs = [history[k][side].kp / diag for k in run]      # resolution independent
            ts = [times[k] for k in run]
            f = OneEuro(cfg.euro_min_cutoff, cfg.euro_beta, cfg.euro_d_cutoff)
            fwd = [f(x, t) for x, t in zip(xs, ts)]
            if cfg.zero_phase and len(run) > 2:
                b = OneEuro(cfg.euro_min_cutoff, cfg.euro_beta, cfg.euro_d_cutoff)
                bwd = [b(x, t) for x, t in zip(xs[::-1], ts[::-1])][::-1]
                out = [(p + q) / 2 for p, q in zip(fwd, bwd)]
            else:
                out = fwd
            for k, y in zip(run, out):
                h = history[k][side]
                h.kp_raw = h.kp
                h.kp = y * diag


# ---------------------------------------------------------------- 4. all frames
def expand(history, proc_idx, n_frames):
    """history is per processed frame; return one entry per video frame."""
    out = [dict.fromkeys(SIDES) for _ in range(n_frames)]
    for k, i in enumerate(proc_idx):
        if i < n_frames:
            out[i] = dict(history[k])
    if proc_idx:                                   # frames after the last processed one
        for i in range(proc_idx[-1] + 1, n_frames):
            out[i] = dict(history[-1])
    for k in range(len(proc_idx) - 1):
        i0, i1 = proc_idx[k], proc_idx[k + 1]
        for i in range(i0 + 1, min(i1, n_frames)):
            w = (i - i0) / (i1 - i0)
            for side in SIDES:
                a, b = history[k][side], history[k + 1][side]
                if a is not None and b is not None and a.track_id == b.track_id:
                    h = Tracked(kp=(1 - w) * a.kp + w * b.kp, score=min(a.score, b.score),
                                src="fill", raw_label="", track_id=a.track_id)
                    out[i][side] = h
                elif a is not None and w <= 0.5:
                    out[i][side] = a
                elif b is not None and w > 0.5:
                    out[i][side] = b
    return out
