"""Honest numbers: plain MediaPipe (the organisers' starter code) vs our pipeline,
measured on the same frames in the same run. No ground truth is needed for these;
for true accuracy use the labelling mode in the viewer.
"""
import numpy as np

from .tracker import SIDES

DETECTED = ("mp", "roi", "enh")


def _pct(x):
    return round(100.0 * x, 1)


def jitter(hist, attr):
    """Mean frame-to-frame acceleration of the 21 points, in % of hand size.
    Only over 3 consecutive *detected* frames, so filled frames don't flatter us."""
    vals = []
    for side in SIDES:
        for k in range(2, len(hist)):
            a, b, c = hist[k - 2][side], hist[k - 1][side], hist[k][side]
            if a is None or b is None or c is None:
                continue
            if not (a.src in DETECTED and b.src in DETECTED and c.src in DETECTED):
                continue
            if not (a.track_id == b.track_id == c.track_id):
                continue
            pa, pb, pc = getattr(a, attr), getattr(b, attr), getattr(c, attr)
            size = np.ptp(pb, axis=0).max() + 1e-6
            vals.append(np.linalg.norm(pa - 2 * pb + pc, axis=1).mean() / size)
    return round(100 * float(np.mean(vals)), 2) if vals else None


def compute(hist, raw, swap, timing, n_video_frames, extra):
    n = max(len(hist), 1)
    m = {"processed_frames": len(hist)}

    # --- detection share ---
    for side in SIDES:
        raw_hit = sum(any(r.label == side for r in fr) for fr in raw)      # labels as MediaPipe gives them
        det = sum(fr[side] is not None and fr[side].src in DETECTED for fr in hist)
        tot = sum(fr[side] is not None for fr in hist)
        m[f"{side}_share_baseline"] = _pct(raw_hit / n)
        m[f"{side}_share_detected"] = _pct(det / n)
        m[f"{side}_share_final"] = _pct(tot / n)
    m["any_hand_baseline"] = _pct(sum(len(fr) > 0 for fr in raw) / n)
    m["any_hand_final"] = _pct(sum(any(h is not None for h in fr.values()) for fr in hist) / n)
    m["both_hands_baseline"] = _pct(sum(len({r.label for r in fr}) == 2 for fr in raw) / n)
    m["both_hands_final"] = _pct(sum(all(h is not None for h in fr.values()) for fr in hist) / n)

    # --- left/right problems in the baseline ---
    flips = 0
    for side in SIDES:
        for k in range(1, len(hist)):
            a, b = hist[k - 1][side], hist[k][side]
            if a and b and a.src == "mp" and b.src == "mp" and a.track_id == b.track_id \
                    and a.raw_label != b.raw_label:
                flips += 1
    m["baseline_label_flips"] = flips
    m["baseline_same_label_pairs"] = sum(len(fr) == 2 and fr[0].label == fr[1].label for fr in raw)
    m["handedness_swapped_for_clip"] = bool(swap)

    # --- recovery ---
    for src in ("roi", "enh", "interp"):
        m[f"recovered_{src}"] = sum(fr[s] is not None and fr[s].src == src for fr in hist for s in SIDES)

    # --- jitter ---
    m["jitter_baseline"] = jitter(hist, "kp_raw")
    m["jitter_final"] = jitter(hist, "kp")
    if m["jitter_baseline"] and m["jitter_final"] is not None:
        m["jitter_reduction_pct"] = round(100 * (1 - m["jitter_final"] / m["jitter_baseline"]), 1)

    # --- speed ---
    wall = timing.get("total", 0.0)
    m["seconds_total"] = round(wall, 2)
    m["fps_processed"] = round(len(hist) / wall, 1) if wall else None
    m["fps_video_equiv"] = round(n_video_frames / wall, 1) if wall else None
    m["ms_per_frame"] = {k: round(1000 * v / n, 1) for k, v in timing.items()
                         if k not in ("total", "pass1")}
    m.update(extra)
    return m
