"""PS2: is each hand touching nothing, a portable object or a fixed surface?

The organisers' rule is ">= 2 fingertips inside the object's box". In head-camera
video that alone fires constantly: hands are almost always *in front of* the bench
or hovering over parts in 2D. So a fingertip match must also have physical evidence:

  * fixed surface  -> the hand is STILL in scene coordinates (resting / pressing).
                      Scene coordinates = after removing head motion (camera.py).
  * portable object-> the fingers are GRIPPING (curled), or the object MOVES WITH
                      the hand.
  * a portable and a fixed object both qualify (tool lying on a bench) -> portable wins
  * a box that is really the hand itself (glove detected as object) is ignored
  * "sticky grasp": the detector often loses a held object because the hand covers
    it, so we keep it attached to the hand for up to `sticky_s`.
  * clearly gripping (grip >= unknown_grip) but no object detected at all
    -> "holding unknown object": honest, and better than calling the hand free.

Over time:
  * hysteresis - a new state must hold `enter_frames` (contact) / `exit_frames`
    (release) frames before it counts; the change is then dated back to when it
    really started, so timestamps are not delayed
  * motion check - a "fixed" object that moved in scene coordinates during the
    contact is re-labelled portable
"""
from collections import defaultdict

import numpy as np

from .camera import to_scene
from .config import MCPS, TIPS, WRIST
from .tracker import SIDES, _iou

NONE, PORTABLE, FIXED, NOT_VISIBLE = "none", "portable", "fixed", "not_visible"


def _hand_box(kp):
    return np.array([*kp.min(0), *kp.max(0)])


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def grip_score(kp):
    """0 = open hand, 1 = closed grip. Fingertip distance to the palm centre,
    divided by palm size (wrist -> knuckles), which does not change when fingers curl."""
    palm = kp[[WRIST] + MCPS].mean(0)
    scale = np.linalg.norm(kp[MCPS[1:]] - kp[WRIST], axis=1).mean() + 1e-6
    r = np.linalg.norm(kp[TIPS] - palm, axis=1).mean() / scale
    return float(np.clip((1.3 - r) / 0.6, 0.0, 1.0))


def _scene_tracks(hist, obj_frames, scene):
    """Hand and object centres in scene coordinates, per processed frame."""
    hands = {s: [None] * len(hist) for s in SIDES}
    objs = defaultdict(lambda: [None] * len(hist))
    for k, fr in enumerate(hist):
        for s in SIDES:
            if fr[s] is not None:
                hands[s][k] = to_scene(scene[k], fr[s].kp.mean(0))
        for tid, b in obj_frames[k].items():
            objs[tid][k] = to_scene(scene[k], ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return hands, objs


def _speed(track, k, times, w):
    """Path length travelled around frame k divided by time (pixels / s).
    Path length (not net displacement) so a hand turning around is not "still"."""
    a, b = max(0, k - w), min(len(track) - 1, k + w)
    pts = [(times[j], track[j]) for j in range(a, b + 1) if track[j] is not None]
    if len(pts) < 2:
        return None
    path = sum(np.linalg.norm(q[1] - p[1]) for p, q in zip(pts, pts[1:]))
    return float(path / max(pts[-1][0] - pts[0][0], 1e-6))


def frame_candidates(hist, obj_frames, kinds, times, cfg, W, H, scene=None):
    """Raw (un-smoothed in time) contact guess per processed frame and hand."""
    frame_area = W * H
    if scene is None:
        scene = [np.eye(3)] * len(hist)
    hand_sc, obj_sc = _scene_tracks(hist, obj_frames, scene)
    w = max(1, int(round(cfg.motion_window_s / max(times[1] - times[0], 1e-6)))) if len(times) > 1 else 1
    cand = [dict() for _ in hist]
    ghosts = [dict() for _ in hist]          # sticky boxes we drew without a detection
    held = {s: None for s in SIDES}          # side -> (tid, offset from hand centre, wh, t)
    for k, fr in enumerate(hist):
        for side in SIDES:
            h = fr[side]
            if h is None:
                cand[k][side] = (NOT_VISIBLE, None)
                continue
            hb = _hand_box(h.kp)
            hs = float(np.hypot(hb[2] - hb[0], hb[3] - hb[1])) + 1e-6
            pad = cfg.tip_pad * hs
            tips = h.kp[TIPS]
            grip = grip_score(h.kp)
            v_hand = _speed(hand_sc[side], k, times, w)
            still = v_hand is not None and v_hand / hs < cfg.still_sizes_per_s
            best = None
            for tid, b in obj_frames[k].items():
                ab, ah = _area(b), _area(hb)
                if _iou(b, hb) > 0.6 and 0.6 < ab / max(ah, 1) < 1.6:
                    continue                                  # this "object" is the hand itself
                inside = int(((tips[:, 0] > b[0] - pad) & (tips[:, 0] < b[2] + pad) &
                              (tips[:, 1] > b[1] - pad) & (tips[:, 1] < b[3] + pad)).sum())
                if inside < cfg.min_tips:
                    continue
                kind = kinds[tid]
                if ab > cfg.fixed_area_frac * frame_area:
                    kind = FIXED                              # huge box = surface
                if kind == FIXED:
                    if not still:
                        continue                              # hovering / moving over the bench
                else:
                    v_obj = _speed(obj_sc[tid], k, times, w)
                    comove = (v_obj is not None and v_hand is not None
                              and v_obj / hs > cfg.still_sizes_per_s and v_hand / hs > cfg.still_sizes_per_s)
                    if not (grip >= cfg.min_grip or comove):
                        continue                              # open hand hovering over a part
                # rank: portable first, then more fingertips, then smaller box
                key = (kind == PORTABLE, inside, -ab)
                if best is None or key > best[0]:
                    best = (key, tid, kind)
            if best is None and held[side] is not None:
                tid, off, wh, t0 = held[side]
                if times[k] - t0 <= cfg.sticky_s and tid not in obj_frames[k] and grip >= cfg.min_grip * 0.6:
                    c = h.kp.mean(0) + off
                    ghosts[k][tid] = np.array([c[0] - wh[0] / 2, c[1] - wh[1] / 2,
                                               c[0] + wh[0] / 2, c[1] + wh[1] / 2])
                    best = (None, tid, PORTABLE)
            if best is None and grip >= cfg.unknown_grip:
                cand[k][side] = (PORTABLE, None)              # clearly gripping, detector found nothing
                held[side] = None
                continue
            if best is None:
                cand[k][side] = (NONE, None)
                held[side] = None
                continue
            _, tid, kind = best
            cand[k][side] = (kind, tid)
            if kind == PORTABLE:
                b = obj_frames[k].get(tid)
                if b is not None:                             # refresh the sticky memory
                    c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                    held[side] = (tid, c - h.kp.mean(0), (b[2] - b[0], b[3] - b[1]), times[k])
            else:
                held[side] = None
    return cand, ghosts


def hysteresis(cand, cfg):
    """Remove flicker. Returns the cleaned state per frame and side."""
    n = len(cand)
    out = [dict() for _ in range(n)]
    for side in SIDES:
        cur = cand[0][side] if n else (NONE, None)
        pending, start, count = None, 0, 0
        states = []
        for k in range(n):
            c = cand[k][side]
            if c == cur:
                pending, count = None, 0
            elif c == pending:
                count += 1
            else:
                pending, start, count = c, k, 1
            if c[0] in (NONE, NOT_VISIBLE):
                need = cfg.exit_frames
            elif cur[0] in (PORTABLE, FIXED):
                need = cfg.switch_frames                  # object -> different object: be sure
            else:
                need = cfg.enter_frames
            if pending is not None and count >= need:
                for j in range(start, k):                     # date the change back
                    states[j] = pending
                cur, pending, count = pending, None, 0
            states.append(cur)
        for k in range(n):
            out[k][side] = states[k]
    return out


def segments(states, times, dt, names, cfg):
    segs = []
    for side in SIDES:
        run = None
        for k, fr in enumerate(states):
            s = fr[side]
            if run is None or s != run["key"]:
                if run:
                    segs.append(run)
                run = {"key": s, "hand": side, "k0": k, "k1": k}
            else:
                run["k1"] = k
        if run:
            segs.append(run)
    out = []
    for r in segs:
        state, tid = r["key"]
        out.append({"hand": r["hand"], "state": state, "obj": tid,
                    "object_name": names.get(tid, "") if tid is not None else
                    ("unknown object" if state == PORTABLE else ""),
                    "k0": r["k0"], "k1": r["k1"],
                    "t_start": round(times[r["k0"]], 3), "t_end": round(times[r["k1"]] + dt, 3)})
    # merge too-short contact blips into "none" (resting on a surface must last longer)
    for s in out:
        min_len = cfg.min_fixed_s if s["state"] == FIXED else cfg.min_portable_s
        if s["state"] in (PORTABLE, FIXED) and s["t_end"] - s["t_start"] < min_len:
            s["state"], s["obj"], s["object_name"] = NONE, None, ""
    merged = []
    for s in sorted(out, key=lambda s: (s["hand"], s["k0"])):
        p = merged[-1] if merged else None
        if p and p["hand"] == s["hand"] and p["state"] == s["state"] and \
                (p["obj"] == s["obj"] or (p["object_name"] and p["object_name"] == s["object_name"])):
            p["k1"], p["t_end"] = s["k1"], s["t_end"]
        else:
            merged.append(dict(s))
    return merged


def merge_segments(segs):
    """Join neighbouring segments of the same hand that now say the same thing."""
    out = []
    for s in sorted(segs, key=lambda s: (s["hand"], s["k0"])):
        p = out[-1] if out else None
        if p and p["hand"] == s["hand"] and p["state"] == s["state"] and \
                p.get("object_name", "") == s.get("object_name", "") and s["k0"] <= p["k1"] + 1:
            p["k1"], p["t_end"] = s["k1"], s["t_end"]
        else:
            out.append(dict(s))
    return out


def refine_kinds(segs, obj_frames, to_scene_list, cfg):
    """Fixed-labelled object that moved during the contact -> it was portable."""
    for s in segs:
        if s["state"] != FIXED or s["obj"] is None:
            continue
        pts, size = [], 0.0
        for k in range(s["k0"], s["k1"] + 1):
            b = obj_frames[k].get(s["obj"])
            if b is not None:
                c = to_scene(to_scene_list[k], ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
                pts.append(c)
                size = max(size, np.hypot(b[2] - b[0], b[3] - b[1]))
        if len(pts) >= 3 and size > 0:
            d = max(np.linalg.norm(p - pts[0]) for p in pts) / size
            s["moved"] = round(float(d), 2)
            if d > cfg.moved_sizes:
                s["state"] = PORTABLE
    return segs


def apply_segments(states, segs):
    """Write refined segment states back to per-frame states."""
    for s in segs:
        for k in range(s["k0"], s["k1"] + 1):
            states[k][s["hand"]] = (s["state"], s["obj"])
    return states


def tool_usage(segs):
    use = defaultdict(lambda: {"grabs": 0, "seconds": 0.0, "left": 0, "right": 0})
    for s in segs:
        if s["state"] == PORTABLE and s["object_name"] and s["object_name"] != "unknown object":
            u = use[s["object_name"]]
            u["grabs"] += 1
            u["seconds"] = round(u["seconds"] + s["t_end"] - s["t_start"], 2)
            u[s["hand"]] += 1
    return dict(sorted(use.items(), key=lambda kv: -kv[1]["seconds"]))


def safety_events(hist, states, obj_frames, hazard_ids, hazard_motion, names, times, dt, cfg):
    """Hand close to a hazard object that is currently moving (and not held by that hand)."""
    events = []
    for side in SIDES:
        cur = None
        for k, fr in enumerate(hist):
            h = fr[side]
            hit = None
            if h is not None:
                hb = _hand_box(h.kp)
                hs = np.hypot(hb[2] - hb[0], hb[3] - hb[1])
                for tid in hazard_ids:
                    b = obj_frames[k].get(tid)
                    if b is None or states[k][side][1] == tid:
                        continue
                    if hazard_motion[k].get(tid, 0.0) < cfg.motion_thresh:
                        continue
                    pad = cfg.hazard_pad * hs
                    tips = h.kp[TIPS]
                    near = ((tips[:, 0] > b[0] - pad) & (tips[:, 0] < b[2] + pad) &
                            (tips[:, 1] > b[1] - pad) & (tips[:, 1] < b[3] + pad)).any()
                    if near:
                        hit = tid
                        break
            if hit is not None and cur and cur["obj"] == hit and k - cur["k1"] <= 2:
                cur["k1"] = k
            elif hit is not None:
                if cur:
                    events.append(cur)
                cur = {"hand": side, "obj": hit, "object_name": names[hit], "k0": k, "k1": k}
        if cur:
            events.append(cur)
    out = []
    for e in events:
        if e["k1"] - e["k0"] >= 2:
            e["t_start"], e["t_end"] = round(times[e["k0"]], 3), round(times[e["k1"]] + dt, 3)
            out.append(e)
    return out
