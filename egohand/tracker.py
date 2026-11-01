"""Keeps a stable LEFT and RIGHT identity for the camera wearer's two hands.

Why not trust MediaPipe's labels directly?
  * labels are for mirrored selfies (wrong for head cameras) -> per-clip auto calibration
  * labels flicker on blurred / gloved hands               -> identity comes from motion
  * other workers' hands appear in view                     -> size filter + only 2 slots
  * hands leave and re-enter                                -> slot "birth" uses label + position

Each frame we try every way of assigning detections to the two slots and keep the
cheapest (at most 4 detections -> at most 21 combinations, so brute force is instant).
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from .handdet import RawHand

SIDES = ("left", "right")
OTHER = {"left": "right", "right": "left"}
ASSIGN_REWARD = 2.5          # assigning a detection must cost less than this to be worth it


@dataclass
class Tracked:
    kp: np.ndarray
    score: float
    src: str            # mp | roi | enh | interp | fill
    raw_label: str      # what MediaPipe said (for the baseline swap metric)
    track_id: int = -1  # changes every time a hand is lost and found again


class Slot:
    def __init__(self, side):
        self.side = side
        self.reset()

    def reset(self):
        self.active = False
        self.center = None
        self.vel = np.zeros(2)
        self.size = 0.0
        self.bbox = None
        self.last_k = -10**9
        self.votes = deque()         # +1 label agrees with slot side, -1 disagrees
        self.track_id = -1

    def predict(self, k):
        return self.center + self.vel * (k - self.last_k)

    def lost(self, k):
        return k - self.last_k - 1


def finger_up(kp):
    """+1 = fingers point up the image (away from the wearer), -1 = pointing down."""
    v = kp[9] - kp[0]                                  # wrist -> middle-finger knuckle
    return float(-v[1] / (np.linalg.norm(v) + 1e-6))


def forearm_entry(kp, W, H):
    """Where the forearm enters the frame: follow the line knuckle -> wrist on to the
    bottom edge. Returns x in 0..1 (0 = left edge), or None if fingers point down.
    Works when hands cross: the arms still come from their own side of the body."""
    v = kp[9] - kp[0]
    if v[1] > -1e-3:
        return None
    return float((kp[0, 0] - v[0] / v[1] * (H - kp[0, 1])) / W)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class HandTracker:
    def __init__(self, cfg, width, height):
        self.cfg = cfg
        self.W, self.H = width, height
        self.diag = float(np.hypot(width, height))
        self.slots = {s: Slot(s) for s in SIDES}
        self.history = []            # one dict per processed frame: side -> Tracked | None
        self.swap_votes = 0          # >0: MediaPipe labels are mirrored for this clip
        self.repairs = []            # (processed_frame, description)
        self.n_rejected_small = 0
        self.n_rejected_other = 0
        self.next_id = 0
        mode = str(cfg.swap_handedness).lower()
        self.swap_mode = "auto" if mode == "auto" else ("on" if mode in ("1", "true", "on", "yes") else "off")

    # ---------- left/right label calibration ----------
    @property
    def swap(self):
        if self.swap_mode != "auto":
            return self.swap_mode == "on"
        return self.swap_votes >= 0      # no evidence yet: MediaPipe docs say head-cam needs swap

    def calibrated(self, label):
        return OTHER[label] if self.swap else label

    def _vote_calibration(self, dets):
        """Two clearly separated hands with different labels: the one further left in
        the image is almost always the wearer's left hand in first-person video."""
        if len(dets) != 2 or dets[0].label == dets[1].label:
            return
        a, b = sorted(dets, key=lambda d: d.center[0])
        if b.center[0] - a.center[0] < 0.5 * max(a.size, b.size):
            return
        self.swap_votes += 1 if a.label == "right" else -1
        self.swap_votes = int(np.clip(self.swap_votes, -50, 50))

    # ---------- filtering ----------
    def _filter(self, dets):
        keep = []
        for d in sorted(dets, key=lambda d: -d.score):
            if d.size < self.cfg.min_hand_frac * self.diag:
                self.n_rejected_small += 1          # far away = probably someone else's hand
                continue
            if self.is_other_person(d):
                self.n_rejected_other += 1          # across the table: wrist high, fingers not up
                continue
            if any(_iou(d.bbox, k.bbox) > 0.5 for k in keep):
                continue                              # duplicate detection of the same hand
            keep.append(d)
        return keep

    def is_other_person(self, d):
        """Someone else's hand: wrist high in the frame with fingers not pointing up
        (across the table), or fingers pointing clearly TOWARD the camera from the upper
        part of the frame (a person sitting or standing opposite)."""
        wy, up = d.kp[0, 1] / self.H, finger_up(d.kp)
        return ((wy < self.cfg.other_wrist_y and up < self.cfg.other_finger_up) or
                (wy < self.cfg.other_down_y and up < self.cfg.other_down_up))

    def side_evidence(self, d):
        """Penalty-ready evidence for each side: MediaPipe label + forearm entry point."""
        lab = self.calibrated(d.label)
        e = forearm_entry(d.kp, self.W, self.H)
        conf_e = 0.0 if e is None else min(1.0, abs(e - 0.5) * 4)
        side_e = None if e is None else ("left" if e < 0.5 else "right")
        return lab, min(1.0, d.score), side_e, conf_e

    # ---------- costs ----------
    def _cost(self, slot, d, k):
        cfg = self.cfg
        lab, conf, side_e, conf_e = self.side_evidence(d)
        label_pen = cfg.label_weight * ((lab != slot.side) * conf * 0.6 +
                                        (side_e is not None and side_e != slot.side) * conf_e * 0.6)
        birth = self._birth_cost(slot, d, k, label_pen)
        if slot.active:
            lost = slot.lost(k)
            dist = np.linalg.norm(d.center - slot.predict(k)) / max(slot.size, 1.0)
            size_pen = 1.5 * abs(np.log(d.size / max(slot.size, 1.0)))
            track = dist + size_pen + label_pen
            if dist > cfg.gate_hand_sizes * (1 + 0.3 * lost):
                track = np.inf
            # a hand lost for a moment may come back somewhere else (fast head turn)
            return min(track, birth + 0.5) if lost >= 1 else track
        return birth

    def _birth_cost(self, slot, d, k, label_pen):
        """Cost of starting this slot's track on detection d: label + position prior."""
        nx = d.center[0] / self.W
        pos_pen = max(0.0, nx - 0.55) if slot.side == "left" else max(0.0, 0.45 - nx)
        other = self.slots[OTHER[slot.side]]
        if other.active and np.linalg.norm(d.center - other.predict(k)) < 0.6 * max(other.size, 1):
            return np.inf                             # that's the other hand, not a new one
        return 1.0 + label_pen + 2.0 * pos_pen

    def _assign(self, dets, k):
        best, best_cost = {}, 0.0
        options = [None] + list(range(len(dets)))
        for li in options:
            for ri in options:
                if li is not None and li == ri:
                    continue
                total, ok = 0.0, True
                for side, idx in (("left", li), ("right", ri)):
                    if idx is None:
                        continue
                    c = self._cost(self.slots[side], dets[idx], k)
                    if not np.isfinite(c):
                        ok = False
                        break
                    total += c - ASSIGN_REWARD
                if ok and total < best_cost:
                    best_cost = total
                    best = {s: dets[i] for s, i in (("left", li), ("right", ri)) if i is not None}
        return best

    # ---------- main update ----------
    def step(self, dets, k, recover=None):
        """dets: RawHands from the full frame. recover(side, box) -> extra RawHands
        found by zooming into where a missing hand should be."""
        dets = self._filter(dets)
        self._vote_calibration(dets)
        found = self._assign(dets, k)

        if recover is not None and self.cfg.roi_recovery:
            for side, slot in self.slots.items():
                if side in found or not slot.active or slot.lost(k) >= self.cfg.roi_max_lost:
                    continue
                c = slot.predict(k)
                half = slot.size * self.cfg.roi_scale / 2
                box = [c[0] - half, c[1] - half, c[0] + half, c[1] + half]
                other = self.slots[OTHER[side]]
                oc = other.predict(k) if other.active else None

                def plausible(d):
                    dist = np.linalg.norm(d.center - c)
                    if not (0.5 < d.size / slot.size < 2.0) or dist > slot.size:
                        return False
                    if self.is_other_person(d):
                        return False
                    if any(_iou(d.bbox, f.bbox) > 0.3 for f in found.values()):
                        return False          # that hand is already taken
                    # the crop may contain the other hand: it must be nearer to OUR prediction
                    return oc is None or dist < np.linalg.norm(d.center - oc)
                cands = [d for d in recover(side, box) if plausible(d)]
                if cands:
                    found[side] = min(cands, key=lambda d: np.linalg.norm(d.center - c))

        frame = {}
        for side, slot in self.slots.items():
            d = found.get(side)
            if d is None:
                if slot.active and slot.lost(k) + 1 > self.cfg.max_lost:
                    slot.reset()
                frame[side] = None
                continue
            c = d.center
            if slot.active and np.linalg.norm(c - slot.predict(k)) / max(slot.size, 1.0) > \
                    self.cfg.gate_hand_sizes * (1 + 0.3 * slot.lost(k)):
                slot.active = False                   # re-found far away: start a fresh track
            if slot.active:
                dt = max(1, k - slot.last_k)
                slot.vel = 0.5 * slot.vel + 0.5 * (c - slot.center) / dt
                slot.size = 0.7 * slot.size + 0.3 * d.size
            else:
                slot.vel, slot.size, slot.votes = np.zeros(2), d.size, deque()
                slot.track_id, self.next_id = self.next_id, self.next_id + 1
            slot.active, slot.center, slot.bbox, slot.last_k = True, c, d.bbox, k
            if d.src == "mp":
                slot.votes.append(1 if self.calibrated(d.label) == side else -1)
                while len(slot.votes) > self.cfg.swap_window:
                    slot.votes.popleft()
            frame[side] = Tracked(kp=d.kp.copy(), score=d.score, src=d.src, raw_label=d.label,
                                  track_id=slot.track_id)
        self.history.append(frame)
        self._repair_swaps(k)
        return frame

    # ---------- side-swap repair ----------
    def _disagrees(self, slot):
        v = slot.votes
        return len(v) >= self.cfg.swap_window and (np.array(v) < 0).mean() >= self.cfg.swap_ratio

    def _repair_swaps(self, k):
        L, R = self.slots["left"], self.slots["right"]
        both_wrong = L.active and R.active and self._disagrees(L) and self._disagrees(R)
        lone = None
        for s in (L, R):
            o = self.slots[OTHER[s.side]]
            if s.active and not o.active and self._disagrees(s):
                nx = s.center[0] / self.W
                # only flip a lone hand if its position also supports the label
                if (s.side == "left" and nx > 0.5) or (s.side == "right" and nx < 0.5):
                    lone = s
        if not (both_wrong or lone):
            return
        n = self.cfg.swap_window
        # relabel the evidence window retroactively, then swap slot identities
        for fr in self.history[-n:]:
            fr["left"], fr["right"] = fr["right"], fr["left"]
        self.slots = {"left": R, "right": L}
        L.side, R.side = "right", "left"
        for s in (L, R):
            s.votes = deque(-v for v in s.votes)
        self.repairs.append((k, "both" if both_wrong else "lone"))
