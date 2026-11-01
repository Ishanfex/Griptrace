"""Draw results onto the clip -> demo video for the organisers' Drive folder."""
import cv2
import numpy as np

from .tracker import SIDES

# BGR colours
SIDE_COL = {"left": (182, 196, 46), "right": (28, 159, 255)}      # teal / orange
STATE_COL = {"portable": (224, 157, 77), "fixed": (71, 197, 232), "none": (90, 90, 90),
             "not_visible": (60, 60, 60)}
HAZARD_COL = (60, 60, 230)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
         (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
         (17, 18), (18, 19), (19, 20)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _writer(path, fps, size):
    for cc in ("avc1", "mp4v"):                 # avc1 plays in browsers/Drive; mp4v always works
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError("Could not open a video writer")


def _chip(img, text, org, col, scale=0.45):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + 3), col, -1)
    cv2.putText(img, text, (x + 4, y - 2), FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)


def _timeline(n_frames, segs, safety, fps, width, height=34):
    """Pre-draw the two-lane grab/release timeline once."""
    bar = np.full((height, width, 3), 24, np.uint8)
    lane = (height - 6) // 2
    for s in segs:
        if s["state"] in ("portable", "fixed"):
            x0, x1 = int(s["t_start"] * fps / n_frames * width), int(s["t_end"] * fps / n_frames * width)
            y0 = 3 + (0 if s["hand"] == "left" else lane)
            cv2.rectangle(bar, (x0, y0), (max(x1, x0 + 1), y0 + lane - 2), STATE_COL[s["state"]], -1)
    for e in safety:
        x = int(e["t_start"] * fps / n_frames * width)
        cv2.line(bar, (x, 0), (x, height), HAZARD_COL, 2)
    cv2.putText(bar, "L", (2, 3 + lane - 4), FONT, 0.35, SIDE_COL["left"], 1, cv2.LINE_AA)
    cv2.putText(bar, "R", (2, 3 + 2 * lane - 4), FONT, 0.35, SIDE_COL["right"], 1, cv2.LINE_AA)
    return bar


def render(clip, out_path, per_frame, states, proc_idx, obj_frames, ghosts, obj_info, segs, safety,
           fps, scale, max_seconds=None, out_width=854):
    """Draw results on the clip, resized to `out_width` px wide so demo files stay small."""
    cap = cv2.VideoCapture(str(clip))
    W0, H0 = int(cap.get(3)), int(cap.get(4))
    r = min(1.0, out_width / W0) if out_width else 1.0
    W, H = int(round(W0 * r)), int(round(H0 * r))
    scale = scale * r                              # processing coords -> output coords
    n = len(per_frame)
    bar = _timeline(n, segs, safety, fps, W)
    out = _writer(out_path, fps, (W, H + bar.shape[0]))
    k = 0
    unsafe = set()
    for e in safety:
        unsafe.update(range(int(e["t_start"] * fps), int(e["t_end"] * fps) + 1))
    for i in range(n):
        ok, img = cap.read()
        if not ok or (max_seconds and i > max_seconds * fps):
            break
        if r != 1.0:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        while k + 1 < len(proc_idx) and proc_idx[k + 1] <= i:
            k += 1
        held = {states[i][s][1] for s in SIDES}
        # objects
        for tid, b in list(obj_frames[k].items()) + list(ghosts[k].items()):
            info = obj_info.get(str(tid)) or obj_info.get(tid) or {}
            b = (np.array(b) * scale).astype(int)
            col = HAZARD_COL if info.get("hazard") else STATE_COL.get(info.get("kind"), (200, 200, 200))
            th = 3 if tid in held else 1
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), col, th, cv2.LINE_AA)
            cv2.putText(img, f'{info.get("name", "?")}#{tid}', (b[0] + 3, b[1] + 14), FONT, 0.42, col, 1,
                        cv2.LINE_AA)
        # hands
        for side in SIDES:
            h = per_frame[i][side]
            if h is None:
                continue
            kp = (h.kp * scale).astype(int)
            col = SIDE_COL[side]
            dim = h.src in ("interp", "fill")
            for a, b in BONES:
                cv2.line(img, tuple(kp[a]), tuple(kp[b]), col, 1 if dim else 2, cv2.LINE_AA)
            for p in kp:
                cv2.circle(img, tuple(p), 2, (255, 255, 255), -1, cv2.LINE_AA)
            st, obj = states[i][side]
            label = f"{side[0].upper()}{h.track_id}"
            if st in ("portable", "fixed") and obj is not None:
                name = (obj_info.get(str(obj)) or obj_info.get(obj) or {}).get("name", "")
                label += f" | {'holds' if st == 'portable' else 'on'} {name}"
                b = obj_frames[k].get(obj, ghosts[k].get(obj))
                if b is not None:
                    c = (np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) * scale).astype(int)
                    tip = kp[[4, 8, 12]].mean(0).astype(int)
                    cv2.line(img, tuple(tip), tuple(c), STATE_COL[st], 1, cv2.LINE_AA)
            _chip(img, label, (kp[:, 0].min(), max(14, kp[:, 1].min() - 6)), col)
        if i in unsafe:
            cv2.rectangle(img, (0, 0), (W - 1, H - 1), HAZARD_COL, 6)
            _chip(img, "HAND NEAR MOVING HAZARD", (10, H - 12), HAZARD_COL, 0.55)
        _chip(img, f"t={i / fps:6.2f}s  frame {i}", (8, 20), (230, 230, 230), 0.45)
        b2 = bar.copy()
        x = int(i / max(n, 1) * W)
        cv2.line(b2, (x, 0), (x, b2.shape[0]), (255, 255, 255), 1)
        out.write(np.vstack([img, b2]))
    out.release()
    cap.release()
