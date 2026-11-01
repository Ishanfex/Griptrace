"""Write outputs in the organisers' suggested formats + one compact file for the viewer."""
import csv
import json

import numpy as np

from .tracker import SIDES

SRC = {"mp": 0, "roi": 1, "enh": 2, "interp": 3, "fill": 4}
STATE = {"none": 0, "portable": 1, "fixed": 2, "not_visible": 3}


def _r(x, nd=1):
    return [round(float(v), nd) for v in x]


def write_hands_jsonl(path, per_frame, states, fps, scale):
    with open(path, "w") as f:
        for i, fr in enumerate(per_frame):
            hands = []
            for side in SIDES:
                h = fr[side]
                if h is None:
                    continue
                kp = h.kp * scale
                st, obj = states[i][side]
                hands.append({"side": side, "bbox": _r([*kp.min(0), *kp.max(0)]),
                              "score": round(float(h.score), 3), "kp2d": [_r(p) for p in kp],
                              "track_id": int(h.track_id), "source": h.src,
                              "contact": st, "object_id": obj})
            f.write(json.dumps({"frame": i, "t": round(i / fps, 3), "hands": hands}) + "\n")


def write_contacts_csv(path, clip, segs):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "t_start", "t_end", "hand", "state", "object_name"])
        for s in sorted(segs, key=lambda s: (s["t_start"], s["hand"])):
            w.writerow([clip, f'{s["t_start"]:.2f}', f'{s["t_end"]:.2f}', s["hand"], s["state"],
                        s["object_name"]])


def viewer_json(path, meta, per_frame, states, proc_idx, raw, obj_frames, ghosts, obj_info,
                segs, tools, safety, metrics, scale):
    frames = []
    for i, fr in enumerate(per_frame):
        row = []
        for side in SIDES:
            h = fr[side]
            if h is None:
                row.append(None)
                continue
            st, obj = states[i][side]
            row.append([SRC[h.src], STATE[st], -1 if obj is None else int(obj)] +
                       [int(round(v)) for v in (h.kp * scale).ravel()])
        frames.append(row)
    raw_out = [[[r.label[0]] + [int(round(v)) for v in (r.kp * scale).ravel()] for r in fr] for fr in raw]
    objs = []
    for k in range(len(obj_frames)):
        row = [[int(t)] + [int(round(v * scale)) for v in b] + [0] for t, b in obj_frames[k].items()]
        row += [[int(t)] + [int(round(v * scale)) for v in b] + [1] for t, b in ghosts[k].items()]
        objs.append(row)
    clean = lambda s: {k: v for k, v in s.items() if k not in ("key",)}
    data = {"meta": meta, "proc_idx": proc_idx, "frames": frames, "raw": raw_out,
            "objects": objs, "object_info": obj_info,
            "segments": [clean(s) for s in segs], "tools": tools,
            "safety": safety, "metrics": metrics,
            "codes": {"src": list(SRC), "state": list(STATE)}}
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"), default=_np)


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
