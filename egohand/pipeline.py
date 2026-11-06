"""The whole PS2 pipeline for one clip.

Pass 1 (online, per processed frame):
    head-motion -> dark-frame fix -> MediaPipe -> tracker (+ zoomed-crop recovery)
    -> object detector every N frames
Pass 2 (offline): side fixing, gap filling, zero-lag smoothing, contacts, timeline,
    tool usage, safety, metrics
Pass 3: write files and the demo video
"""
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

from . import contact as C
from . import postprocess as P
from .camera import CameraMotion
from .enhance import enhance, mean_luma
from .export import viewer_json, write_contacts_csv, write_hands_jsonl
from .metrics import compute
from .objects import ObjectDetector, ObjectTracker
from .render import render
from .tracker import SIDES, HandTracker
from .video import frames, video_info


class Timer:
    def __init__(self):
        self.t = defaultdict(float)

    def __call__(self, key):
        timer = self

        class _T:
            def __enter__(s):
                s.t0 = time.perf_counter()

            def __exit__(s, *a):
                timer.t[key] += time.perf_counter() - s.t0
        return _T()


def run(clip, out_dir, cfg, hand_detector=None, object_detector=None, log=print, max_frames=None):
    clip, out_dir = Path(clip), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = video_info(clip)
    fps, n_video = info["fps"], info["n_frames"]
    W, H = (cfg.width, int(round(info["height"] * cfg.width / info["width"]))) if cfg.width \
        else (info["width"], info["height"])
    scale = info["width"] / W                       # processing pixels -> original pixels

    if hand_detector is None:
        from .handdet import MediaPipeHands
        hand_detector = MediaPipeHands(cfg)
    objdet = object_detector or ObjectDetector(cfg)
    tracker, cam, otrack = HandTracker(cfg, W, H), CameraMotion(W, H), ObjectTracker(cfg)
    T = Timer()
    raw, times, proc_idx, hazard_obs = [], [], [], defaultdict(dict)
    n_dark = 0
    t_start = time.perf_counter()

    # ------------------------------------------------------------------ pass 1
    for k, (i, t, img) in enumerate(frames(clip, cfg.step, cfg.width)):
        if max_frames and k >= max_frames:
            break
        proc_idx.append(i)
        times.append(t)
        with T("camera"):
            prev = tracker.history[-1] if tracker.history else {}
            cam.update(img, [h.kp.min(0).tolist() + h.kp.max(0).tolist() for h in prev.values() if h])
        with T("enhance"):
            dark = mean_luma(img) < cfg.dark_luma
            det_img = enhance(img) if dark else img
            n_dark += dark
        with T("hands"):
            dets = hand_detector.detect(det_img, t * 1000)
        raw.append(dets)

        def recover(side, box):
            with T("recovery"):
                found = hand_detector.detect_crop(det_img, box, cfg.roi_size)
                if not found and not dark:
                    found = hand_detector.detect_crop(enhance(img), box, cfg.roi_size)
                    for f in found:
                        f.src = "enh"
                return found

        with T("tracker"):
            tracker.step(dets, k, recover)
        if objdet.backend != "none" and k % cfg.object_every == 0:
            with T("objects"):
                cur = tracker.history[-1]
                hand_boxes = [h.kp.min(0).tolist() + h.kp.max(0).tolist() for h in cur.values() if h is not None]
                otrack.update(objdet.detect(img, hand_boxes), k)
                for tid, tr in otrack.tracks.items():
                    if tr["last_k"] == k and otrack.name(tid) in cfg.vocab_hazard:
                        hazard_obs[tid][k] = cam.motion_in_box(tr["obs"][k])
        if k % 100 == 0:
            log(f"  frame {i}/{n_video}  ({time.perf_counter() - t_start:.0f}s)")
    T.t["pass1"] = time.perf_counter() - t_start

    # ------------------------------------------------------------------ pass 2
    hist = tracker.history
    n_proc = len(hist)
    dt = cfg.step / fps
    with T("post"):
        n_side_fix = P.fix_sides(hist, tracker.calibrated, W, H)
        P.fill_gaps(hist, cfg.max_gap, cfg.gap_max_jump)
        n_dupes = P.drop_duplicates(hist, W, H)
        P.smooth(hist, times, cfg, (W ** 2 + H ** 2) ** 0.5)

        obj_frames = otrack.boxes_per_frame(n_proc, cfg.obj_ttl)
        names = {tid: otrack.name(tid) for tid in otrack.tracks}
        kinds = {tid: objdet.kind(n) if objdet.backend != "none" else "portable" for tid, n in names.items()}
        hazard_ids = [tid for tid, n in names.items() if n in cfg.vocab_hazard]
        hazard_motion = [dict() for _ in range(n_proc)]
        for tid, obs in hazard_obs.items():
            ks = sorted(obs)
            for a_i, a in enumerate(ks):
                end = ks[a_i + 1] if a_i + 1 < len(ks) else min(n_proc, a + cfg.obj_ttl + 1)
                for kk in range(a, min(end, n_proc)):
                    hazard_motion[kk][tid] = obs[a]

        cand, ghosts = C.frame_candidates(hist, obj_frames, kinds, times, cfg, W, H, cam.to_scene)
        states = C.hysteresis(cand, cfg)
        segs = C.segments(states, times, dt, names, cfg)
        segs = C.refine_kinds(segs, obj_frames, cam.to_scene, cfg)
        if cfg.clip_namer and objdet.backend != "none":
            with T("namer"):
                from .namer import name_segments
                n_named = name_segments(clip, segs, hist, proc_idx, scale, cfg, log=log)
            segs = C.merge_segments(segs)
        states = C.apply_segments(states, segs)
        tools = C.tool_usage(segs)
        safety = C.safety_events(hist, states, obj_frames, hazard_ids, hazard_motion, names, times, dt, cfg)

        # per video frame
        n_out = (min(n_video, proc_idx[-1] + 1) if max_frames else max(n_video, proc_idx[-1] + 1)) \
            if proc_idx else 0
        per_frame = P.expand(hist, proc_idx, n_out)
        vstates, k = [], 0
        for i in range(n_out):
            while k + 1 < len(proc_idx) and proc_idx[k + 1] <= i:
                k += 1
            vstates.append({s: (states[k][s] if per_frame[i][s] is not None else ("not_visible", None))
                            for s in SIDES})
    T.t["total"] = time.perf_counter() - t_start
    if hasattr(hand_detector, "close"):
        hand_detector.close()

    used = {s["obj"] for s in segs if s["obj"] is not None} | set().union(*[set(f) for f in obj_frames]) \
        if obj_frames else set()
    obj_info = {str(t): {"name": names[t], "kind": kinds[t], "hazard": t in hazard_ids} for t in used}
    for s in safety:
        s.pop("k0", None), s.pop("k1", None)
    extra = {"dark_frames": n_dark, "side_repairs_online": len(tracker.repairs),
             "side_fixes_offline": n_side_fix, "duplicates_removed": n_dupes, "rejected_small_hands": tracker.n_rejected_small,
             "rejected_coworker_hands": tracker.n_rejected_other,
             "camera_motion_ok_pct": round(100 * cam.ok_frames / max(1, n_proc - 1), 1),
             "grabs": sum(s["state"] == "portable" for s in segs),
             "fixed_contacts": sum(s["state"] == "fixed" for s in segs),
             "safety_events": len(safety), "objects_tracked": len(names)}
    metrics = compute(hist, raw, tracker.swap, dict(T.t), n_out, extra)
    meta = {"clip": clip.name, "fps": fps, "width": info["width"], "height": info["height"],
            "n_frames": n_out, "step": cfg.step, "process_width": W,
            "objects_backend": objdet.backend, "hardware": f"{platform.processor() or platform.machine()} "
            f"({platform.system()}), CPU only", "created": time.strftime("%Y-%m-%d %H:%M")}

    # ------------------------------------------------------------------ pass 3
    stem = clip.stem
    write_hands_jsonl(out_dir / "hands.jsonl", per_frame, vstates, fps, scale)
    write_contacts_csv(out_dir / "contacts.csv", stem, segs)
    viewer_json(out_dir / f"{stem}.results.json", meta, per_frame, vstates, proc_idx, raw, obj_frames,
                ghosts, obj_info, segs, tools, safety, metrics, scale)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if cfg.render:
        log("  rendering demo video ...")
        render(clip, out_dir / f"{stem}_demo.mp4", per_frame, vstates, proc_idx, obj_frames, ghosts,
               obj_info, segs, safety, fps, scale)
    return {"metrics": metrics, "segments": segs, "tools": tools, "safety": safety,
            "per_frame": per_frame, "states": vstates, "hist": hist, "raw": raw}
