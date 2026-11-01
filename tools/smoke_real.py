"""Real-model smoke test: first 30 s of up to 3 clips, with real MediaPipe + YOLO-World.
Prints versions, speed and the key numbers so we can check everything on this laptop."""
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egohand.config import Config  # noqa: E402
from egohand.pipeline import run  # noqa: E402


def pick_clips(n=3):
    """Up to n clips, one per recording series (prefix before the last '-')."""
    from tools.find_clips import found
    out, seen = [], set()
    for d in sorted(found, key=lambda d: ("480" not in d, d)):       # prefer clips_480p
        for v in found[d]:
            series = v.rsplit("-", 1)[0]
            if series not in seen and len(out) < n:
                seen.add(series)
                out.append(Path(d) / v)
    return out


def main():
    import cv2
    import mediapipe
    print("python", platform.python_version(), "| mediapipe", mediapipe.__version__, "| opencv", cv2.__version__)
    try:
        import ultralytics
        print("ultralytics", ultralytics.__version__)
    except ImportError:
        print("ultralytics NOT installed")
    clips = [Path(a) for a in sys.argv[1:]] or pick_clips()
    if not clips:
        print("SMOKE no clip found")
        return
    for clip in clips:
        smoke(clip)


def smoke(clip):
    from egohand.video import video_info
    info = video_info(clip)
    width = 854 if info["width"] > 960 else 0          # full-res clips: shrink for laptop speed
    print("SMOKE clip:", clip.name, info, "-> process width", width or info["width"])
    for backend in ("world", "coco", "none"):
        cfg = Config(objects=backend, width=width)
        if backend == "world" and not Path(cfg.object_model).exists():
            print("  YOLO-World model missing (prep.py failed) -> trying coco")
            continue
        try:
            t0 = time.time()
            res = run(clip, Path("outputs") / clip.stem, cfg, max_frames=450)
            m = res["metrics"]
            keep = {k: v for k, v in m.items() if k != "ms_per_frame"}
            print("SMOKE backend:", backend, "| wall", round(time.time() - t0, 1), "s")
            print("SMOKE metrics:", json.dumps(keep))
            print("SMOKE ms_per_frame:", json.dumps(m["ms_per_frame"]))
            print("SMOKE segments:", json.dumps([{k: s[k] for k in ("hand", "state", "object_name", "t_start", "t_end")}
                                                 for s in res["segments"] if s["state"] in ("portable", "fixed")]))
            print("SMOKE OK")
            return
        except Exception as e:                                    # show the error, try the next backend
            import traceback
            traceback.print_exc()
            print("SMOKE backend failed:", backend, repr(e))


if __name__ == "__main__":
    main()
