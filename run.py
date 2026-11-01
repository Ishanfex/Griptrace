"""One command to run the whole pipeline.

    python run.py egohack-data/clips_480p/factory_001.mp4
    python run.py egohack-data/clips_480p/ --all          # every clip in a folder
    python run.py clip.mp4 --objects none                 # hand tracking only (PS1)
"""
import argparse
import json
import sys
from pathlib import Path

from egohand.config import Config
from egohand.pipeline import run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("input", help="a video file, or a folder with --all")
    ap.add_argument("--all", action="store_true", help="process every .mp4 in the folder")
    ap.add_argument("--out", default="outputs", help="output folder (git-ignored)")
    ap.add_argument("--step", type=int, default=2, help="process every Nth frame (default 2)")
    ap.add_argument("--width", type=int, default=-1,
                    help="resize before processing (-1 = auto: 854 px for clips wider than 960; 0 = keep)")
    ap.add_argument("--objects", choices=["world", "coco", "none"], default="world")
    ap.add_argument("--object-every", type=int, default=3)
    ap.add_argument("--swap", default="auto", help="handedness swap: auto | true | false")
    ap.add_argument("--no-render", action="store_true", help="skip the demo video")
    ap.add_argument("--max-frames", type=int, default=0, help="stop early (quick tests)")
    ap.add_argument("--skip-done", action="store_true", help="skip clips whose results already exist")
    ap.add_argument("--render-only", default="", help="comma list of clip names to render; others get no demo video")
    ap.add_argument("--no-agent", action="store_true",
                    help="skip the vision agent that names held objects (it runs only when an API key is set)")
    a = ap.parse_args()

    cfg = Config(step=a.step, width=max(a.width, 0), objects=a.objects, object_every=a.object_every,
                 swap_handedness=a.swap, render=not a.no_render)
    src = Path(a.input)
    clips = sorted(src.glob("*.mp4")) if a.all else [src]
    if not clips:
        sys.exit(f"No clips found in {src}")
    show = {s.strip() for s in a.render_only.split(",") if s.strip()}
    for clip in clips:
        if a.skip_done and (Path(a.out) / clip.stem / f"{clip.stem}.results.json").exists():
            print(f"[{clip.name}] already done, skipping")
            continue
        print(f"[{clip.name}]", flush=True)
        if show:
            cfg.render = (not a.no_render) and clip.stem in show
        if a.width == -1:                       # full-HD clips are slow on a laptop CPU
            from egohand.video import video_info
            cfg.width = 854 if video_info(clip)["width"] > 960 else 0
        res = run(clip, Path(a.out) / clip.stem, cfg, max_frames=a.max_frames or None)
        m = res["metrics"]
        if not a.no_agent:
            from egohand import agent
            if agent.available():
                rp = Path(a.out) / clip.stem / f"{clip.stem}.results.json"
                agent.run_agent(rp, clip)
                m = json.loads(rp.read_text())["metrics"]
            else:
                print("  agent: skipped (no API key: put it in api_key.txt to name objects with AI)")
        print(json.dumps({k: m[k] for k in (
            "left_share_baseline", "left_share_final", "right_share_baseline", "right_share_final",
            "jitter_baseline", "jitter_final", "baseline_label_flips", "grabs", "fps_video_equiv")
            if k in m}, indent=1))
        print(f"  -> {Path(a.out) / clip.stem}/  (open viewer/index.html and drop the "
              f"{clip.stem}.results.json + the clip)")


if __name__ == "__main__":
    main()
