"""Find the hackathon clips on this computer (prints folders, never copies anything)."""
import os
from pathlib import Path

home = Path.home()
roots = [home / d for d in ("Downloads", "Desktop", "Documents", "OneDrive", "Videos")] + [Path("D:/"), Path("E:/")]
skip = {"AppData", ".venv", "node_modules", ".git", "$Recycle.Bin", "Windows", "Program Files"}
found = {}
for root in roots:
    if not root.exists():
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        if dirpath.count(os.sep) - str(root).count(os.sep) > 5:
            dirnames[:] = []
        vids = [f for f in filenames if f.lower().endswith((".mp4", ".mov", ".mkv"))]
        if vids and ("clip" in dirpath.lower() or "egohack" in dirpath.lower() or any("factory" in v.lower() for v in vids)):
            found[dirpath] = sorted(vids)
for d, v in found.items():
    print(f"CLIPS_DIR {d}  ({len(v)} videos: {', '.join(v[:5])}{' ...' if len(v) > 5 else ''})")
if not found:
    print("CLIPS_DIR none found - put the egohack-data folder in Downloads or Documents")
