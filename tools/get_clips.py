"""Move the hackathon clips downloaded from Drive (Downloads folder) into
egohack-data/clips/ inside this project (that folder is git-ignored).

Only COMPLETE downloads are moved: the browser writes into the final .mp4 name while
downloading, so a file must be unchanged for 2 minutes AND open as a full video."""
import shutil
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
dst = ROOT / "egohack-data" / "clips"
dst.mkdir(parents=True, exist_ok=True)
src = Path.home() / "Downloads"


def complete(f):
    if time.time() - f.stat().st_mtime < 120:
        return False
    cap = cv2.VideoCapture(str(f))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) if cap.isOpened() else 0
    cap.release()
    return n > 100


moved = waiting = 0
for f in sorted(src.glob("*.mp4")):
    if not f.name.startswith(("MTL-", "mytronlabs-")):
        continue
    target = dst / f.name
    if target.exists():
        continue
    if not complete(f):
        waiting += 1
        print("still downloading (left in Downloads):", f.name)
        continue
    shutil.move(str(f), str(target))
    moved += 1
    print("moved:", f.name)
print(f"GET_CLIPS moved {moved}, still downloading {waiting}, clips folder: {dst}")
