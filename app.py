"""GripTrace local app: the viewer + the pipeline behind one local web server.

    python app.py              # opens http://127.0.0.1:8765 in your browser
    python app.py --port 9000 --no-browser

What it does
  * serves viewer/index.html, which then loads every result and video by itself
  * lists clips in egohack-data/clips (videos) and outputs/ (results)
  * runs run.py on a clip when you press "Analyse" in the browser (one at a time, with progress)
  * accepts new clips you add from the browser (saved to egohack-data/clips, which is git-ignored)

It listens on 127.0.0.1 only, so nothing is reachable from other computers and no footage leaves this one.
Standard library only, so it needs nothing beyond requirements.txt.
"""
import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
CLIPS = ROOT / "egohack-data" / "clips"
PREVIEWS = ROOT / "egohack-data" / "browser"     # H.264 copies the browser can play (the clips are H.265)
OUT = ROOT / "outputs"
VIEWER = ROOT / "viewer"
LOGS = OUT / "_app_logs"
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".m4v")
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,120}$")
# extra run.py arguments, e.g. GRIPTRACE_RUN_ARGS="--objects none" for a hands-only machine
RUN_ARGS = os.environ.get("GRIPTRACE_RUN_ARGS", "").split()
AUTO = True          # analyse new clips and AI-name older results without being asked (--no-auto turns it off)


def key_available():
    from egohand import agent
    return agent.available()


_ai_cache = {}


def ai_status(stem):
    """'done' / 'failed' / None, read cheaply from the end of results.json (the agent block is written last)."""
    p = results_path(stem)
    if not p.exists():
        return None
    key = (stem, p.stat().st_mtime)
    if key not in _ai_cache:
        with open(p, "rb") as f:
            f.seek(max(0, p.stat().st_size - 20000))
            m = re.search(rb'"agent":\{"status":"(\w+)"', f.read())
        _ai_cache[key] = m.group(1).decode() if m else None
    return _ai_cache[key]


def safe_name(name):
    """A plain file name for any video name: 'MTL-SKL00089-011 (1).mp4' -> 'MTL-SKL00089-011.mp4'.
    Windows adds ' (1)' to repeated downloads; brackets, accents and symbols become '-'."""
    p = Path(name)
    stem = re.sub(r"\s*\(\d+\)$", "", p.stem)                 # ' (1)' from a repeated download
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "-", stem).strip(" .-_") or "clip"
    return stem[:100] + p.suffix.lower()


def tidy_clip_names():
    """Rename videos copied into the clips folder whose names the app can't serve (keeps them otherwise)."""
    if not CLIPS.exists():
        return
    for p in list(CLIPS.iterdir()):
        if p.suffix.lower() not in VIDEO_EXT or SAFE.match(p.name):
            continue
        new = CLIPS / safe_name(p.name)
        n = 2
        while new.exists():
            new = CLIPS / f"{Path(safe_name(p.name)).stem}-{n}{p.suffix.lower()}"
            n += 1
        try:
            os.replace(p, new)
            print(f"  renamed '{p.name}' -> '{new.name}'", flush=True)
        except OSError:
            pass                                      # still being copied: try again on the next pass


def results_path(stem):
    return OUT / stem / f"{stem}.results.json"


def video_path(stem):
    for ext in VIDEO_EXT:
        p = CLIPS / f"{stem}{ext}"
        if p.exists():
            return p
    return None


_check_cache = {}


def video_check(path):
    """'ok', 'copying' (still being written) or 'broken' (unfinished download / damaged file)."""
    st = path.stat()
    if time.time() - st.st_mtime < 8 and st.st_size != _check_cache.get(("size", path)):
        _check_cache[("size", path)] = st.st_size
        return "copying"
    key = (str(path), st.st_size, int(st.st_mtime))
    if key not in _check_cache:
        import cv2
        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 10 and cap.read()[0]
        cap.release()
        _check_cache[key] = "ok" if ok else "broken"
    return _check_cache[key]


def file_key(path):
    st = path.stat()
    return (path.name, st.st_size, int(st.st_mtime))


def preview_path(stem):
    p = PREVIEWS / f"{stem}.mp4"
    v = video_path(stem)
    return p if p.exists() and v and p.stat().st_mtime >= v.stat().st_mtime - 1 else None


def browser_video(stem):
    """What the browser gets: the H.264 copy once it exists, else the original file."""
    return preview_path(stem) or video_path(stem)


def make_preview(src, dst, width=960):
    """Re-encode to H.264 so Chrome/Edge can play it. The organisers' clips are H.265 (HEVC), which browsers on
    most Windows PCs can't decode: the video stays black. Uses ffmpeg if installed, else OpenCV's H.264 writer."""
    tmp = dst.with_name(dst.stem + ".part.mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg")
    if ff:
        r = subprocess.run([ff, "-y", "-v", "error", "-i", str(src), "-vf", f"scale={width}:-2", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
                            str(tmp)], capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0 and tmp.exists():
            os.replace(tmp, dst)
            return True
    import cv2                                   # same OpenCV the pipeline uses
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not w0:
        return False
    w = min(width, w0)
    h = int(round(h0 * w / w0 / 2)) * 2
    out = None
    for cc in ("avc1", "H264", "h264"):
        out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*cc), fps, (w, h))
        if out.isOpened():
            break
        out.release()
        out = None
    if out is None:
        cap.release()
        return False
    while True:
        ok, img = cap.read()
        if not ok:
            break
        out.write(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if (w, h) != (w0, h0) else img)
    cap.release()
    out.release()
    if tmp.exists() and tmp.stat().st_size > 1000:
        os.replace(tmp, dst)
        return True
    return False


class Previews:
    """Makes the browser copies in the background, one clip at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.busy = None
        self.failed = set()
        threading.Thread(target=self._worker, daemon=True).start()

    def state(self, stem):
        if preview_path(stem):
            return "ready"
        v = video_path(stem)
        with self.lock:
            return "making" if self.busy == stem else "failed" if v and file_key(v) in self.failed else "waiting"

    def _worker(self):
        while True:
            todo = None
            if CLIPS.exists():
                for p in sorted(CLIPS.iterdir()):
                    if p.suffix.lower() in VIDEO_EXT and file_key(p) not in self.failed \
                            and not preview_path(p.stem) and video_check(p) == "ok":
                        todo = p
                        break
            if not todo:
                time.sleep(3)
                continue
            with self.lock:
                self.busy = todo.stem
            try:
                ok = make_preview(todo, PREVIEWS / f"{todo.stem}.mp4")
            except Exception as e:               # keep the server alive
                print(f"  preview failed for {todo.name}: {e}", flush=True)
                ok = False
            with self.lock:
                self.busy = None
                if not ok:
                    self.failed.add(file_key(todo))


def median(a):
    a = sorted(a)
    if not a:
        return 0
    m = len(a) // 2
    return a[m] if len(a) % 2 else (a[m - 1] + a[m]) / 2


_summ_cache = {}


def summary(stem):
    """A small version of results.json for the clip list (the full file is 1-4 MB)."""
    p = results_path(stem)
    key = (stem, p.stat().st_mtime)
    if key not in _summ_cache:
        d = json.loads(p.read_text(encoding="utf-8"))
        segs = [{"hand": s["hand"], "state": s["state"], "t0": s["t_start"], "t1": s["t_end"], "name": s["object_name"]}
                for s in d["segments"]]
        contacts = [s["t_end"] - s["t_start"] for s in d["segments"] if s["state"] in ("portable", "fixed")]
        _summ_cache[key] = {"meta": d["meta"], "metrics": d["metrics"], "T": d["meta"]["n_frames"] / d["meta"]["fps"],
                            "segs": segs, "medHold": median(contacts),
                            "objects": (d["meta"].get("objects_backend") or "none") != "none"}
    return _summ_cache[key]


def list_clips():
    tidy_clip_names()
    stems = set()
    if CLIPS.exists():
        stems |= {p.stem for p in CLIPS.iterdir() if p.suffix.lower() in VIDEO_EXT and not p.name.startswith(".")}
    if OUT.exists():
        stems |= {p.name for p in OUT.iterdir() if p.is_dir() and results_path(p.name).exists()}
    rows = []
    for s in sorted(stems):
        v = video_path(s)
        b = browser_video(s)
        r = results_path(s)
        rows.append({"stem": s, "results": r.exists(), "rmtime": r.stat().st_mtime if r.exists() else None,
                     "video": v.name if v else None,
                     "video_mb": round(v.stat().st_size / 1e6, 1) if v else None, "job": JOBS.state(s),
                     "check": video_check(v) if v and not r.exists() else ("ok" if v else None),
                     "preview": PREVIEWS_.state(s) if v else None, "ai": ai_status(s), "vver": int(b.stat().st_mtime) if b else None})
    return rows


# ------------------------------------------------------------------ analysis jobs
class Jobs:
    """Runs run.py on one clip at a time; the rest wait in a queue."""

    def __init__(self):
        self.lock = threading.Lock()
        self.queue = []          # stems waiting
        self.cur = None          # {"stem", "frame", "total", "phase", "started", "proc"}
        self.done = {}           # stem -> {"ok", "msg", "finished"}
        self.kind = {}           # stem -> "analyse" (run.py, includes the agent) or "name" (agent only)
        threading.Thread(target=self._worker, daemon=True).start()

    def add(self, stem, kind="analyse"):
        with self.lock:
            if stem in self.queue or (self.cur and self.cur["stem"] == stem):
                return
            self.queue.append(stem)
            self.kind[stem] = kind
            self.done.pop(stem, None)

    def cancel(self, stem):
        with self.lock:
            if stem in self.queue:
                self.queue.remove(stem)
            if self.cur and self.cur["stem"] == stem and self.cur.get("proc"):
                self.cur["proc"].terminate()

    def state(self, stem):
        with self.lock:
            if self.cur and self.cur["stem"] == stem:
                c = self.cur
                return {"status": "running", "frame": c["frame"], "total": c["total"], "phase": c["phase"],
                        "elapsed": round(time.time() - c["started"]), "kind": self.kind.get(stem, "analyse")}
            if stem in self.queue:
                return {"status": "queued", "position": self.queue.index(stem) + 1, "kind": self.kind.get(stem, "analyse")}
            if stem in self.done:
                return {"status": "done" if self.done[stem]["ok"] else "failed", "kind": self.kind.get(stem, "analyse"),
                        **self.done[stem]}
        return None

    def snapshot(self):
        with self.lock:
            stems = ([self.cur["stem"]] if self.cur else []) + list(self.queue)
        return {s: self.state(s) for s in stems}

    def _worker(self):
        while True:
            with self.lock:
                stem = self.queue.pop(0) if self.queue else None
                if stem:
                    self.cur = {"stem": stem, "frame": 0, "total": 0, "phase": "starting", "started": time.time()}
            if not stem:
                time.sleep(0.5)
                continue
            ok, msg = self._run(stem)
            with self.lock:
                self.done[stem] = {"ok": ok, "msg": msg, "finished": time.time()}
                self.cur = None

    def _run(self, stem):
        v = video_path(stem)
        if not v:
            return False, "video not found"
        LOGS.mkdir(parents=True, exist_ok=True)
        if self.kind.get(stem) == "name":
            cmd = [sys.executable, "-u", "-m", "egohand.agent", str(results_path(stem)), str(v)]
        else:
            cmd = [sys.executable, "-u", str(ROOT / "run.py"), str(v), "--no-render", *RUN_ARGS]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        log = open(LOGS / f"{stem}.log", "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", env=env,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            log.close()
            return False, str(e)
        with self.lock:
            self.cur["proc"] = proc
            self.cur["phase"] = "loading models"
        tail = []
        for line in proc.stdout:
            log.write(line)
            tail = (tail + [line.strip()])[-8:]
            m = re.search(r"frame (\d+)/(\d+)", line)
            ma = re.search(r"agent: named (\d+)/(\d+)", line)
            with self.lock:
                if "agent:" in line:
                    self.cur["phase"] = "AI naming objects"
                    if ma:
                        self.cur["phase"] = f"AI naming objects {ma.group(1)}/{ma.group(2)}"
                    elif "surveying" in line:
                        self.cur["phase"] = "AI surveying the workstation"
                elif m:
                    self.cur["frame"], self.cur["total"] = int(m.group(1)), int(m.group(2))
                    self.cur["phase"] = "tracking hands and objects"
                elif "rendering" in line:
                    self.cur["phase"] = "rendering"
                elif self.cur["total"] and self.cur["frame"] >= self.cur["total"] - 400 and line.strip():
                    self.cur["phase"] = "naming objects and finishing"
        rc = proc.wait()
        log.close()
        if rc == 0 and results_path(stem).exists():
            agent_line = next((t for t in reversed(tail) if "agent:" in t), "")
            return True, agent_line.replace("agent:", "AI:").strip() or "finished"
        err = next((t for t in reversed(tail) if "Error" in t or "error" in t), tail[-1] if tail else "")
        return False, f"run.py stopped (code {rc}). {err}"[:300]


JOBS = Jobs()
PREVIEWS_ = Previews()


def auto_queue():
    """Self-running mode: new videos get analysed, and results without AI names get named (once each)."""
    tried = set()
    while True:
        time.sleep(4)
        if not AUTO:
            continue
        try:
            has_key = key_available()
            for row in list_clips():
                s, job = row["stem"], row["job"]
                if job and job.get("status") in ("running", "queued"):
                    continue
                if not row["video"]:
                    continue
                k = file_key(video_path(s))
                if not row["results"] and row["check"] == "ok" and (k, "analyse") not in tried:
                    tried.add((k, "analyse"))
                    JOBS.add(s, "analyse")
                elif row["results"] and has_key and row["ai"] is None and (k, "name") not in tried:
                    tried.add((k, "name"))
                    JOBS.add(s, "name")
        except Exception as e:                       # never stop the server over this
            print(f"  auto-queue: {e}", flush=True)


# ------------------------------------------------------------------ HTTP
class Handler(SimpleHTTPRequestHandler):
    server_version = "GripTrace"

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(VIEWER), **k)

    def log_message(self, fmt, *args):            # keep the console quiet except for errors
        if args and str(args[1])[:1] in "45":
            sys.stderr.write("  %s %s\n" % (self.command, fmt % args))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stem(self, raw):
        s = unquote(raw)
        return s if SAFE.match(s) and ".." not in s else None

    def do_GET(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        try:
            if u.path == "/api/ping":
                return self._json({"app": "griptrace", "clips_dir": str(CLIPS), "outputs_dir": str(OUT),
                                   "ai_key": key_available(), "auto": AUTO, "key_file": str(ROOT / "api_key.txt")})
            if u.path == "/api/clips":
                return self._json(list_clips())
            if u.path == "/api/jobs":
                return self._json(JOBS.snapshot())
            if len(parts) == 3 and parts[:2] == ["api", "summary"]:
                s = self._stem(parts[2])
                return self._json(summary(s)) if s and results_path(s).exists() else self._json({"error": "no results"}, 404)
            if len(parts) == 3 and parts[:2] == ["api", "results"]:
                s = self._stem(parts[2])
                if not s or not results_path(s).exists():
                    return self._json({"error": "no results"}, 404)
                return self._file(results_path(s), "application/json")
            if len(parts) == 2 and parts[0] == "video":
                s = self._stem(parts[1])
                v = browser_video(s) if s else None
                return self._file(v, mimetypes.guess_type(v.name)[0] or "video/mp4") if v else self._json({"error": "no video"}, 404)
        except (ConnectionError, BrokenPipeError):
            return
        except Exception as e:                       # report, don't crash the server
            return self._json({"error": f"{e.__class__.__name__}: {e}"}, 500)
        if u.path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        return super().do_GET()                      # the viewer itself

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/api/analyse":
            s = self._stem(q.get("clip", [""])[0])
            if not s or not video_path(s):
                return self._json({"error": "no such clip video"}, 404)
            if video_check(video_path(s)) == "broken":
                return self._json({"error": "this video file is incomplete or damaged - download it again"}, 400)
            JOBS.add(s)
            return self._json({"ok": True, "job": JOBS.state(s)})
        if u.path == "/api/name":
            s = self._stem(q.get("clip", [""])[0])
            if not s or not video_path(s) or not results_path(s).exists():
                return self._json({"error": "analyse the clip first"}, 404)
            if not key_available():
                return self._json({"error": "no API key: put it in api_key.txt"}, 400)
            JOBS.add(s, "name")
            return self._json({"ok": True})
        if u.path == "/api/cancel":
            s = self._stem(q.get("clip", [""])[0])
            if s:
                JOBS.cancel(s)
            return self._json({"ok": True})
        if u.path == "/api/upload":
            name = safe_name(Path(unquote(q.get("name", [""])[0])).name)
            if Path(name).suffix.lower() not in VIDEO_EXT:
                return self._json({"error": "that is not a video file (use .mp4, .mov, .webm or .mkv)"}, 400)
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0:
                return self._json({"error": "empty upload"}, 400)
            CLIPS.mkdir(parents=True, exist_ok=True)
            dst, tmp = CLIPS / name, CLIPS / (name + ".part")
            with open(tmp, "wb") as f:
                left = n
                while left:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            if left:
                tmp.unlink(missing_ok=True)
                return self._json({"error": "upload interrupted"}, 400)
            os.replace(tmp, dst)
            _check_cache[("size", dst)] = dst.stat().st_size          # fully written: skip the "copying" wait
            if video_check(dst) == "broken":
                bad = ROOT / "egohack-data" / "incomplete"
                bad.mkdir(parents=True, exist_ok=True)
                os.replace(dst, bad / name)
                return self._json({"error": "this video is incomplete or damaged (the download probably didn't "
                                            "finish). Download it again and re-add it."}, 400)
            if q.get("analyse", ["0"])[0] == "1":
                JOBS.add(dst.stem)
            return self._json({"ok": True, "stem": dst.stem})
        return self._json({"error": "not found"}, 404)

    def _file(self, path, ctype):
        """Static file with HTTP Range support, so the video can be scrubbed."""
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
            if start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def main():
    ap = argparse.ArgumentParser(description="GripTrace local app")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-auto", action="store_true", help="don't start analyses or AI naming on your own")
    a = ap.parse_args()
    global AUTO
    AUTO = not a.no_auto
    threading.Thread(target=auto_queue, daemon=True).start()
    CLIPS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    n = len(list_clips())
    print(f"\n  GripTrace is running at {url}\n  {n} clip(s) found · clips: {CLIPS} · results: {OUT}\n"
          f"  AI object naming: {'ON' if key_available() else 'OFF (put your Anthropic API key in api_key.txt)'}"
          f" · automatic mode: {'ON' if AUTO else 'OFF'}\n"
          f"  Only this computer can open it. Close this window to stop.\n", flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
