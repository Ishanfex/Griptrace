"""Vision agent: names what each hand holds or touches, using Claude's vision API.

Why: the local detectors (YOLO-World + CLIP) were never trained on these factory parts. On 33 hand-judged holds
they named 9 correctly. A large vision model, shown the crop around the hand, does much better.

What it does for one clip, after run.py has produced <clip>.results.json (no detectors are re-run):
  1. Survey   - 6 frames from the clip -> the setting, the likely task and a word list of the parts and tools
                at this workstation (so every new video gets its own vocabulary automatically).
  2. Name     - every contact segment -> one crop around the marked hand, 12 numbered crops per request ->
                holding / touching / empty + the object's name.
  3. Report   - the named timeline -> a short work summary with steps and timestamps.
  4. Write    - updates results.json, contacts.csv and hands.jsonl in place, plus report.md.

Only small JPEG crops are sent (never the video). Data rules allow cloud AI APIs; nothing is stored by the API.
Needs an API key: environment variable ANTHROPIC_API_KEY, or the first line of api_key.txt in the project folder
(git-ignored). Without a key, or offline, everything else still works and the detector names are kept.

    python -m egohand.agent outputs/X/X.results.json egohack-data/clips/X.mp4 [--force]
"""
import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
API_URL = os.environ.get("GRIPTRACE_API_URL", "https://api.anthropic.com/v1/messages")
MODEL = os.environ.get("GRIPTRACE_MODEL", "claude-sonnet-5-5")
TILE = 320            # px per crop in the grid
GRID = (4, 3)         # 12 crops per request
MARK = (255, 0, 255)  # magenta dots mark the hand in question (BGR)
STATE_IDX = {"none": 0, "portable": 1, "fixed": 2, "not_visible": 3}


class AgentError(Exception):
    pass


# ------------------------------------------------------------------ API
def api_key():
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not k:
        f = ROOT / "api_key.txt"
        if f.exists():
            txt = f.read_text(encoding="utf-8-sig").strip()
            k = txt.splitlines()[0].strip() if txt else ""
    return k or None


def available():
    return api_key() is not None


def _jpeg(img, q=82):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def ask(images, text, max_tokens=2000, transport=None):
    """One Messages API call with images (BGR arrays) + text. Returns the reply text."""
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg(im)}}
               for im in images] + [{"type": "text", "text": text}]
    body = {"model": MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    if transport:                                  # tests inject a fake API
        return transport(body)
    key = api_key()
    if not key:
        raise AgentError("no API key (set ANTHROPIC_API_KEY or create api_key.txt)")
    data = json.dumps(body).encode()
    for attempt in range(4):
        req = urllib.request.Request(API_URL, data=data, method="POST", headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            return "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (401, 403):
                raise AgentError(f"API key rejected ({e.code}). Check api_key.txt.")
            if e.code == 404:
                raise AgentError(f"model {MODEL} not available to this key ({detail}). Set GRIPTRACE_MODEL.")
            if e.code in (429, 500, 502, 503, 529) and attempt < 3:
                time.sleep(4 * (attempt + 1))
                continue
            raise AgentError(f"API error {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            raise AgentError(f"cannot reach the API ({e}). Are you online?")
    raise AgentError("API failed after retries")


def parse_json(text):
    """First JSON object/array in a reply (models sometimes wrap it in prose or code fences)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise AgentError("reply had no JSON")
    s = min(starts)
    for end in range(len(text), s, -1):
        try:
            return json.loads(text[s:end])
        except json.JSONDecodeError:
            continue
    raise AgentError("could not parse the reply")


# ------------------------------------------------------------------ frames and crops
class Clip:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise AgentError(f"cannot open {path}")

    def frame(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = self.cap.read()
        return img if ok else None

    def scan(self, wanted):
        """Yield (i, frame) for the wanted frame indices in one forward pass: far faster than seeking
        for every frame of an H.265 file."""
        wanted = sorted(set(int(i) for i in wanted))
        if not wanted:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
        pos = wanted[0]
        for i in wanted:
            if i - pos > 150:                      # long gap: seeking is cheaper than decoding through it
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                pos = i
            while pos < i:
                if not self.cap.grab():
                    return
                pos += 1
            ok, img = self.cap.read()
            pos += 1
            if not ok:
                return
            yield i, img

    def close(self):
        self.cap.release()


def hand_kp(res, i, side):
    r = res["frames"][i][0 if side == "left" else 1]
    return None if r is None else np.array(r[3:45], float).reshape(21, 2)


def best_frame(res, seg):
    """A frame inside the segment where the hand is tracked, nearest the middle."""
    fps = res["meta"]["fps"]
    n = len(res["frames"])
    a, b = int(seg["t_start"] * fps), min(n - 1, int(seg["t_end"] * fps))
    mid = (a + b) // 2
    for d in range(0, max(1, (b - a) // 2 + 1)):
        for i in (mid - d, mid + d):
            if a <= i <= b and hand_kp(res, i, seg["hand"]) is not None:
                return i
    return None


def hand_crop(img, kp, out=TILE):
    """Square crop around the hand (2.6x its size, so the held object is in view), with the hand marked."""
    H, W = img.shape[:2]
    size = max(np.ptp(kp[:, 0]), np.ptp(kp[:, 1]))
    side = int(max(2.6 * size, 0.22 * H))
    c = (kp[[0, 4, 8, 12, 16, 20]].mean(0) * 0.5 + kp.mean(0) * 0.5)
    x1, y1 = int(c[0] - side / 2), int(c[1] - side / 2)
    pad = cv2.copyMakeBorder(img, side, side, side, side, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = pad[y1 + side:y1 + 2 * side, x1 + side:x1 + 2 * side].copy()
    s = out / side
    crop = cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)
    for j in (0, 4, 8, 12, 16, 20):                              # wrist + fingertips
        p = ((kp[j] - [x1, y1]) * s).astype(int)
        cv2.circle(crop, tuple(int(v) for v in p), 4, MARK, -1, cv2.LINE_AA)
        cv2.circle(crop, tuple(int(v) for v in p), 5, (0, 0, 0), 1, cv2.LINE_AA)
    return crop


def grid(tiles, labels):
    cols, rows = GRID
    G = np.zeros((rows * TILE, cols * TILE, 3), np.uint8)
    for n, (t, lab) in enumerate(zip(tiles, labels)):
        r, c = divmod(n, cols)
        G[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = t
        cv2.rectangle(G, (c * TILE, r * TILE), (c * TILE + 44, r * TILE + 30), (0, 0, 0), -1)
        cv2.putText(G, str(lab), (c * TILE + 6, r * TILE + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.rectangle(G, (c * TILE, r * TILE), ((c + 1) * TILE - 1, (r + 1) * TILE - 1), (60, 60, 60), 1)
    return G


# ------------------------------------------------------------------ the three steps
SURVEY_PROMPT = """These 6 frames come from one head-mounted (egocentric) camera clip filmed in a factory or workshop.
Describe the workstation. Return only JSON:
{"setting": "<5-10 words, e.g. 'LED tube light assembly bench'>",
 "task": "<one sentence: what the worker appears to be doing>",
 "objects": [{"name": "<short lowercase noun phrase>", "kind": "portable" or "fixed"}]}
List 10-25 distinct physical things the worker's hands could hold or touch: parts, tools, materials, containers
(portable), and surfaces or machines (fixed). Use plain, specific names a supervisor would use
(e.g. "led tube", "circuit board", "digital caliper", "workbench"), not brand names."""

NAME_PROMPT = """These are {n} numbered crops from head-mounted camera video at: {setting}.
In each crop, ONE of the worker's hands is marked with magenta dots (wrist and fingertips). Our hand tracker
thinks that hand is in contact with something. For each crop, say what the MARKED hand is doing:
- "holding": it grasps or carries an object
- "touching": it rests on or presses a surface or a fixed machine (table, bench, machine body)
- "none": the hand is empty and not touching anything
- "unclear": you cannot tell
Name the object. Prefer these names for this workstation when one fits: {vocab}.
Otherwise use a short, specific lowercase name (1-3 words). Do not guess brands.
Return only JSON, one entry per crop:
[{{"id": 1, "contact": "holding", "object": "led tube", "confidence": 0.9}}]"""

REPORT_PROMPT = """You are writing a short shift note for a factory supervisor from head-camera activity data.
Setting: {setting}. Clip length: {length}.
Contacts per hand (start-end seconds, hand, holding/touching, object):
{timeline}

Return only JSON:
{{"summary": "<2-3 sentences: what the worker did, plainly>",
  "steps": [{{"t_start": <s>, "t_end": <s>, "step": "<what happened, 3-10 words>"}}],
  "observations": ["<0-4 short factual notes, e.g. long idle gaps, one hand doing most of the work>"]}}
Use 3-8 steps that cover the clip in order. Do not invent objects that are not in the data."""


def survey(res, clip, transport=None):
    n = len(res["frames"])
    imgs = []
    for f in np.linspace(0.08, 0.92, 6):
        img = clip.frame(int(f * (n - 1)))
        if img is not None:
            imgs.append(cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))))
    if not imgs:
        raise AgentError("could not read frames")
    while len(imgs) < 6:
        imgs.append(np.zeros_like(imgs[0]))
    sheet = np.vstack([np.hstack(imgs[0:3]), np.hstack(imgs[3:6])])
    out = parse_json(ask([sheet], SURVEY_PROMPT, 1200, transport))
    objs = [o for o in out.get("objects", []) if isinstance(o, dict) and o.get("name")]
    return {"setting": str(out.get("setting", "factory workstation"))[:120], "task": str(out.get("task", ""))[:300],
            "objects": [{"name": str(o["name"]).lower().strip()[:40], "kind": "fixed" if o.get("kind") == "fixed"
                         else "portable"} for o in objs][:30]}


def name_segments(res, clip, scene, log=print, transport=None, force=False):
    segs = [s for s in res["segments"] if s["state"] in ("portable", "fixed")
            and (force or "ai" not in s)]
    vocab = ", ".join(sorted({o["name"] for o in scene["objects"]})) or "(none listed)"
    jobs = []
    for s in segs:
        i = best_frame(res, s)
        if i is not None:
            jobs.append((s, i))
    per = GRID[0] * GRID[1]
    done = 0
    log(f"  agent: cutting {len(jobs)} hand crops ...")
    crops = {}
    by_frame = defaultdict(list)
    for n, (s, i) in enumerate(jobs):
        by_frame[i].append(n)
    for i, img in clip.scan(by_frame):
        for n in by_frame[i]:
            crops[n] = hand_crop(img, hand_kp(res, i, jobs[n][0]["hand"]))
    blank = np.zeros((TILE, TILE, 3), np.uint8)
    for b in range(0, len(jobs), per):
        batch = jobs[b:b + per]
        tiles = [crops.get(b + n, blank) for n in range(len(batch))]
        prompt = NAME_PROMPT.format(n=len(batch), setting=scene["setting"], vocab=vocab)
        try:
            reply = parse_json(ask([grid(tiles, range(1, len(batch) + 1))], prompt, 1500, transport))
        except AgentError as e:
            if "key" in str(e) or "reach" in str(e) or "model" in str(e):
                raise
            log(f"  agent: batch {b // per + 1} skipped ({e})")
            continue
        by_id = {int(r.get("id", -1)): r for r in reply if isinstance(r, dict)} if isinstance(reply, list) else {}
        for n, (s, i) in enumerate(batch, 1):
            r = by_id.get(n)
            if r:
                s["ai"] = {"contact": str(r.get("contact", "unclear")).lower(),
                           "object": str(r.get("object", "")).lower().strip()[:40],
                           "confidence": float(r.get("confidence", 0.5) or 0), "frame": i}
                done += 1
        log(f"  agent: named {min(b + per, len(jobs))}/{len(jobs)} contacts")
    return done


def apply_names(res):
    """Turn the agent's answers into segment states and names; return how many changed."""
    changed = 0
    for s in res["segments"]:
        a = s.get("ai")
        if not a or s["state"] not in ("portable", "fixed", "none"):
            continue
        s.setdefault("det_name", s["object_name"])
        s.setdefault("det_state", s["state"])
        st, nm, cf = a["contact"], a["object"], a["confidence"]
        new = None
        if st == "holding" and cf >= 0.4 and nm:
            new = ("portable", nm)
        elif st == "touching" and cf >= 0.4 and nm:
            new = ("fixed", nm)
        elif st == "none" and cf >= 0.7:
            new = ("none", "")
        if new and (s["state"], s["object_name"]) != new:
            s["state"], s["object_name"] = new
            if new[0] == "none":
                s["obj"] = None
            changed += 1
        if new:
            s["named_by"] = "vision"
    return changed


def merge(res, gap=0.35):
    """Join neighbouring segments of one hand that now have the same state and name."""
    out = []
    for side in ("left", "right"):
        segs = sorted([s for s in res["segments"] if s["hand"] == side], key=lambda s: s["t_start"])
        for s in segs:
            p = out[-1] if out and out[-1]["hand"] == side else None
            if p and p["state"] == s["state"] and p["object_name"] == s["object_name"] and s["t_start"] - p["t_end"] <= gap:
                p["t_end"] = max(p["t_end"], s["t_end"])
                p["k1"] = max(p.get("k1", 0), s.get("k1", 0))
            else:
                out.append(s)
    res["segments"] = sorted(out, key=lambda s: (s["t_start"], s["hand"]))


def recount(res):
    use = defaultdict(lambda: {"grabs": 0, "seconds": 0.0, "left": 0, "right": 0})
    for s in res["segments"]:
        if s["state"] == "portable" and s["object_name"] and s["object_name"] != "unknown object":
            u = use[s["object_name"]]
            u["grabs"] += 1
            u["seconds"] = round(u["seconds"] + s["t_end"] - s["t_start"], 2)
            u[s["hand"]] += 1
    res["tools"] = dict(sorted(use.items(), key=lambda kv: -kv[1]["seconds"]))
    m = res["metrics"]
    m["grabs"] = sum(s["state"] == "portable" for s in res["segments"])
    m["fixed_contacts"] = sum(s["state"] == "fixed" for s in res["segments"])
    # per-frame contact state follows the segments (the viewer's hand cards and overlays read it)
    fps = res["meta"]["fps"]
    for s in res["segments"]:
        a, b = int(round(s["t_start"] * fps)), min(len(res["frames"]), int(round(s["t_end"] * fps)))
        n = 0 if s["hand"] == "left" else 1
        for i in range(a, b):
            r = res["frames"][i][n]
            if r is not None and s["state"] in STATE_IDX and r[1] != STATE_IDX["not_visible"]:
                r[1] = STATE_IDX[s["state"]]
                if s["state"] == "none":
                    r[2] = -1


def report(res, scene, transport=None):
    lines = []
    for s in sorted(res["segments"], key=lambda s: s["t_start"]):
        if s["state"] in ("portable", "fixed"):
            lines.append(f'{s["t_start"]:.1f}-{s["t_end"]:.1f} {s["hand"]} '
                         f'{"holding" if s["state"] == "portable" else "touching"} {s["object_name"]}')
    if len(lines) > 220:                       # keep the request small: keep the 220 longest contacts, in order
        dur = lambda l: float(l.split()[0].split("-")[1]) - float(l.split()[0].split("-")[0])
        cut = sorted((dur(l) for l in lines), reverse=True)[219]
        lines = [l for l in lines if dur(l) >= cut][:220]
    T = len(res["frames"]) / res["meta"]["fps"]
    out = parse_json(ask([], REPORT_PROMPT.format(setting=scene["setting"], length=f"{T:.0f} s",
                                                  timeline="\n".join(lines) or "(no contacts)"), 1500, transport))
    steps = [{"t_start": float(x.get("t_start", 0)), "t_end": float(x.get("t_end", 0)), "step": str(x.get("step", ""))[:120]}
             for x in out.get("steps", []) if isinstance(x, dict)]
    return {"summary": str(out.get("summary", ""))[:800], "steps": steps[:10],
            "observations": [str(o)[:200] for o in out.get("observations", [])][:5]}


# ------------------------------------------------------------------ files
def write_outputs(res_path, res):
    out_dir = Path(res_path).parent
    stem = res["meta"]["clip"].rsplit(".", 1)[0]
    tmp = Path(res_path).with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    os.replace(tmp, res_path)
    with open(out_dir / "contacts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "t_start", "t_end", "hand", "state", "object_name"])
        for s in sorted(res["segments"], key=lambda s: (s["t_start"], s["hand"])):
            w.writerow([stem, f'{s["t_start"]:.2f}', f'{s["t_end"]:.2f}', s["hand"], s["state"], s["object_name"]])
    hj = out_dir / "hands.jsonl"
    if hj.exists():                             # keep per-frame contact labels consistent with the segments
        names = {"none": "none", "portable": "portable", "fixed": "fixed", "not_visible": "not_visible"}
        lines = []
        with open(hj) as f:
            for line in f:
                d = json.loads(line)
                for h in d["hands"]:
                    n = 0 if h["side"] == "left" else 1
                    r = res["frames"][d["frame"]][n] if d["frame"] < len(res["frames"]) else None
                    if r is not None:
                        h["contact"] = names[list(STATE_IDX)[r[1]]]
                        if r[1] == 0:
                            h["object_id"] = None
                lines.append(json.dumps(d))
        hj.write_text("\n".join(lines) + "\n")
    m = (out_dir / "metrics.json")
    m.write_text(json.dumps(res["metrics"], indent=2))
    a = res.get("agent", {})
    rep = a.get("report") or {}
    md = [f"# {stem}", "", f"**Setting:** {a.get('scene', {}).get('setting', '')}", "",
          rep.get("summary", ""), "", "## Steps", ""]
    md += [f"- {x['t_start']:.0f}–{x['t_end']:.0f} s: {x['step']}" for x in rep.get("steps", [])]
    if rep.get("observations"):
        md += ["", "## Observations", ""] + [f"- {o}" for o in rep["observations"]]
    md += ["", "## Objects handled", "", "| Object | Grabs | Seconds | Left | Right |", "|---|---|---|---|---|"]
    md += [f"| {k} | {u['grabs']} | {u['seconds']:.0f} | {u['left']} | {u['right']} |" for k, u in res["tools"].items()]
    md += ["", f"_Names by {a.get('model', 'vision model')} from hand crops; {a.get('named', 0)} contacts named._"]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def run_agent(res_path, clip_path, force=False, log=print, transport=None):
    """Full agent pass on one clip. Returns a short status string; never raises on API problems."""
    res_path = Path(res_path)
    res = json.loads(res_path.read_text())
    if res.get("agent", {}).get("status") == "done" and not force:
        return "already named"
    if not transport and not available():
        return "skipped: no API key"
    t0 = time.time()
    clip = Clip(clip_path)
    try:
        log("  agent: surveying the workstation ...")
        scene = survey(res, clip, transport)
        log(f"  agent: {scene['setting']} · {len(scene['objects'])} object names")
        n = name_segments(res, clip, scene, log, transport, force)
        changed = apply_names(res)
        merge(res)
        recount(res)
        try:
            rep = report(res, scene, transport)
        except AgentError as e:
            log(f"  agent: report skipped ({e})")
            rep = None
        res["agent"] = {"status": "done", "model": MODEL, "scene": scene, "report": rep, "named": n,
                        "changed": changed, "seconds": round(time.time() - t0, 1)}
        write_outputs(res_path, res)
        msg = f"named {n} contacts, changed {changed}, {res['metrics']['grabs']} grabs"
        log(f"  agent: done · {msg} · {time.time() - t0:.0f}s")
        return msg
    except AgentError as e:
        log(f"  agent: stopped - {e} (detector names kept)")
        return f"failed: {e}"
    finally:
        clip.close()


def main():
    ap = argparse.ArgumentParser(description="Name held objects with Claude's vision API")
    ap.add_argument("results")
    ap.add_argument("clip")
    ap.add_argument("--force", action="store_true", help="re-ask even if already named")
    a = ap.parse_args()
    msg = run_agent(a.results, a.clip, force=a.force)
    print(msg)
    sys.exit(1 if msg.startswith("failed") else 0)


if __name__ == "__main__":
    main()
