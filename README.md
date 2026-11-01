# GripTrace — what are the worker's hands touching?

**Problem statement: PS2 — "What is the hand touching?"** (built on a strengthened PS1 hand tracker)
**Hardware: laptop CPU only, no GPU. Video never leaves the laptop; the optional AI naming step sends small hand crops to the Claude API only if you add a key.**

GripTrace turns head-camera factory video into a per-hand timeline of what each hand is
doing: *free*, *holding a portable object* (and which one), or *on a fixed surface*. It also
counts tool use and flags hands near moving hazards. Results open in an offline viewer
that shows plain MediaPipe and our pipeline side by side, and lets the team hand-label
frames to measure real accuracy.

```
clip.mp4 ─► head-motion estimate ─► dark-frame fix ─► MediaPipe Hands (VIDEO mode)
        ─► LEFT/RIGHT tracker (+ zoom-in recovery of missed hands)
        ─► YOLO-World objects every 3rd frame ─► offline clean-up:
           side voting · gap filling · zero-lag One Euro smoothing
        ─► contact rules with physical evidence ─► hysteresis ─► timeline
        ─► hands.jsonl · contacts.csv · <clip>.results.json · <clip>_demo.mp4
```

---

## Quick start (Windows, Python 3.10–3.12)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python prep.py                 :: once, BEFORE the event: downloads model weights
python -m pytest -q            :: optional: 15 tests incl. the synthetic benchmark
```

Run on a clip (full-HD clips are resized to 854 px automatically):

```bat
python run.py egohack-data\clips\MTL-SKL00089-007.mp4
python run.py egohack-data\clips --all                 :: every clip (1080p is auto-resized to 854 px)
python run.py clip.mp4 --objects none                   :: hands only (PS1)
python run.py clip.mp4 --objects coco                   :: if YOLO-World won't install
```

**All clips at once** (the organisers' 40 demo clips): put them in `egohack-data\clips` (or leave Drive downloads in
your Downloads folder; complete files are moved automatically), then double-click **`run_all.bat`**.
It skips clips that are already done (safe to stop and re-run), renders demo videos only for the showcase clips,
and writes **`outputs\summary.md`**, one table over every clip. **`make_demo.bat`** then cuts the
≤ 2-minute demo video for the private Drive folder.

To try different contact settings on an existing result in seconds (no detectors re-run):

```bat
python -m tools.recontact outputs\X\X.results.json egohack-data\clips\X.mp4 --write
```

Then start the app: **`start_app.bat`** (or `python app.py`). It opens in your browser with every clip and result
loaded, and can analyse new clips from there. See "The app" below.

(macOS/Linux: `source .venv/bin/activate`, and use `/` in paths.)

## Outputs (`outputs/<clip>/`, git-ignored)

| File | Contents |
|---|---|
| `hands.jsonl` | One line per frame, in the organisers' format (`side`, `bbox`, `score`, `kp2d`), plus `track_id`, `source` (detected / recovered / interpolated) and `contact` |
| `contacts.csv` | `clip, t_start, t_end, hand, state, object_name` — the organisers' format |
| `<clip>.results.json` | Everything the viewer needs, including MediaPipe's raw output for comparison |
| `metrics.json` | Detection share, jitter, label flips, recoveries, speed per stage |
| `<clip>_demo.mp4` | Rendered demo (skeletons, held-object links, timeline bar) for the private Drive folder |

## The app (`start_app.bat`)

Double-click **`start_app.bat`**. A small local server (`app.py`, Python standard library only) starts and your
browser opens at `http://127.0.0.1:8765`. It listens on this computer only; nothing is uploaded anywhere.

- Every clip in `egohack-data\clips` and every result in `outputs` shows up by itself. No file picking.
- Clips that have not been analysed yet have an **Analyse** button. The pipeline (`run.py`) runs in the
  background, one clip at a time, with a live progress bar and time left. When it finishes, the clip opens like the others.
- **+ Add clip** (or drag a video onto the page) copies a new clip into `egohack-data\clips` and starts analysing it.
- Logs of each run go to `outputs\_app_logs\`.

Without the server, `viewer/index.html` still works on its own: open it and load the folders by hand (below).

## AI object naming (the vision agent, `egohand/agent.py`)

The local detectors don't know these factory parts (9 of 33 judged holds named correctly). With an Anthropic API key,
an agent step runs automatically after each analysis:

1. **Survey:** 6 frames → the workstation, the likely task, and a word list of the parts and tools at this bench.
   Every new video gets its own vocabulary, with nothing to configure.
2. **Name:** every contact → one crop around the hand (hand marked with dots), 12 numbered crops per request to
   Claude → holding / touching / empty + the object's name. Empty-hand false contacts are corrected.
3. **Report:** a short shift note (summary, steps with timestamps, observations), shown on the Overview and saved
   as `outputs\<clip>\report.md`.

`results.json`, `contacts.csv` and `hands.jsonl` are updated in place; the detector's original name is kept as
`det_name`. Only small JPEG crops are sent, never the video (the data rules allow cloud AI APIs). Model:
`claude-sonnet-5-5` (change with the `GRIPTRACE_MODEL` environment variable). Roughly 20 requests per 3-minute clip.

**Setup:** save your key as the only line of `api_key.txt` in the project folder (git-ignored, never commit it), or
set `ANTHROPIC_API_KEY`. Without a key everything else works and the detector names are kept.

```bat
python -m egohand.agent outputs\X\X.results.json egohack-data\clips\X.mp4     :: name one existing result
python run.py clip.mp4 --no-agent                                              :: skip it
```

**Self-running app:** with `start_app.bat` open, any video you add (button, drag-and-drop, or just copying it into
`egohack-data\clips`) is converted for the browser, analysed, AI-named and reported with no clicks. Results made
before the key was added are AI-named automatically too. `python app.py --no-auto` turns this off.

## The viewer (`viewer/index.html`)

Open it in Chrome or Edge (double-click; no server, no install). Click **Load** and choose the `outputs` folder, then
load the `egohack-data\clips` folder too to see video (or drag both folders onto the page). Videos are matched to
results by file name. Everything is read inside the browser; nothing is uploaded.

- **00 All clips:** every loaded clip as a card with its activity strip, plus totals across clips.
- **01 Overview:** the insights first: grabs, how much of the time the hands are busy, median contact time, an
  activity tape for both hands (click to replay that moment), hand split, objects handled, key-moment thumbnails,
  event log and safety events. Exports `contacts.csv`, a copy-ready summary, or a printed report.
- **02 Replay:** video with hand skeletons, held-object brackets and live hand cards; a scrubbable timeline; a
  side-by-side **vs MediaPipe** mode that shows the starter code's output on the same frame.
- **03 Tracking quality:** MediaPipe → GripTrace on this clip, what was fixed, where each hand position came from,
  CPU time per stage, and a table of all loaded clips.
- **04 Audit:** judge 100 frames (the same ones for every teammate, seeded by clip name) with keys `1–6` / `Q–Y` to
  get a measured accuracy number.

Keys: `space` play/pause · `←` `→` step a frame (`shift` = 1 s) · `1`–`4` switch views.

---

## How it works, and why each step exists

### Hand tracking (judged in every PS — 30 pts)

| Problem in head-camera footage | What we do | Where |
|---|---|---|
| MediaPipe labels hands as if the video were a mirrored selfie | **Auto-calibrate per clip:** when two separated hands are visible, the one on the left of the image is almost always the wearer's left. A running vote decides whether to swap labels. | `tracker.py` |
| Labels flicker on blurred or gloved hands | Identity comes from **motion, not labels.** Two fixed slots (LEFT, RIGHT); each frame we try every detection-to-slot assignment (≤21 combinations) and keep the cheapest. The cost uses distance from the predicted position, change in hand size, and label agreement. | `tracker.py` |
| Whole tracks occasionally end up on the wrong side | **Offline side vote per track:** if most of a track's frames say it is the other hand, it is moved, but only where that side is free. | `postprocess.py` |
| Co-workers' hands in view | Hands smaller than 5% of the frame diagonal are ignored. A hand with its wrist in the top 35% of the frame and fingers not pointing up is someone across the table. At most 2 slots exist. | `tracker.py` |
| Hands crossing over | Forearm entry point (knuckle → wrist line extended to the bottom edge) is an extra left/right vote | `tracker.py`, `postprocess.py` |
| Missed hands (blur, gloves, occlusion) | **Zoom-in recovery:** crop around where the missing hand should be, enlarge it, and run MediaPipe again (with a contrast boost if needed). The recovered hand must be nearer its own prediction than the other hand's. | `tracker.py`, `handdet.py` |
| Dark scenes | CLAHE on the lightness channel when mean brightness < 70 | `enhance.py` |
| Short drop-outs | Linear interpolation for gaps ≤ 6 processed frames, drawn dashed and labelled `interp` (never counted as "detected") | `postprocess.py` |
| Jitter | **Zero-lag One Euro filter:** run forwards and backwards and average. This is possible because clips are processed offline. | `postprocess.py` |
| Laptop speed | Process every 2nd frame and fill the frames in between; run the object detector every 3rd processed frame | `video.py`, `pipeline.py` |

### Contact (PS2 — 20 pts)

The organisers' starting rule is "≥ 2 fingertips inside the object's box". In head-camera
video that fires all the time, because hands are almost always *in front of* the bench and
*hovering over* parts in 2D. So a fingertip match also needs **physical evidence**:

- **Fixed surface** (bench, machine, shelf, or any box larger than 22% of the frame): the hand must be **still in scene coordinates**. Head motion is removed first: corner points are tracked on the background (hands masked out), and a similarity transform between frames is fitted and chained (`camera.py`).
- **Portable object:** the fingers must be **gripping** (fingertip-to-palm distance divided by palm size, which stays the same when fingers curl) **or the object must move together with the hand**.
- **Tool lying on a bench:** both qualify; the portable object wins.
- **CLIP naming:** for every hold, crops around the hand at 25/50/75% of the hold are scored by CLIP against
  "a photo of a hand holding a <object>" for every portable object, plus "empty hand" prompts (which turn a false hold
  back into "free"). CLIP over-uses some words ("power drill" won 75 holds on the gauge clip), so each label's score is
  divided by the square root of its average over the clip before picking the winner. **CLIP only names holds where
  the detector has no box** (the "unknown object" holds). We judged 33 real holds (3 clips) both ways: when there is a held
  box, the detector's name was right 9 times and CLIP's only 5, so the detector keeps its name there
  (`clip_rename_boxed = False`; CLIP's guess is still saved as `clip_top` in the results).
- **Hand-zoom detection:** besides the full frame, the detector also runs on a zoomed crop around each hand (2.2× the hand box, 320 px). Held objects are small and half-covered by fingers, so the zoom finds many more of them.
- **Holding an unknown object:** if the fingers clearly grip (grip score ≥ 0.75) but no object box is found, the hand is marked "holding unknown object" rather than "free". This is honest about what the detector missed, and it keeps the grab/release timeline right even when the name is not.
- **No giant boxes:** a "portable" box covering more than 20% of the frame is a false detection and is ignored.
- **Switching objects** needs 6 frames of evidence, and holds shorter than 0.4 s are dropped as flicker.
- **"Sticky grasp":** the hand hides the object it holds, so the detector loses it. We keep the object attached to the hand for up to 1 s (drawn dashed: "held, hidden").
- **Hysteresis:** a contact must last 3 frames to start and 4 to end. The change is then **dated back** to the frame where it really began, so timestamps aren't delayed.
- A "fixed" object that **moved in scene coordinates** during the contact is relabelled portable.
- **Tool usage** = number of portable grabs per object name, with total time and left/right split.
- **Safety:** a hand within half a hand-size of a hazard-class object (saw blade, press, conveyor…) **while that object is moving** after head-motion removal.

Every threshold is in `egohand/config.py`, with a one-line reason next to it.

---

## Results

### Synthetic stress test (known ground truth)

The real clips have no labels, so `tests/synthetic.py` generates a clip where the truth is
known, together with a fake detector that makes MediaPipe's typical mistakes: mirrored
labels, 12% random label flips, 12% misses plus blur bursts, a co-worker's hand, 2.5 px
jitter, a dark stretch, a hand leaving and re-entering, and an object detector that misses
the held object 50% of the time.

| Metric | Starter (MediaPipe as-is) | GripTrace |
|---|---|---|
| Hand found (visible frames) | 69.1% | **99.4%** |
| Correct left/right | 6.4% (mirrored) | **99.4%** |
| Keypoint error | 3.13 px | **2.23 px** |
| Contact state correct (per frame, per hand) | — | **94.9%** |
| Wrench grab, truth 3.2–7.0 s | — | **3.4–7.3 s, one segment** |

⚠️ **Honest caveat:** this proves the logic works; it is not accuracy on real footage. The
fake detector's zoom-in recovery succeeds 60% of the time, and real MediaPipe will do worse.
**For real numbers, use the viewer's Label tab on the demo clips and report those.**

### Real footage (1080p fisheye, processed at 854 px, CPU only)

**Hand-judged accuracy.** 12 random frames per clip on 3 clips (36 frames), every hand judged by eye, the same
frames for both systems. On the workshop clip, frames were drawn from those where either system showed a hand.

| 36 frames (bench assembly, metal grinding, carpentry) | Correct | Missed | Wrong side | False / duplicate hand |
|---|---|---|---|---|
| MediaPipe as-is (starter code) | 48 | 12 | 5 | 7 |
| **GripTrace** | **53** | 11 | **0** | **1** |

**Automatic numbers** (`metrics.json` / `outputs/summary.md`, final code, full 3-minute clips, 2,700 processed
frames each, laptop CPU). "MP" is MediaPipe as-is, the starter-code baseline.

| Clip | Both hands found (MP → ours) | Jitter, % of hand size (MP → ours) | Label flips fixed | Co-worker hands rejected | Grabs | Surface contacts | Median contact |
|---|---|---|---|---|---|---|---|
| MTL-SKL00089-007 (bench assembly) | 73.3% → **90.9%** | 12.3 → **4.8** (−61%) | 317 | 5 | 218 | 14 | 1.0 s |
| mytronlabs-000003-002 (grinding, co-worker opposite) | 47.3% → **55.0%** | 11.6 → **5.0** (−57%) | 173 | 359 | 122 | 22 | 0.9 s |
| MTL-SGF00106-023 (carpentry) | 30.4% → **45.0%** | 13.3 → **4.8** (−64%) | 249 | 2 | 106 | 12 | 1.0 s |
| MTL-SGF00106-021 (workshop, walking) | 11.4% → 13.9% | 11.3 → **4.1** (−64%) | 80 | 69 | 57 | 2 | 0.7 s |
| **All 4 clips** (12 min of video) | 40.6% → **51.2%** | 12.1 → **4.7** (−61%) | 819 | 435 | 503 | 50 | |

Speed: 6–14 video frames per second on a laptop CPU, full pipeline including the object detector.
On mytronlabs the right-hand share barely moves (63.6% → 65.8%) because the baseline count includes the
co-worker's hands, which we now reject. The low share on SGF-021 is real; the worker mostly looks around the
workshop with their hands out of view.

What the real footage taught us (and the code now handles):
- **Labels are not mirrored for these cameras.** The auto-calibration found this on all 3 clips. A fixed "always swap" rule would have made every label wrong.
- **Co-workers across the table** have big hands close to the camera, so a size filter misses them. A hand whose wrist is high in the frame and whose fingers point sideways or down is someone else's (`is_other_person`).
- **Forearm entry point:** extending the knuckle → wrist line to the bottom edge shows which side the arm comes from. It is an extra left/right vote, useful when hands cross.

## Known limitations (we show these in the demo)

- **Hovering vs touching** can't be fully separated from 2D video. A hand held perfectly still just above a bench looks like resting on it. The stillness rule plus a minimum contact time of 0.6 s reduces this but doesn't eliminate it.
- **The object vocabulary matters.** YOLO-World only finds what we name in `config.py` (`PORTABLE`, `FIXED`). We tuned the list after looking at the clips.
- **Grip estimation** depends on finger keypoints, which are weakest with thick gloves. Co-motion with the object is the backup cue.
- **Self-contact** (arms crossed, hands clasped, gripping your own forearm) can show up as "holding unknown object". We tried a skin-colour test to catch it, but it did not separate the cases on real footage (the sample point lands on the fingers), so we removed it rather than ship a heuristic that does not work.
- **Naming the held object is the weakest part.** Judged by hand on 33 holds (12 bench, 11 carpentry, 10 gauge clip):
  letting CLIP rename every hold gave 5 / 33 right names; keeping the detector's name when it has a box and using
  CLIP only for box-less holds gives 9 / 33 (7 / 12 bench, 2 / 11 carpentry, 0 / 10 gauge). Neither is good enough,
  and the best rule differs per clip: on the gauge clip CLIP was right twice ("digital gauge", "dial indicator")
  where the detector was not. "Is the hand holding something?" is right far more often than the name. Wooden
  frames, the red power planer and small machined parts are not recognised by either model. A few-shot fine-tune
  on ~200 labelled parts is the fix.
- **Object names come from YOLO-World.** It is good at "bottle" and "phone" and weak at factory-specific parts. On the real clips the green LED tubes were never detected with the first word list, which is why the list was rewritten and the hand-zoom crops were added.
- **The baseline numbers are slightly generous to MediaPipe:** on dark frames it receives our brightened image.
- **Crossed arms with a bent wrist** can still get the wrong side. The hand points up while the forearm comes from the other side, which fools the forearm cue (seen once in the audit sample).
- **Hands right in front of the lens** while holding big parts (gauges, tubes) are often missed by MediaPipe's palm detector. These were most of our misses.
- **The real-footage accuracy above** comes from one judge and 36 frames. Use the viewer's Label tab for a bigger, team-labelled number.
- **Head-motion compensation** assumes a mostly flat background. Strong parallax (walking past machines) adds noise to the stillness test.

## Data-use rules — how this repo complies

- `.gitignore` excludes all video, frames, outputs, results and labels **before the first commit.**
- No footage, frames or screenshots in this repo or README. The synthetic test generates its own video.
- No cloud processing: everything runs on the laptop. The viewer is a local file and uploads nothing.
- Delete `egohack-data/` and `outputs/` when the event ends.

## Repository layout

```
run.py              one command: clip -> all outputs (--all, --skip-done, --render-only)
prep.py             one-time model download (allowed before the event)
setup_and_test.bat  Windows one-click setup + tests + 30 s real-model check
run_all.bat         process every clip, skip finished ones, write outputs/summary.md
make_demo.bat       cut the <= 2-minute demo video for the private Drive folder
egohand/
  config.py         every threshold + object vocabulary, with reasons
  video.py          frame reader (step / resize)
  enhance.py        low-light CLAHE
  handdet.py        MediaPipe wrapper: VIDEO mode + zoomed-crop recovery
  tracker.py        LEFT/RIGHT identity, handedness calibration, swap repair
  postprocess.py    side voting, gap filling, zero-lag One Euro, frame expansion
  camera.py         head-motion compensation (scene coordinates)
  objects.py        YOLO-World / COCO detector + IoU tracker
  contact.py        contact rules, hysteresis, segments, tool use, safety
  namer.py          CLIP naming of held objects with per-clip bias correction
  agent.py          vision agent: workstation survey, AI naming of each contact, shift report (Claude API)
  metrics.py        baseline-vs-ours numbers
  export.py         hands.jsonl, contacts.csv, viewer JSON
  render.py         demo video
  pipeline.py       glues the passes together
app.py              local app server: serves the viewer, lists clips, runs the pipeline, accepts new clips
start_app.bat       double-click to start the app
viewer/index.html   the frontend (Clips / Overview / Replay / Tracking quality / Audit); also works without the server
tests/              synthetic ground-truth benchmark + unit tests (15)
tools/              recontact (re-score contacts fast), summary (all clips), make_demo, get_clips
```

## Credits and licences

MediaPipe Hand Landmarker (Apache-2.0) · Ultralytics YOLO-World (AGPL-3.0) · OpenCV (Apache-2.0) ·
One Euro filter: Casiez, Roussel & Vogel, CHI 2012. AI coding assistants were used (allowed by the rules);
every team member can explain every file.
