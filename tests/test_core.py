"""Run:  python -m pytest -q      (or: python -m tests.test_core)

Unit tests for each building block + the synthetic end-to-end benchmark with
minimum scores, so a change that breaks accuracy is caught immediately.
"""
import numpy as np

from egohand import contact as C
from egohand.config import Config
from egohand.handdet import RawHand
from egohand.postprocess import OneEuro
from egohand.tracker import HandTracker, Tracked
from tests import synthetic as S


def test_one_euro_reduces_noise_without_lag():
    rng = np.random.default_rng(0)
    t = np.arange(300) / 30
    truth = np.sin(t)[:, None] * 0.2
    noisy = truth + rng.normal(0, 0.01, truth.shape)
    f, b = OneEuro(1.0, 4.0, 1.0), OneEuro(1.0, 4.0, 1.0)
    fwd = np.array([f(x, ti) for x, ti in zip(noisy, t)])
    bwd = np.array([b(x, ti) for x, ti in zip(noisy[::-1], t[::-1])])[::-1]
    zero_phase = (fwd + bwd) / 2                      # what the pipeline does (offline)
    assert np.abs(zero_phase - truth).mean() < 0.8 * np.abs(noisy - truth).mean()
    # and it is smoother: much less frame-to-frame acceleration
    acc = lambda a: np.abs(np.diff(a, 2, axis=0)).mean()
    assert acc(zero_phase) < 0.3 * acc(noisy)


def test_grip_score_open_vs_closed():
    assert C.grip_score(S.hand_pts("right", np.zeros(2), 100, 0.0)) < 0.2
    assert C.grip_score(S.hand_pts("right", np.zeros(2), 100, 1.0)) > 0.8


def test_hysteresis_removes_one_frame_flicker():
    cfg = Config()
    seq = [("none", None)] * 5 + [("portable", 1)] + [("none", None)] * 5
    cand = [{"left": s, "right": ("none", None)} for s in seq]
    out = C.hysteresis(cand, cfg)
    assert all(fr["left"] == ("none", None) for fr in out)


def test_hysteresis_dates_change_back():
    cfg = Config()
    seq = [("none", None)] * 5 + [("portable", 1)] * 10
    out = C.hysteresis([{"left": s, "right": ("none", None)} for s in seq], cfg)
    assert out[5]["left"] == ("portable", 1)        # starts on the true first frame, not 3 later


def test_tracker_ignores_small_far_hand_and_keeps_sides():
    cfg = Config(swap_handedness="false")
    tr = HandTracker(cfg, 640, 360)
    L = S.hand_pts("left", np.array([170, 340.0]), 95)
    R = S.hand_pts("right", np.array([470, 340.0]), 95)
    tiny = S.hand_pts("right", np.array([320, 60.0]), 15)
    for k in range(10):
        fr = tr.step([RawHand(L, "left", .9), RawHand(R, "right", .9), RawHand(tiny, "left", .9)], k)
    assert fr["left"].kp.mean(0)[0] < 300 and fr["right"].kp.mean(0)[0] > 300
    assert tr.n_rejected_small >= 10


def test_handedness_auto_calibration_detects_mirroring():
    tr = HandTracker(Config(), 640, 360)
    L = S.hand_pts("left", np.array([170, 340.0]), 95)
    R = S.hand_pts("right", np.array([470, 340.0]), 95)
    for k in range(10):                                 # MediaPipe-style mirrored labels
        fr = tr.step([RawHand(L, "right", .9), RawHand(R, "left", .9)], k)
    assert tr.swap and fr["left"].kp.mean(0)[0] < 300


def test_safety_only_when_hazard_moving():
    cfg = Config()
    kp = S.hand_pts("right", np.array([300, 300.0]), 95)
    hist = [{"left": None, "right": Tracked(kp, .9, "mp", "left", 0)} for _ in range(10)]
    states = [{"left": ("not_visible", None), "right": ("none", None)} for _ in range(10)]
    box = np.array([280, 150, 360, 230.0])
    objs = [{7: box} for _ in range(10)]
    times = [k / 15 for k in range(10)]
    still = C.safety_events(hist, states, objs, [7], [{7: 0.0}] * 10, {7: "saw blade"}, times, 1 / 15, cfg)
    moving = C.safety_events(hist, states, objs, [7], [{7: 30.0}] * 10, {7: "saw blade"}, times, 1 / 15, cfg)
    assert not still and len(moving) == 1


def test_synthetic_benchmark_beats_baseline():
    S.rng = np.random.default_rng(7)
    res, ev, _ = S.run_once()
    base_found, ours_found = ev["hand_found_%"]
    base_side, ours_side = ev["side_correct_%"]
    base_err, ours_err = ev["kp_error_px"]
    assert ours_found > base_found + 15
    assert ours_side > 95
    assert ours_err < base_err
    assert ev["contact_state_acc_%"][1] > 85
    grabs = [s for s in res["segments"] if s["state"] == "portable"]
    assert len(grabs) == 1 and grabs[0]["object_name"] == "wrench"
    assert abs(grabs[0]["t_start"] - 3.2) < 0.5 and abs(grabs[0]["t_end"] - 7.0) < 0.5


def test_runs_without_object_detector():
    S.rng = np.random.default_rng(3)
    cfg = Config(objects="none")
    from egohand.objects import ObjectDetector
    import tempfile
    from pathlib import Path
    from egohand.pipeline import run
    tmp = Path(tempfile.mkdtemp())
    S.make_video(tmp / "c.mp4")
    res = run(tmp / "c.mp4", tmp / "o", Config(objects="none", render=False), hand_detector=S.FakeHands(),
              object_detector=ObjectDetector(cfg), log=lambda *a: None, max_frames=60)
    assert (tmp / "o" / "hands.jsonl").exists() and (tmp / "o" / "contacts.csv").exists()
    assert res["metrics"]["processed_frames"] == 60


def test_coworker_hand_across_table_is_ignored():
    """Real-footage failure: a co-worker's hand high in the frame, fingers pointing
    sideways/down, used to be taken as the wearer's hand."""
    tr = HandTracker(Config(swap_handedness="false"), 640, 360)
    L = S.hand_pts("left", np.array([200, 340.0]), 95)
    R = S.hand_pts("right", np.array([440, 340.0]), 95)
    other = S.hand_pts("right", np.array([300, 60.0]), 80)
    other = other * np.array([1, -1]) + np.array([0, 120])   # flipped: fingers point down
    for k in range(10):
        fr = tr.step([RawHand(L, "left", .9), RawHand(other, "right", .95)], k)
    assert fr["right"] is None and tr.n_rejected_other >= 10
    fr = tr.step([RawHand(L, "left", .9), RawHand(R, "right", .9), RawHand(other, "right", .95)], 10)
    assert fr["right"] is not None and fr["right"].kp[0, 1] > 300


def test_forearm_entry_point():
    from egohand.tracker import forearm_entry
    R = S.hand_pts("right", np.array([300, 200.0]), 95)
    R_tilted = (R - R[0]) @ np.array([[0.8, 0.6], [-0.6, 0.8]]) + R[0]   # fingers lean left
    assert abs(forearm_entry(R, 640, 360) - 300 / 640) < 0.05          # straight down
    assert forearm_entry(R_tilted, 640, 360) > 300 / 640               # arm comes from the right


def test_clip_namer_renames_and_drops_empty_holds():
    """CLIP is not installed in CI, so use a fake scorer with the same interface."""
    import tempfile
    from pathlib import Path
    from egohand.namer import name_segments
    tmp = Path(tempfile.mkdtemp())
    S.make_video(tmp / "c.mp4")
    kp = S.hand_pts("right", np.array([470, 340.0]), 95, 1.0)
    hist = [{"left": None, "right": Tracked(kp, .9, "mp", "right", 0)} for _ in range(30)]
    proc = list(range(0, 60, 2))
    segs = [{"hand": "right", "state": "portable", "obj": 3, "object_name": "brush", "k0": 0, "k1": 9},
            {"hand": "right", "state": "portable", "obj": None, "object_name": "unknown object", "k0": 15, "k1": 25}]

    class Fake:
        labels = ["wrench", "green tube", None]
        calls = 0

        def __call__(self, crops):
            Fake.calls += 1
            return np.array([0.7, 0.2, 0.1]) if Fake.calls == 1 else np.array([0.1, 0.1, 0.8])

    n = name_segments(tmp / "c.mp4", segs, hist, proc, 1.0, Config(), scorer=Fake())
    assert n == 1
    assert segs[0]["object_name"] == "brush"              # detector box: its name is kept
    assert segs[0]["clip_top"] == "wrench"                # CLIP's guess is still recorded
    assert segs[1]["state"] == "none"                     # "empty hand" won
    segs[0]["object_name"] = "brush"
    Fake.calls = 0
    n = name_segments(tmp / "c.mp4", segs[:1], hist, proc, 1.0,
                      Config(clip_rename_boxed=True), scorer=Fake())
    assert segs[0]["object_name"] == "wrench" and segs[0]["named_by"] == "clip"


def test_clip_namer_prior_correction_beats_favourite_word():
    """CLIP said "power drill" for most real holds. With 6 holds where "drill" is always
    high but "gauge" stands out in two of them, those two must become "gauge"."""
    import tempfile
    from pathlib import Path
    from egohand.namer import name_segments
    tmp = Path(tempfile.mkdtemp())
    S.make_video(tmp / "c.mp4")
    kp = S.hand_pts("right", np.array([470, 340.0]), 95, 1.0)
    hist = [{"left": None, "right": Tracked(kp, .9, "mp", "right", 0)} for _ in range(60)]
    proc = list(range(0, 120, 2))
    segs = [{"hand": "right", "state": "portable", "obj": None, "object_name": "unknown object",
             "k0": 10 * i, "k1": 10 * i + 8} for i in range(6)]
    seq = [[.60, .10, .20, .10]] * 4 + [[.50, .35, .05, .10]] * 2   # drill, gauge, tube, empty

    class Fake:
        labels = ["power drill", "digital gauge", "green tube", None]
        n = -1

        def __call__(self, crops):
            Fake.n += 1
            return np.array(seq[Fake.n])

    name_segments(tmp / "c.mp4", segs, hist, proc, 1.0, Config(), scorer=Fake())
    names = [s["object_name"] for s in segs]
    assert names[4:] == ["digital gauge", "digital gauge"]
    assert names[:4] == ["power drill"] * 4             # really held most of the time: kept


def test_agent_names_contacts_with_fake_api():
    """The vision agent end to end, with a fake API: survey -> naming -> report -> files rewritten."""
    import tempfile
    import json as _json
    from pathlib import Path
    from egohand import agent as A
    tmp = Path(tempfile.mkdtemp())
    S.make_video(tmp / "c.mp4")
    kp = S.hand_pts("right", np.array([470, 300.0]), 95, 1.0)
    row = [0, 1, 3] + [int(v) for v in kp.ravel()]
    frames = [[None, list(row)] for _ in range(90)]
    segs = [{"hand": "right", "state": "portable", "obj": 3, "object_name": "scissors", "k0": 0, "k1": 14, "t_start": 0.0, "t_end": 1.0},
            {"hand": "right", "state": "portable", "obj": 3, "object_name": "hammer", "k0": 15, "k1": 29, "t_start": 1.0, "t_end": 2.0},
            {"hand": "right", "state": "fixed", "obj": 0, "object_name": "table", "k0": 30, "k1": 44, "t_start": 2.0, "t_end": 3.0},
            {"hand": "left", "state": "not_visible", "obj": None, "object_name": "", "k0": 0, "k1": 44, "t_start": 0.0, "t_end": 3.0}]
    res = {"meta": {"clip": "c.mp4", "fps": 30.0, "width": 640, "height": 360, "n_frames": 90}, "frames": frames,
           "segments": segs, "tools": {}, "safety": [], "metrics": {"grabs": 2, "fixed_contacts": 1}}
    rp = tmp / "c.results.json"
    rp.write_text(_json.dumps(res))
    (tmp / "hands.jsonl").write_text("\n".join(_json.dumps({"frame": i, "t": i / 30, "hands": [
        {"side": "right", "contact": "portable", "object_id": 3}]}) for i in range(90)) + "\n")
    calls = []

    def fake(body):
        text = body["messages"][0]["content"][-1]["text"]
        n_img = sum(c["type"] == "image" for c in body["messages"][0]["content"])
        calls.append(text[:20])
        if "Describe the workstation" in text:
            assert n_img == 1
            return '{"setting": "LED tube assembly bench", "task": "assembling", "objects": [{"name": "led tube", "kind": "portable"}, {"name": "workbench", "kind": "fixed"}]}'
        if "numbered crops" in text:
            assert n_img == 1 and "led tube" in text
            return 'Here you go: ```json [{"id": 1, "contact": "holding", "object": "led tube", "confidence": 0.9}, {"id": 2, "contact": "holding", "object": "led tube", "confidence": 0.8}, {"id": 3, "contact": "none", "object": "", "confidence": 0.9}] ```'
        assert n_img == 0 and "led tube" in text
        return '{"summary": "Worker handles an LED tube.", "steps": [{"t_start": 0, "t_end": 2, "step": "holds the LED tube"}], "observations": []}'

    msg = A.run_agent(rp, tmp / "c.mp4", log=lambda *a: None, transport=fake)
    out = _json.loads(rp.read_text())
    right = [s for s in out["segments"] if s["hand"] == "right"]
    assert msg.startswith("named 3"), msg
    assert right[0]["object_name"] == "led tube" and right[0]["t_end"] == 2.0       # two holds merged
    assert right[0]["det_name"] == "scissors" and right[0]["named_by"] == "vision"
    assert right[1]["state"] == "none"                                              # empty hand corrected
    assert out["tools"]["led tube"]["grabs"] == 1 and out["metrics"]["grabs"] == 1
    assert out["frames"][70][1][1] == 0                                             # per-frame state follows
    assert out["agent"]["report"]["summary"].startswith("Worker")
    assert "led tube,1" not in (tmp / "contacts.csv").read_text() and "led tube" in (tmp / "contacts.csv").read_text()
    last = _json.loads((tmp / "hands.jsonl").read_text().splitlines()[70])
    assert last["hands"][0]["contact"] == "none"
    assert (tmp / "report.md").exists() and len(calls) == 3
    assert A.run_agent(rp, tmp / "c.mp4", log=lambda *a: None, transport=fake) == "already named"


def test_agent_without_key_is_skipped():
    import os
    from egohand import agent as A
    old_env, old_root = os.environ.pop("ANTHROPIC_API_KEY", None), A.ROOT
    try:
        A.ROOT = A.Path("/nonexistent")
        assert not A.available()
    finally:
        A.ROOT = old_root
        if old_env:
            os.environ["ANTHROPIC_API_KEY"] = old_env


def test_drop_duplicates_keeps_the_real_hand():
    """Both slots on one hand: the side that was there before and carries on after keeps it."""
    from egohand.postprocess import drop_duplicates
    L = S.hand_pts("left", np.array([200, 300.0]), 90)
    Rr = S.hand_pts("right", np.array([440, 300.0]), 90)
    mk = lambda kp, src="mp": Tracked(kp.copy(), .9, src, "", 0)
    # gap filling slid the right track onto the left hand for one frame
    h = [{"left": mk(L), "right": mk(Rr)} for _ in range(5)]
    h[2]["right"] = mk(L + 3, "interp")
    assert drop_duplicates(h, 640, 360) == 1 and h[2]["right"] is None and h[2]["left"] is not None
    # MediaPipe reported the left hand twice for two frames
    h = [{"left": mk(L), "right": mk(Rr)} for _ in range(5)]
    h[1]["right"] = mk(L + 2); h[2]["right"] = mk(L + 1)
    assert drop_duplicates(h, 640, 360) == 2 and h[1]["right"] is None and h[2]["left"] is not None
    # real case (bench clip, 24.5 s): sideways right hand, the LEFT track jumped onto it while the
    # real left hand was lost; afterwards only the right track carries on there -> keep RIGHT
    side = np.stack([np.linspace(440, 360, 21), np.full(21, 300.0) + np.linspace(0, 5, 21)], 1)
    h = [{"left": mk(L), "right": mk(side)}, {"left": mk(L), "right": mk(side)},
         {"left": mk(side + 2, "interp"), "right": mk(side)}, {"left": mk(side + 1), "right": mk(side, "fill")},
         {"left": None, "right": mk(side)}, {"left": None, "right": mk(side)}]
    assert drop_duplicates(h, 640, 360) == 2
    assert h[2]["left"] is None and h[3]["left"] is None and h[2]["right"] is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
