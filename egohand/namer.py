"""Name what each hand is holding with CLIP (offline, CPU).

The object detector names boxes; but a held object is small, half-covered by fingers,
and often never gets a box at all. For every "holding" segment we crop the frame around
the hand at three moments and ask CLIP which description fits best:

    "a photo of a hand holding a <object>"   for every portable object in our list
    "a photo of an empty hand" / "a photo of two hands touching each other"

Prior correction: CLIP has favourite words, so each label's score is divided by its
average over all holds in the clip before picking the winner.

If the "empty" prompts win, the segment is turned back into "touching nothing"
(this removes many false holds such as clasped hands). If CLIP is not confident,
the detector's name is kept. If CLIP is not installed, nothing changes.
"""
import cv2
import numpy as np

EMPTY = ["a photo of an empty hand", "a photo of two hands touching each other",
         "a photo of a hand resting on an arm"]


class ClipScorer:
    """Wraps OpenAI CLIP (installed by Ultralytics for YOLO-World)."""

    def __init__(self, labels):
        import clip
        import torch
        from PIL import Image
        self.torch, self.Image = torch, Image
        self.model, self.pre = clip.load("ViT-B/32", device="cpu", download_root="weights/clip")
        prompts = [f"a photo of a hand holding a {n}" for n in labels] + EMPTY
        with torch.no_grad():
            t = self.model.encode_text(clip.tokenize(prompts))
        self.text = t / t.norm(dim=-1, keepdim=True)
        self.labels = list(labels) + [None] * len(EMPTY)      # None = empty hand

    def __call__(self, rgb_crops):
        """Mean probability over crops for every label."""
        torch = self.torch
        with torch.no_grad():
            x = torch.stack([self.pre(self.Image.fromarray(c)) for c in rgb_crops])
            f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ self.text.T).softmax(dim=-1).mean(0).numpy()
        return p


def hand_crop(img, kp, scale=2.2, min_px=96):
    H, W = img.shape[:2]
    c = kp.mean(0)
    half = max(np.ptp(kp[:, 0]), np.ptp(kp[:, 1]), min_px / 2) * scale / 2
    x1, y1 = int(max(0, c[0] - half)), int(max(0, c[1] - half))
    x2, y2 = int(min(W, c[0] + half)), int(min(H, c[1] + half))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def name_segments(clip_path, segs, hist, proc_idx, scale, cfg, scorer=None, log=print):
    """Rename / drop "portable" segments in place. Returns how many were changed."""
    hold = [s for s in segs if s["state"] == "portable"]
    if not hold:
        return 0
    if scorer is None:
        try:
            scorer = ClipScorer(cfg.vocab_portable)
        except Exception as e:                    # CLIP missing: keep detector names
            log(f"  CLIP namer skipped ({e.__class__.__name__}: {e})")
            return 0
    labels = scorer.labels
    cap = cv2.VideoCapture(str(clip_path))
    probs = {}                                   # segment index -> CLIP probabilities
    for n, s in enumerate(hold):
        ks = sorted({int(round(s["k0"] + f * (s["k1"] - s["k0"]))) for f in (0.25, 0.5, 0.75)})
        crops = []
        for k in ks:
            h = hist[k][s["hand"]]
            if h is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, proc_idx[k])
            ok, img = cap.read()
            if not ok:
                continue
            crop = hand_crop(img, h.kp * scale)
            if crop is not None:
                crops.append(crop)
        if crops:
            probs[n] = np.asarray(scorer(crops), float)

    # Prior correction: CLIP has favourite words (on real footage it said "power drill"
    # for most holds). Divide by how much each label wins on average over this clip, so
    # a label only wins where it stands out. Needs enough holds to estimate the average.
    obj_idx = [i for i, lab in enumerate(labels) if lab is not None]
    prior = None
    if len(probs) >= 5:
        # square root = soft correction: a label that is really held most of the time
        # still wins; it just stops winning by default where something else stands out
        prior = np.sqrt(np.mean([p[obj_idx] for p in probs.values()], axis=0) + 1e-6)
    changed = 0
    for n, p in probs.items():
        s = hold[n]
        p_empty = float(sum(p[i] for i, lab in enumerate(labels) if lab is None))
        q = p[obj_idx] / prior if prior is not None else p[obj_idx]
        best = obj_idx[int(np.argmax(q))]
        top3 = np.argsort(-p)[:3]
        s["clip_top"], s["clip_prob"] = labels[best], round(float(p[best]), 3)
        s["clip_raw_top3"] = [[labels[i] or "empty hand", round(float(p[i]), 3)] for i in top3]
        if p_empty > 0.5:
            s["state"], s["obj"], s["object_name"] = "none", None, ""
            changed += 1
        elif p[best] >= cfg.clip_min_prob and (s["obj"] is None or getattr(cfg, "clip_rename_boxed", False)):
            # Judged on 33 real holds (3 clips): with a detector box on the held object its
            # name was right 9 times, CLIP's only 5, so CLIP names box-less holds only.
            if s["object_name"] != labels[best]:
                s["object_name"] = labels[best]
                changed += 1
            s["named_by"] = "clip"
    cap.release()
    return changed
