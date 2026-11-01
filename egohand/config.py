"""All tunable settings in one place, so every threshold can be explained to judges.

Units: "processed frames" means frames we actually run models on (every `step`-th
frame of the video). "hand sizes" means distances divided by the hand's box diagonal,
so rules work the same at 480p and 1080p.
"""
from dataclasses import dataclass, field

# MediaPipe landmark indices
WRIST = 0
TIPS = [4, 8, 12, 16, 20]          # thumb, index, middle, ring, pinky fingertips
MCPS = [1, 5, 9, 13, 17]            # finger bases (used for palm centre)

# Object vocabulary for the open-vocabulary detector (YOLO-World).
# Tuned after looking at the real clips (bench assembly of LED tubes, metal gauging at a
# grinding wheel, a carpentry workshop). Naming what is really in the footage is the single
# biggest accuracy gain for PS2. Generic words that caused confusion on real footage
# ("lid", "can", "tape measure", "clipboard", "component") were removed.
PORTABLE = [
    # bench assembly (MTL-SKL clips)
    "green tube", "LED tube light", "circuit board", "tape dispenser", "tape roll",
    "wire", "cup", "mobile phone",
    # metal work / gauging (mytronlabs clips)
    "digital gauge", "dial indicator", "orange plastic handle", "metal part", "plastic bottle",
    # carpentry (MTL-SGF clips)
    "wooden plank", "wooden frame",          # ("plywood sheet" matched bench tops: removed)
    # common hand tools
    "screwdriver", "wrench", "pliers", "hammer", "power drill", "scissors", "knife",
    "glue gun", "brush", "cloth", "paper", "cardboard box",
]
FIXED = [
    "table", "workbench", "grinding wheel", "machine", "shelf", "rack", "cabinet", "wall",
]
# ("glove" is deliberately NOT listed: a gloved hand would "touch a glove" all the time.)
# Things where a hand nearby is worth a warning (only if the part is moving).
HAZARDS = ["grinding wheel", "saw blade", "rotating spindle", "drill bit"]


def vocabulary(cfg):
    """All class names, without duplicates, in a stable order."""
    seen, out = set(), []
    for n in cfg.vocab_portable + cfg.vocab_fixed + cfg.vocab_hazard:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


@dataclass
class Config:
    # ---- input ----
    step: int = 2                      # process every Nth frame (organisers: 2-3 on laptop)
    width: int = 0                     # resize width before processing (0 = keep)

    # ---- MediaPipe ----
    hand_model: str = "models/hand_landmarker.task"
    num_hands: int = 4                 # ask for >2 so we can reject other people's hands
    det_conf: float = 0.35
    presence_conf: float = 0.35
    track_conf: float = 0.35
    # MediaPipe labels hands as if the image were a mirrored selfie. Head-camera
    # video is not mirrored. "auto" learns the right answer per clip from where
    # the hands sit in the frame; True/False forces it.
    swap_handedness: str = "auto"

    # ---- recovery when a hand is missed ----
    dark_luma: float = 70.0            # mean brightness below this = dark frame -> CLAHE
    roi_recovery: bool = True          # re-detect a missed hand in a zoomed crop
    roi_max_lost: int = 6              # try crop recovery for up to N processed frames
    roi_scale: float = 2.2             # crop = predicted hand box x this
    roi_size: int = 256                # crop is resized to this many pixels

    # ---- tracker ----
    min_hand_frac: float = 0.05        # boxes smaller than this x frame diagonal = not the wearer
    other_wrist_y: float = 0.35        # wrist in the top 35% of the frame...
    other_finger_up: float = 0.5       # ...and fingers not pointing up = a co-worker's hand
    other_down_up: float = -0.5        # fingers pointing toward the camera...
    other_down_y: float = 0.7          # ...with the wrist in the top 70% = someone opposite
                                       # (checked on real clips: 6 of 5017 wearer hands affected)
    gate_hand_sizes: float = 1.5       # max jump between frames for the same hand
    max_lost: int = 12                 # forget a hand after N processed frames unseen
    label_weight: float = 0.8          # cost of a left/right label disagreeing with the slot
    swap_window: int = 12              # frames of evidence needed to repair a side swap
    swap_ratio: float = 0.8

    # ---- post-processing ----
    max_gap: int = 6                   # fill gaps up to N processed frames by interpolation
    gap_max_jump: float = 1.5          # ...only if the hand moved < this many hand sizes
    euro_min_cutoff: float = 1.0       # One Euro filter (tuned on the synthetic benchmark)
    euro_beta: float = 4.0
    euro_d_cutoff: float = 1.0
    zero_phase: bool = True            # forward+backward pass = no smoothing lag (offline)

    # ---- objects ----
    objects: str = "world"             # world | coco | none
    object_model: str = "models/yoloworld_factory.pt"   # rebuilt automatically if the vocabulary changes
    object_every: int = 3              # run detector every N processed frames
    object_conf: float = 0.15
    hand_zoom: bool = True             # also detect objects in a zoomed crop around each hand
    hand_zoom_scale: float = 2.2       # crop = hand box x this
    hand_zoom_imgsz: int = 320
    clip_namer: bool = True            # name each held object with CLIP on a crop around the hand
    clip_min_prob: float = 0.10        # raw CLIP probability needed to replace the detector's name
    clip_rename_boxed: bool = False    # False: CLIP names only holds with no detector box (judged better)
    portable_max_area: float = 0.12    # a "portable" box bigger than this share of the frame is a false box
    object_imgsz: int = 640
    vocab_portable: list = field(default_factory=lambda: list(PORTABLE))
    vocab_fixed: list = field(default_factory=lambda: list(FIXED))
    vocab_hazard: list = field(default_factory=lambda: list(HAZARDS))
    obj_iou_match: float = 0.3
    obj_ttl: int = 9                   # keep an unseen object box for N processed frames
    fixed_area_frac: float = 0.22      # a box bigger than this share of the frame is a surface

    # ---- contact ----
    tip_pad: float = 0.15              # pad object box by this x hand size (organisers' rule)
    min_tips: int = 2                  # >=2 fingertips inside = touching (organisers' rule)
    enter_frames: int = 3              # hysteresis: frames needed to start a contact
    exit_frames: int = 4               # ...and to end one
    min_segment_s: float = 0.2
    min_fixed_s: float = 0.6           # resting on a surface must last at least this long
    unknown_grip: float = 0.75         # gripping this hard with no detected object = "holding unknown object"
    switch_frames: int = 6             # changing from one held object to another needs this many frames
    min_portable_s: float = 0.4        # shorter holds are dropped as flicker
    still_sizes_per_s: float = 0.25    # hand slower than this (scene coords) = resting on a surface
    min_grip: float = 0.5              # 0 open .. 1 fist; needed for a portable contact (or co-motion)
    motion_window_s: float = 0.5       # speeds are measured over +- this window
    moved_sizes: float = 0.6           # object moved > this x its size (scene coords) -> portable
    sticky_s: float = 1.0              # keep a held object attached to the hand if detector loses it

    # ---- safety ----
    hazard_pad: float = 0.5            # hand within this x hand size of a moving hazard = warning
    motion_thresh: float = 12.0        # mean pixel change inside hazard box (after camera fix)

    # ---- output ----
    render: bool = True
