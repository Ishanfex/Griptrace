"""Reading frames from a clip (organisers' starter `frames()` with resize + metadata)."""
import cv2


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def frames(path, step=1, width=0):
    """Yield (frame_index, time_s, BGR image) for every `step`-th frame."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok = cap.grab()                      # grab is cheap; decode only frames we use
        if not ok:
            break
        if i % step == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            if width and img.shape[1] != width:
                img = cv2.resize(img, (width, int(round(img.shape[0] * width / img.shape[1]))),
                                 interpolation=cv2.INTER_AREA)
            yield i, i / fps, img
        i += 1
    cap.release()
