"""Low-light fix: boost local contrast only when a frame is dark.

CLAHE on the lightness channel brightens shadows without blowing out the
harsh highlights common in factories (a plain gamma curve would).
"""
import cv2
import numpy as np

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def mean_luma(bgr):
    small = cv2.resize(bgr, (64, 36), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def enhance(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    # lift very dark images a bit more with a gamma curve
    if l.mean() < 90:
        lut = (np.linspace(0, 1, 256) ** 0.7 * 255).astype(np.uint8)
        l = cv2.LUT(l, lut)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
