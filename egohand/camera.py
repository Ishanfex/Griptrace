"""Head-motion compensation.

In first-person video everything moves when the worker turns their head, so
"did the object move?" must be asked in *scene* coordinates. We track corner
points on the background (hands masked out), fit a similarity transform between
consecutive frames, and chain them. Cost: ~2-4 ms per frame at 320 px wide.
"""
import cv2
import numpy as np

SMALL_W = 320


class CameraMotion:
    def __init__(self, width, height):
        self.s = SMALL_W / width
        self.size = (SMALL_W, int(round(height * self.s)))
        self.prev = None
        self.A = np.eye(3)                # prev frame -> current frame (full-res pixels)
        self.to_scene = [np.eye(3)]       # frame k -> frame 0 coordinates, one per processed frame
        self.ok_frames = 0

    def update(self, bgr, hand_boxes=()):
        g = cv2.cvtColor(cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        A = np.eye(3)
        if self.prev is not None:
            mask = np.full(g.shape, 255, np.uint8)
            for b in hand_boxes:                       # hands move on their own: ignore them
                x1, y1, x2, y2 = (np.array(b) * self.s).astype(int)
                cv2.rectangle(mask, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), 0, -1)
            p0 = cv2.goodFeaturesToTrack(self.prev, 150, 0.01, 8, mask=mask)
            if p0 is not None and len(p0) >= 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None, winSize=(21, 21), maxLevel=3)
                good = st.ravel() == 1
                if good.sum() >= 8:
                    M, inl = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC,
                                                         ransacReprojThreshold=2.0)
                    if M is not None and inl is not None and inl.sum() >= 6:
                        S = np.diag([self.s, self.s, 1.0])
                        A = np.linalg.inv(S) @ np.vstack([M, [0, 0, 1]]) @ S
                        self.ok_frames += 1
            self.to_scene.append(self.to_scene[-1] @ np.linalg.inv(A))
        self.A = A
        self.prev_warped = None if self.prev is None else cv2.warpAffine(
            self.prev, (np.diag([self.s, self.s, 1]) @ A @ np.diag([1 / self.s, 1 / self.s, 1]))[:2],
            (g.shape[1], g.shape[0]))
        self.cur = g
        self.prev = g
        return A

    def motion_in_box(self, box):
        """Mean pixel change inside `box` after removing head motion (0 = still)."""
        if self.prev_warped is None:
            return 0.0
        x1, y1, x2, y2 = (np.array(box) * self.s).astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        a = self.cur[y1:y2, x1:x2]
        b = self.prev_warped[y1:y2, x1:x2]
        if a.size == 0:
            return 0.0
        return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def to_scene(T, pt):
    v = T @ np.array([pt[0], pt[1], 1.0])
    return v[:2] / v[2]
