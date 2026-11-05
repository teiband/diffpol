"""Webcam helpers + intrinsics I/O."""
import json
import sys

import cv2
import numpy as np


def open_camera(index=0, width=1280, height=720, fps=60, manual_exposure=None):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # needed for high fps over USB2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Autofocus changes the intrinsics -> always lock focus.
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if manual_exposure is not None:
        # Short exposure = less motion blur (the #1 problem for webcam tracking).
        # Value semantics are driver specific (DirectShow: log2 seconds, e.g. -7 = 1/128 s).
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, manual_exposure)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index}")
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"camera {index}: {int(w)}x{int(h)} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps (requested {fps})")
    return cap


class Intrinsics:
    """Pinhole (+ radial/tangential, 'pinhole') or Kannala-Brandt ('fisheye') camera model."""

    def __init__(self, K, D, model, size):
        self.K = np.asarray(K, np.float64)
        self.D = np.asarray(D, np.float64).reshape(-1)
        self.model = model
        self.size = tuple(size)

    def undistort_points(self, pts):
        """(N,2) distorted pixels -> (N,2) undistorted pixels (same K, zero distortion)."""
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        if self.model == "fisheye":
            out = cv2.fisheye.undistortPoints(pts, self.K, self.D.reshape(4, 1), P=self.K)
        else:
            out = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
        return out.reshape(-1, 2)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"K": self.K.tolist(), "D": self.D.tolist(), "model": self.model,
                       "size": list(self.size)}, f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            j = json.load(f)
        return Intrinsics(j["K"], j["D"], j["model"], j["size"])
