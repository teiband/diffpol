"""Camera intrinsic calibration with the ChArUco board.

Live:   python -m umi_lite.capture.calibrate --camera 0 --out configs/camera.json
        (press C to grab a view, ~20-30 views from many angles/distances, covering the
         image corners; press Q to calibrate)
Images: python -m umi_lite.capture.calibrate --images recordings/calib_imgs --out configs/camera.json
Use --fisheye for lenses wider than ~120 deg.
Use the SAME resolution here as for recording.
"""
import argparse
import glob
import os

import cv2
import numpy as np

from umi_lite import markers as M
from umi_lite.capture.camera import Intrinsics, open_camera
from umi_lite.paths import resolve_args


def detect(detector, board, gray):
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 8:
        return None
    obj, img = board.matchImagePoints(ch_corners, ch_ids)
    return obj.astype(np.float64), img.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--images", default=None, help="folder with calibration images instead of live")
    ap.add_argument("--fisheye", action="store_true")
    ap.add_argument("--out", default="configs/camera.json")
    a = ap.parse_args()

    resolve_args(a, "images", "out")   # data/, runs/, ... -> data store (umi_lite/paths.py)
    board = M.charuco_board()
    detector = cv2.aruco.CharucoDetector(board)
    views, size = [], None

    if a.images:
        for p in sorted(glob.glob(os.path.join(a.images, "*.*"))):
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            size = im.shape[::-1]
            r = detect(detector, board, im)
            print(p, "ok" if r else "board not found")
            if r:
                views.append(r)
    else:
        cap = open_camera(a.camera, a.width, a.height, 30)
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            r = detect(detector, board, gray)
            vis = frame.copy()
            if r:
                for p in r[1].reshape(-1, 2):
                    cv2.circle(vis, tuple(int(v) for v in p), 3, (0, 255, 0), -1)
            cv2.putText(vis, f"views: {len(views)}  [C]apture [Q]uit+calibrate", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("calibrate", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("c") and r:
                views.append(r)
            elif k == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    if len(views) < 10:
        raise SystemExit(f"only {len(views)} usable views - need at least 10 (20+ recommended)")
    obj = [v[0].reshape(-1, 1, 3) for v in views]
    img = [v[1].reshape(-1, 1, 2) for v in views]

    if a.fisheye:
        K, D = np.zeros((3, 3)), np.zeros((4, 1))
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, K, D, _, _ = cv2.fisheye.calibrate(obj, img, size, K, D, flags=flags,
                                                criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-6))
        model = "fisheye"
    else:
        rms, K, D, _, _ = cv2.calibrateCamera([o.astype(np.float32) for o in obj],
                                              [i.astype(np.float32) for i in img], size, None, None,
                                              flags=cv2.CALIB_RATIONAL_MODEL)
        model = "pinhole"
    print(f"RMS reprojection error: {rms:.3f} px  (good: < 0.5, acceptable: < 1.0)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Intrinsics(K, D, model, size).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
