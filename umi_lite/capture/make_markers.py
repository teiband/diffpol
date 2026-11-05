"""Generate printable marker sheets (PNG, 300 DPI). Print at 100 % / "actual size",
then measure and update umi_lite/markers.py.

    python -m umi_lite.capture.make_markers --out markers/
"""
import argparse
import os

import cv2
import numpy as np

from umi_lite import markers as M
from umi_lite.paths import resolve_args

DPI = 300
A4_LANDSCAPE = (3508, 2480)  # px at 300 DPI


def m2px(m):
    return int(round(m / 0.0254 * DPI))


def on_a4(img, label):
    W, H = A4_LANDSCAPE
    page = np.full((H, W), 255, np.uint8)
    h, w = img.shape[:2]
    assert w <= W and h <= H, "board does not fit on A4 - reduce sizes in markers.py"
    y0, x0 = (H - h) // 2, (W - w) // 2
    page[y0:y0 + h, x0:x0 + w] = img
    cv2.putText(page, label, (60, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="markers")
    a = ap.parse_args()
    resolve_args(a, "out")   # data/, runs/, ... -> data store (umi_lite/paths.py)
    os.makedirs(a.out, exist_ok=True)

    # world board
    # sizes are built from whole-pixel marker/gap lengths: OpenCV (>=5) asserts if the requested
    # image size does not match the board's aspect ratio exactly
    wb = M.world_board()
    mk, sp = m2px(M.WORLD_MARKER_LEN), m2px(M.WORLD_MARKER_SEP)
    size = (M.WORLD_COLS * mk + (M.WORLD_COLS - 1) * sp, M.WORLD_ROWS * mk + (M.WORLD_ROWS - 1) * sp)
    img = wb.generateImage(size, marginSize=0, borderBits=1)
    cv2.imwrite(os.path.join(a.out, "world_board_A4.png"),
                on_a4(img, f"WORLD board: marker {M.WORLD_MARKER_LEN*1000:.0f} mm, gap "
                           f"{M.WORLD_MARKER_SEP*1000:.0f} mm - print 100%, tape flat on table"))

    # charuco calibration board
    cb = M.charuco_board()
    sx, sy = M.CHARUCO_SQUARES
    sq = m2px(M.CHARUCO_SQUARE_LEN)
    img = cb.generateImage((sx * sq, sy * sq), marginSize=0, borderBits=1)
    cv2.imwrite(os.path.join(a.out, "charuco_calib_A4.png"),
                on_a4(img, f"CHARUCO calibration: square {M.CHARUCO_SQUARE_LEN*1000:.0f} mm - "
                           "glue onto rigid flat board"))

    # finger markers (with a white quiet zone)
    d = M.world_dictionary()
    side = m2px(M.FINGER_MARKER_LEN)
    pad = side // 3
    for mid in M.FINGER_IDS:
        mk = cv2.aruco.generateImageMarker(d, mid, side)
        mk = cv2.copyMakeBorder(mk, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(os.path.join(a.out, f"finger_marker_{mid}.png"), mk)
    print(f"wrote marker sheets to {a.out}/")


if __name__ == "__main__":
    main()
