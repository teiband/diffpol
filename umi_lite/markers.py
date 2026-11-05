"""Marker layout shared by printing, calibration and tracking.

IMPORTANT: after printing, measure the real marker side lengths with a ruler/caliper
and update the values below (printers often rescale by a few %). All lengths in meters.
"""
import cv2

# --- World / table board: fixed on the table, defines the world frame -----------------
WORLD_DICT = cv2.aruco.DICT_4X4_250
WORLD_COLS, WORLD_ROWS = 5, 4          # fits on A4 landscape
WORLD_MARKER_LEN = 0.040               # black square side length
WORLD_MARKER_SEP = 0.010               # gap between markers
# Optional: print several boards and lay them out side by side for a larger workspace.
# Then give each board its own id range and offset in WORLD_BOARD_OFFSETS (not needed for one board).

# --- Gripper finger markers (one per finger, same dictionary as world board) -----------
FINGER_IDS = (100, 101)                # left, right
FINGER_MARKER_LEN = 0.015

# --- ChArUco board for camera intrinsics (different dictionary -> no id clashes) -------
CHARUCO_DICT = cv2.aruco.DICT_5X5_100
CHARUCO_SQUARES = (7, 5)
CHARUCO_SQUARE_LEN = 0.035
CHARUCO_MARKER_LEN = 0.026


def world_dictionary():
    return cv2.aruco.getPredefinedDictionary(WORLD_DICT)


def world_board():
    return cv2.aruco.GridBoard((WORLD_COLS, WORLD_ROWS), WORLD_MARKER_LEN, WORLD_MARKER_SEP,
                               world_dictionary())


def charuco_board():
    d = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    return cv2.aruco.CharucoBoard(CHARUCO_SQUARES, CHARUCO_SQUARE_LEN, CHARUCO_MARKER_LEN, d)


def aruco_detector(dictionary):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)
