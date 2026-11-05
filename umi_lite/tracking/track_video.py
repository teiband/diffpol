"""Camera-only 6-DoF tracking of the handheld gripper (no IMU).

* Camera pose: PnP against the ArUco world board taped to the table -> metric, drift-free
  pose of the wrist camera in the board ("world") frame, every frame the board is visible.
* Gripper width: 3-D distance between the two finger markers, mapped to meters with a
  one-off gripper calibration.

1) gripper calibration (record a short clip opening/closing the gripper fully):
    python -m umi_lite.tracking.track_video --intrinsics configs/camera.json \
        --calib-gripper recordings/gripper_calib.mp4 --max-width 0.08
2) track all episodes of a session:
    python -m umi_lite.tracking.track_video --intrinsics configs/camera.json \
        --session recordings/session1 [--show]
Writes ep_XXX_traj.npz next to every video.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from umi_lite import markers as M
from umi_lite.capture.camera import Intrinsics
from umi_lite.geometry import inv_pose, make_pose, mat_to_rot6d, rodrigues_to_mat, rot6d_to_mat
from umi_lite.paths import resolve_args


class Tracker:
    def __init__(self, intr: Intrinsics, min_markers=2, max_reproj_px=3.0):
        self.intr = intr
        self.board = M.world_board()
        self.det = M.aruco_detector(M.world_dictionary())
        self.world_ids = set(int(i) for i in self.board.getIds().reshape(-1))
        self.min_markers = min_markers
        self.max_reproj = max_reproj_px
        self.prev_T = None
        L = M.FINGER_MARKER_LEN / 2
        self.finger_obj = np.array([[-L, L, 0], [L, L, 0], [L, -L, 0], [-L, -L, 0]], np.float64)

    def camera_pose(self, corners, ids):
        sel = [k for k, i in enumerate(ids) if int(i) in self.world_ids]
        if len(sel) < self.min_markers:
            return None, len(sel)
        obj, img = self.board.matchImagePoints([corners[k] for k in sel],
                                                 ids[sel].reshape(-1, 1).astype(np.int32))
        img = self.intr.undistort_points(img)
        obj = obj.reshape(-1, 3).astype(np.float64)
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, self.intr.K, None,
                                                    flags=cv2.SOLVEPNP_IPPE)
        if n == 0:
            return None, len(sel)
        # planar PnP has two solutions: take the lower error, break near-ties by temporal continuity
        cands = []
        for r, t, e in zip(rvecs, tvecs, np.asarray(errs).reshape(-1)):
            r, t = cv2.solvePnPRefineLM(obj, img, self.intr.K, None, r, t)
            T_cw = make_pose(t.reshape(3), rodrigues_to_mat(r))
            cands.append((float(e), inv_pose(T_cw)))
        cands.sort(key=lambda c: c[0])
        best = cands[0]
        if len(cands) > 1 and self.prev_T is not None and cands[1][0] < 1.5 * cands[0][0]:
            d = [np.linalg.norm(c[1][:3, :3] - self.prev_T[:3, :3]) for c in cands[:2]]
            best = cands[int(np.argmin(d))]
        if best[0] > self.max_reproj:
            return None, len(sel)
        self.prev_T = best[1]
        return best[1], len(sel)  # T_world_camera

    def finger_distance(self, corners, ids):
        centers = {}
        for k, i in enumerate(ids.reshape(-1)):
            if int(i) in M.FINGER_IDS:
                img = self.intr.undistort_points(corners[k].reshape(4, 2))
                ok, r, t = cv2.solvePnP(self.finger_obj, img, self.intr.K, None,
                                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    centers[int(i)] = t.reshape(3)
        if len(centers) < 2:
            return None
        a, b = (centers[i] for i in M.FINGER_IDS)
        return float(np.linalg.norm(a - b))

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None:
            return None, None, (corners, ids)
        ids = ids.reshape(-1)
        T, _ = self.camera_pose(corners, ids)
        return T, self.finger_distance(corners, ids), (corners, ids)


def fill_gaps(t, pose, width, valid, max_gap):
    """Interpolate short gaps; return (slice of longest usable segment) or None."""
    ok = valid.copy()
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return None
    # split at gaps longer than max_gap
    segs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a - 1 > max_gap:
            segs.append((start, a))
            start = b
    segs.append((start, idx[-1]))
    s0, s1 = max(segs, key=lambda s: s[1] - s[0])
    vi = idx[(idx >= s0) & (idx <= s1)]
    rng = np.arange(s0, s1 + 1)
    pos = np.stack([np.interp(t[rng], t[vi], pose[vi, k, 3]) for k in range(3)], -1)
    r6 = mat_to_rot6d(pose[vi, :3, :3])
    r6 = np.stack([np.interp(t[rng], t[vi], r6[:, k]) for k in range(6)], -1)
    out = np.tile(np.eye(4), (len(rng), 1, 1))
    out[:, :3, :3] = rot6d_to_mat(r6)
    out[:, :3, 3] = pos
    return rng, out, ok


def interp_width(t, w):
    m = ~np.isnan(w)
    if m.sum() == 0:
        return None
    return np.interp(t, t[m], w[m])


def track_file(video, tracker, gcal, show=False, max_gap=6):
    ts_path = video[:-4] + "_ts.npy"
    cap = cv2.VideoCapture(video)
    poses, widths, valid = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        T, d, (corners, ids) = tracker.process(frame)
        valid.append(T is not None)
        poses.append(T if T is not None else np.eye(4))
        if d is None:
            widths.append(np.nan)
        else:
            x = (d - gcal["d_closed"]) / max(gcal["d_open"] - gcal["d_closed"], 1e-6)
            widths.append(float(np.clip(x, 0, 1) * gcal["max_width"]))
        if show:
            vis = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            msg = "pose OK" if T is not None else "NO POSE"
            cv2.putText(vis, f"{msg}  width={widths[-1]:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0) if T is not None else (0, 0, 255), 2)
            cv2.imshow("track", cv2.resize(vis, (960, int(960 * vis.shape[0] / vis.shape[1]))))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                show = False
                cv2.destroyAllWindows()
    cap.release()
    n = len(poses)
    t = np.load(ts_path)[:n] if os.path.exists(ts_path) else np.arange(n) / 30.0
    t = t - t[0]
    pose, valid, width = np.stack(poses), np.array(valid), np.array(widths)
    print(f"{os.path.basename(video)}: {n} frames, pose valid {valid.mean()*100:.0f} %, "
          f"width valid {(~np.isnan(width)).mean()*100:.0f} %")
    r = fill_gaps(t, pose, width, valid, max_gap)
    if r is None:
        print("  -> skipped (no pose)")
        return
    rng, pose_f, _ = r
    if len(rng) < 0.8 * n:
        print(f"  warning: long tracking gap, keeping frames {rng[0]}..{rng[-1]}")
    w = interp_width(t, width)
    if w is None:
        print("  -> skipped (finger markers never seen)")
        return
    np.savez_compressed(video[:-4] + "_traj.npz", t=t[rng], frame_idx=rng, pose=pose_f,
                        gripper=w[rng], valid=valid[rng])


def calib_gripper(video, tracker, max_width, out):
    cap = cv2.VideoCapture(video)
    ds = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, d, _ = tracker.process(frame)
        if d is not None:
            ds.append(d)
    if len(ds) < 10:
        raise SystemExit("finger markers not detected in calibration video")
    g = {"d_closed": float(np.percentile(ds, 2)), "d_open": float(np.percentile(ds, 98)),
         "max_width": max_width}
    with open(out, "w") as f:
        json.dump(g, f, indent=2)
    print("gripper calibration:", g, "->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--session", help="folder with ep_XXX.mp4 files")
    ap.add_argument("--gripper-calib", default="configs/gripper.json")
    ap.add_argument("--calib-gripper", metavar="VIDEO", help="run gripper calibration on VIDEO")
    ap.add_argument("--max-width", type=float, default=0.08, help="max finger opening [m]")
    ap.add_argument("--min-markers", type=int, default=2)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    resolve_args(a, "session", "calib_gripper", "intrinsics", "gripper_calib")   # data/, runs/, ... -> data store (umi_lite/paths.py)
    tracker = Tracker(Intrinsics.load(a.intrinsics), min_markers=a.min_markers)
    if a.calib_gripper:
        calib_gripper(a.calib_gripper, tracker, a.max_width, a.gripper_calib)
        return
    with open(a.gripper_calib) as f:
        gcal = json.load(f)
    for v in sorted(glob.glob(os.path.join(a.session, "ep_*.mp4"))):
        tracker.prev_T = None
        track_file(v, tracker, gcal, a.show)


if __name__ == "__main__":
    main()
