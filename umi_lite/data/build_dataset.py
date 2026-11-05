"""Turn tracked recordings into training episodes (same format as the simulator writes).

    python -m umi_lite.data.build_dataset --sessions recordings/session1 recordings/session2 \
        --out data/real_cup --hz 10 --res 96

Episode file format (episode_XXXX.npz):
    rgb      (T,H,W,3) uint8   wrist camera image, RGB
    pose     (T,4,4)   float32 camera/gripper pose in a fixed world frame
    gripper  (T,)      float32 measured finger opening [m]
    gripper_cmd (T,)   optional commanded opening (sim only; otherwise 'gripper' is the action)
"""
import argparse
import glob
import os

import cv2
import numpy as np
from umi_lite.paths import resolve_args


def square_resize(frame_bgr, res):
    h, w = frame_bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    img = cv2.resize(frame_bgr[y0:y0 + s, x0:x0 + s], (res, res), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_episode(video, traj, hz, res):
    t, fidx, pose, grip = traj["t"], traj["frame_idx"], traj["pose"], traj["gripper"]
    grid = np.arange(t[0], t[-1], 1.0 / hz)
    pick = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)  # nearest-after sample
    want = {int(fidx[p]): k for k, p in enumerate(pick)}
    frames = [None] * len(pick)
    cap = cv2.VideoCapture(video)
    i = 0
    last = max(want)
    while i <= last:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            img = square_resize(f, res)
            for k, p in enumerate(pick):  # several grid points may map to one frame
                if int(fidx[p]) == i:
                    frames[k] = img
        i += 1
    cap.release()
    keep = [k for k, f in enumerate(frames) if f is not None]
    return dict(rgb=np.stack([frames[k] for k in keep]),
                pose=pose[pick[keep]].astype(np.float32),
                gripper=grip[pick[keep]].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=10.0, help="policy control frequency")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--min-len", type=int, default=20)
    a = ap.parse_args()
    resolve_args(a, "sessions", "out")   # data/, runs/, ... -> data store (umi_lite/paths.py)
    os.makedirs(a.out, exist_ok=True)
    n = len(glob.glob(os.path.join(a.out, "episode_*.npz")))
    for s in a.sessions:
        for tr in sorted(glob.glob(os.path.join(s, "ep_*_traj.npz"))):
            video = tr.replace("_traj.npz", ".mp4")
            ep = build_episode(video, dict(np.load(tr)), a.hz, a.res)
            if len(ep["rgb"]) < a.min_len:
                print("skip (too short)", tr)
                continue
            np.savez_compressed(os.path.join(a.out, f"episode_{n:04d}.npz"), **ep)
            print(f"episode_{n:04d}: {len(ep['rgb'])} steps  <- {video}")
            n += 1


if __name__ == "__main__":
    main()
