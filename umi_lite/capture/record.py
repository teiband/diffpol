"""Record demonstration episodes with the gripper-mounted webcam.

    python -m umi_lite.capture.record --camera 0 --out recordings/session1 --exposure -7

SPACE = start/stop an episode, Q = quit. Each episode -> ep_XXX.mp4 + ep_XXX_ts.npy (host
timestamps per frame; webcams drop frames, so never assume a constant frame rate).

Demo tips (from UMI): start every episode with the world board clearly visible, move
smoothly (motion blur kills marker detection), keep >=2 board markers in view, keep the
finger markers visible, vary object positions between episodes, 50-200 episodes per task.
"""
import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np

from umi_lite.capture.camera import open_camera
from umi_lite.paths import resolve_args


class EpisodeWriter(threading.Thread):
    def __init__(self, path, fps, size):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=600)
        self.vw = cv2.VideoWriter(path + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.ts_path = path + "_ts.npy"
        self.ts = []
        self.start()

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            t, frame = item
            self.vw.write(frame)
            self.ts.append(t)
        self.vw.release()
        np.save(self.ts_path, np.asarray(self.ts))

    def close(self):
        self.q.put(None)
        self.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--exposure", type=float, default=None, help="manual exposure (driver units)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    resolve_args(a, "out")   # data/, runs/, ... -> data store (umi_lite/paths.py)
    os.makedirs(a.out, exist_ok=True)
    cap = open_camera(a.camera, a.width, a.height, a.fps, a.exposure)

    ep = len([f for f in os.listdir(a.out) if f.endswith(".mp4")])
    writer, t_last, n = None, time.perf_counter(), 0
    while True:
        ok, frame = cap.read()
        t = time.perf_counter()
        if not ok:
            continue
        if writer is not None:
            try:
                writer.q.put_nowait((t, frame))
            except queue.Full:
                print("warning: disk writer too slow, frame dropped")
        n += 1
        if n % 10 == 0:
            fps = 10 / (t - t_last)
            t_last = t
        else:
            fps = None
        vis = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
        status = f"REC ep_{ep:03d}" if writer else "idle - SPACE to record"
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if writer else (0, 200, 0), 2)
        if fps:
            cv2.setWindowTitle("record", f"record  {fps:.0f} fps")
        cv2.imshow("record", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            if writer is None:
                h, w = frame.shape[:2]
                writer = EpisodeWriter(os.path.join(a.out, f"ep_{ep:03d}"), a.fps, (w, h))
            else:
                writer.close()
                print(f"saved ep_{ep:03d}")
                writer, ep = None, ep + 1
        elif k == ord("q"):
            break
    if writer is not None:
        writer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
