"""Offline evaluation on recorded (e.g. held-out real) demonstrations - no robot needed.

    python -m umi_lite.eval_offline --ckpt runs/real_cup/last.pt --data data/real_cup_test

Reports open-loop prediction errors. Low error is necessary but NOT sufficient for a
working closed-loop policy (compounding errors are invisible here) - see README.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from umi_lite.data.dataset import EpisodeDataset, load_episodes
from umi_lite.geometry import rot6d_to_mat
from umi_lite.policy import DiffusionPolicy
from umi_lite.paths import resolve_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    resolve_args(a, "ckpt", "data")   # data/, runs/, ... -> data store (umi_lite/paths.py)
    pol, _ = DiffusionPolicy.load(a.ckpt, a.device)
    ds = EpisodeDataset(load_episodes(a.data), pol.To, pol.Tp)
    H = a.exec_horizon
    pos, rot, grip = [], [], []
    for b in DataLoader(ds, batch_size=64):
        p = pol.predict(b["rgb"].to(a.device), b["lowdim"].to(a.device)).cpu().numpy()[:, :H]
        g = b["action"].numpy()[:, :H]
        pos.append(np.linalg.norm(p[..., :3] - g[..., :3], axis=-1).reshape(-1))
        Rp, Rg = rot6d_to_mat(p[..., 3:9]), rot6d_to_mat(g[..., 3:9])
        cos = (np.trace(np.swapaxes(Rp, -1, -2) @ Rg, axis1=-2, axis2=-1) - 1) / 2
        rot.append(np.degrees(np.arccos(np.clip(cos, -1, 1))).reshape(-1))
        grip.append(np.abs(p[..., 9] - g[..., 9]).reshape(-1))
    pos, rot, grip = map(np.concatenate, (pos, rot, grip))
    print(f"samples: {len(ds)}  (first {H} predicted steps)")
    print(f"position error : mean {pos.mean()*1000:.1f} mm, median {np.median(pos)*1000:.1f} mm")
    print(f"rotation error : mean {rot.mean():.2f} deg")
    print(f"gripper error  : mean {grip.mean()*1000:.1f} mm")


if __name__ == "__main__":
    main()
