"""Episode dataset with UMI's relative pose representation.

For a sample at step t (obs horizon To, prediction horizon Tp):
  obs.rgb    : images at t-To+1 .. t                                  (To,H,W,3) uint8
  obs.lowdim : poses at t-To+1 .. t expressed in the frame of pose t  (To, 9) + gripper (To,1)
  action     : poses at t+1 .. t+Tp expressed in the frame of pose t  (Tp, 9) + gripper (Tp,1)
Because everything is relative to the current gripper pose, the policy never needs the
world frame -> no calibration between the data-collection world and the robot base.
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from umi_lite.geometry import inv_pose, pose_to_vec9, vec9_to_pose

LOWDIM_DIM = 10
ACTION_DIM = 10


def make_lowdim(poses, grips):
    """poses (To,4,4) oldest..current, grips (To,) -> (To,10)"""
    rel = inv_pose(poses[-1])[None] @ poses
    return np.concatenate([pose_to_vec9(rel), grips[:, None]], -1).astype(np.float32)


def make_actions(T_cur, future_poses, future_grips):
    rel = inv_pose(T_cur)[None] @ future_poses
    return np.concatenate([pose_to_vec9(rel), future_grips[:, None]], -1).astype(np.float32)


def actions_to_targets(T_cur, actions):
    """(Tp,10) relative actions -> list of (absolute 4x4 target pose, gripper width)"""
    T_rel = vec9_to_pose(actions[:, :9].astype(np.float64))
    return [(T_cur @ T_rel[k], float(actions[k, 9])) for k in range(len(actions))]


def load_episodes(path):
    files = sorted(glob.glob(os.path.join(path, "episode_*.npz")))
    if not files:
        raise FileNotFoundError(f"no episode_*.npz in {path}")
    eps = []
    for f in files:
        d = np.load(f)
        ep = {k: d[k] for k in d.files}
        ep["pose"] = ep["pose"].astype(np.float64)
        if "gripper_cmd" not in ep:
            ep["gripper_cmd"] = ep["gripper"]
        eps.append(ep)
    return eps


class EpisodeDataset(Dataset):
    def __init__(self, episodes, obs_horizon=2, pred_horizon=16):
        self.eps, self.To, self.Tp = episodes, obs_horizon, pred_horizon
        self.index = [(e, t) for e, ep in enumerate(episodes) for t in range(len(ep["rgb"]))]

    def __len__(self):
        return len(self.index)

    def get_numpy(self, i):
        e, t = self.index[i]
        ep = self.eps[e]
        L = len(ep["rgb"])
        oi = np.clip(np.arange(t - self.To + 1, t + 1), 0, L - 1)   # pad by repeating edges
        ai = np.clip(np.arange(t + 1, t + 1 + self.Tp), 0, L - 1)
        lowdim = make_lowdim(ep["pose"][oi], ep["gripper"][oi])
        action = make_actions(ep["pose"][t], ep["pose"][ai], ep["gripper_cmd"][ai])
        return ep["rgb"][oi], lowdim, action

    def __getitem__(self, i):
        rgb, lowdim, action = self.get_numpy(i)
        return {"rgb": torch.from_numpy(rgb), "lowdim": torch.from_numpy(lowdim),
                "action": torch.from_numpy(action)}

    def stats(self):
        lows, acts = [], []
        for i in range(len(self)):
            _, l, a = self.get_numpy(i)
            lows.append(l)
            acts.append(a)
        lows, acts = np.concatenate(lows), np.concatenate(acts)
        return {"lowdim": (lows.min(0), lows.max(0)), "action": (acts.min(0), acts.max(0))}
