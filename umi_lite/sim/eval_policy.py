"""Closed-loop evaluation of a trained policy in the MuJoCo pick-and-place task.

    python -m umi_lite.sim.eval_policy --ckpt runs/sim_pick/last.pt --episodes 50 --video runs/sim_pick/eval
"""
import argparse
import collections
import json
import os

import cv2
import numpy as np
import torch

from umi_lite.data.dataset import actions_to_targets, make_lowdim
from umi_lite.policy import DiffusionPolicy
from umi_lite.sim.env import OPEN, PickPlaceEnv
from umi_lite.paths import resolve_args


def wrap90(a):
    """angle modulo the cube's 90 deg symmetry -> [-45, 45) deg (radians)"""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def hand_yaw(env):
    R = env.hand_pose()[:3, :3]
    return float(np.arctan2(R[1, 0], R[0, 0]))


def smooth_actions(acts, k):
    """centered moving average (window k) over the predicted pose sequence (xyz + rot6d), anchored at the
    current pose (identity) in front; gripper untouched. Removes step-to-step jitter of the sampled chunk."""
    if k <= 1:
        return acts
    ident = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], acts.dtype)
    h = k // 2
    x = np.concatenate([np.repeat(ident[None], h, 0), acts[:, :9], np.repeat(acts[-1:, :9], h, 0)])
    out = acts.copy()
    out[:, :9] = np.stack([x[i:i + k].mean(0) for i in range(len(acts))])
    return out


def run_episode(env, pol, seed, exec_horizon, max_steps, device, video_path=None, smooth=1):
    torch.manual_seed(seed)                      # reproducible sampling noise per episode
    obs = env.reset(seed)
    # grasp diagnostics: yaw offset between finger axis and cube faces (0 = parallel faces, 45 = diagonal)
    info = {"seed": seed, "init_yaw_deg": float(np.degrees(wrap90(hand_yaw(env) - env.cube_yaw()))),
            "grasp_yaw_deg": None, "grasp_dz_mm": None, "grasp_dxy_mm": None, "lifted": False}
    hist = collections.deque([obs] * pol.To, maxlen=pol.To)
    frames, steps = [], 0
    while steps < max_steps:
        rgb = torch.from_numpy(np.stack([o["rgb"] for o in hist]))[None].to(device)
        low = make_lowdim(np.stack([o["pose"] for o in hist]), np.array([o["gripper"] for o in hist]))
        acts = smooth_actions(pol.predict(rgb, torch.from_numpy(low)[None].to(device))[0].cpu().numpy(), smooth)
        for T, g in actions_to_targets(hist[-1]["pose"], acts)[:exec_horizon]:
            obs = env.step(T, g)
            hist.append(obs)
            steps += 1
            hp, cp = obs["pose"][:3, 3], env.cube_pos()
            if info["grasp_yaw_deg"] is None and g < 0.5 * OPEN and hp[2] < 0.08:   # first close command near the table
                info["grasp_yaw_deg"] = float(np.degrees(wrap90(hand_yaw(env) - env.cube_yaw())))
                info["grasp_dz_mm"] = float((hp[2] - cp[2]) * 1000)
                info["grasp_dxy_mm"] = float(np.linalg.norm(hp[:2] - cp[:2]) * 1000)
            info["lifted"] |= bool(cp[2] > 0.05)
            if video_path:
                ov = env.render_overview(320)
                inset = cv2.resize(obs["rgb"], (110, 110), interpolation=cv2.INTER_NEAREST)
                ov[5:115, 5:115] = inset
                frames.append(ov)
            if env.success() or steps >= max_steps:
                break
        if env.success():
            break
    ok = env.success()
    if video_path:
        vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 320))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
    info.update(success=bool(ok), steps=steps)
    return info


def summarize(log):
    """success vs |yaw offset| at grasp and at start -> is the diagonal grasp the failure cause?"""
    print("grasp diagnostics  (|yaw| = angle between finger axis and nearest cube face; 45 deg = diagonal)")
    for key, name in (("grasp_yaw_deg", "at close"), ("init_yaw_deg", "at start")):
        print(f"  |yaw| {name:9s} bin   episodes  success")
        for lo in range(0, 45, 9):
            sel = [e for e in log if e[key] is not None and lo <= abs(e[key]) < lo + 9]
            if sel:
                print(f"      {lo:2d}-{lo + 9:2d} deg      {len(sel):3d}    {np.mean([e['success'] for e in sel]) * 100:5.1f} %")
        nc = sum(e[key] is None for e in log)
        if nc:
            print(f"      never closed     {nc:3d}    {np.mean([e['success'] for e in log if e[key] is None]) * 100:5.1f} %")
    for ok in (True, False):
        g = [abs(e["grasp_yaw_deg"]) for e in log if e["success"] == ok and e["grasp_yaw_deg"] is not None]
        if g:
            print(f"  {'success' if ok else 'fail   '}: median |yaw| at close {np.median(g):4.1f} deg, "
                  f"cube lifted in {np.mean([e['lifted'] for e in log if e['success'] == ok]) * 100:.0f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=10_000, help="different from training seeds")
    ap.add_argument("--res", type=int, default=96, help="must match the training images")
    ap.add_argument("--video", default=None, help="folder for mp4s of the first 5 episodes")
    ap.add_argument("--ddim-steps", type=int, default=None, help="override sampling steps (trained default 10)")
    ap.add_argument("--smooth", type=int, default=1, help="moving-average window over the predicted chunk (1 = off)")
    ap.add_argument("--log", default=None, help="json with per-episode grasp diagnostics (default: next to ckpt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    resolve_args(a, "ckpt", "video", "log")   # data/, runs/, ... -> data store (umi_lite/paths.py)
    pol, ck = DiffusionPolicy.load(a.ckpt, a.device)
    if a.ddim_steps:
        pol.n_infer = a.ddim_steps
    env = PickPlaceEnv(img_res=a.res)
    if a.video:
        os.makedirs(a.video, exist_ok=True)
    log = []
    for i in range(a.episodes):
        vp = os.path.join(a.video, f"ep_{i:03d}.mp4") if a.video and i < 5 else None
        e = run_episode(env, pol, a.seed + i, a.exec_horizon, a.max_steps, a.device, vp, a.smooth)
        log.append(e)
        gy = "  -  " if e["grasp_yaw_deg"] is None else f"{e['grasp_yaw_deg']:+5.1f}"
        print(f"episode {i}: {'SUCCESS' if e['success'] else 'fail'} ({e['steps']} steps)  "
              f"yaw start {e['init_yaw_deg']:+5.1f} close {gy} deg  lifted {int(e['lifted'])}  "
              f"running rate {np.mean([x['success'] for x in log]) * 100:.0f} %")
    log_path = a.log or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "eval_grasp.json")
    json.dump(log, open(log_path, "w"), indent=1)
    summarize(log)
    print(f"per-episode log: {log_path}")
    print(f"success rate: {np.mean([e['success'] for e in log]) * 100:.1f} % over {a.episodes} episodes")


if __name__ == "__main__":
    main()
