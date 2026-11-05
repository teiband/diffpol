"""Generate UMI-style demonstrations in simulation (wrist RGB + gripper pose + width).

    python -m umi_lite.sim.collect_demos --n 100 --n-perturb 60 --n-regrasp 40 --out data/sim_pick_v2

Three kinds of episodes (all saved in the same format, only successful ones):
  clean    the scripted expert from start to finish
  perturb  DART-style: during the approach the commanded pose is pushed off (15-40 mm, 10-35 deg); recording
           starts *after* the push, so the episode shows the expert correcting from an off-centre state
           (the push itself is never a training target)
  regrasp  the first grasp is deliberately off-centre (misses); recording starts when the expert notices the
           empty gripper (width ~ 0), so the episode shows open -> lift -> re-align -> grasp -> place
"""
import argparse
import os

import numpy as np

from umi_lite.sim.env import OPEN, PickPlaceEnv
from umi_lite.sim.expert import ScriptedExpert
from umi_lite.paths import resolve_args


def run(env, seed, mode, max_steps):
    """returns (dict of arrays, success) or (None, False) if the episode is unusable for this mode"""
    rng = np.random.default_rng(seed)
    obs = env.reset(seed)
    miss = None
    if mode == "regrasp":   # lateral offset 24-34 mm (fingers land on the cube edge / next to it), slightly high
        ang = rng.uniform(0, 2 * np.pi)
        miss = np.array([*(rng.uniform(0.024, 0.034) * np.array([np.cos(ang), np.sin(ang)])), rng.uniform(0, 0.008)])
    ex = ScriptedExpert(env, rng, miss_offset=miss)
    t_push = int(rng.integers(2, 12)) if mode == "perturb" else -1
    recording = mode == "clean"
    rec = {"rgb": [], "pose": [], "gripper": [], "gripper_cmd": []}
    last_cmd = OPEN
    for t in range(max_steps):
        if t == t_push:
            if ex.phase > 1 or ex.pos[2] < 0.06:      # too close to the cube: a push could knock it over
                return None, False
            ang = rng.uniform(0, 2 * np.pi)
            dxy = rng.uniform(0.015, 0.04) * np.array([np.cos(ang), np.sin(ang)])
            target, g = ex.perturb([*dxy, rng.uniform(-0.01, 0.02)], np.radians(rng.uniform(10, 35)) * rng.choice([-1, 1]))
            for _ in range(2):                        # move there unrecorded, let it settle
                obs = env.step(target, g)
            recording = True
        target, g = ex.act()
        if mode == "regrasp" and ex.missed and not recording:
            recording = True                          # first recorded state: closed on nothing next to the cube
        if recording:
            rec["rgb"].append(obs["rgb"])
            rec["pose"].append(obs["pose"])
            rec["gripper"].append(obs["gripper"])
            rec["gripper_cmd"].append(last_cmd)       # command that was active while reaching this state
        obs = env.step(target, g)
        last_cmd = g
        if ex.done:
            break
    if not recording or len(rec["rgb"]) < 20:
        return None, False
    return rec, env.success()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="clean episodes")
    ap.add_argument("--n-perturb", type=int, default=0, help="DART-style recovery episodes")
    ap.add_argument("--n-regrasp", type=int, default=0, help="missed-grasp -> re-grasp episodes")
    ap.add_argument("--out", default="data/sim_pick_v2")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    resolve_args(a, "out")   # data/, runs/, ... -> data store (umi_lite/paths.py)
    os.makedirs(a.out, exist_ok=True)
    env = PickPlaceEnv(a.res)
    saved, seed = 0, a.seed
    for mode, n in (("clean", a.n), ("perturb", a.n_perturb), ("regrasp", a.n_regrasp)):
        k = tries = 0
        while k < n:
            rec, ok = run(env, seed, mode, a.max_steps)
            seed += 1
            tries += 1
            if rec is None:
                continue
            print(f"{mode} seed {seed - 1}: {len(rec['rgb'])} steps, success={ok}")
            if ok:
                np.savez_compressed(os.path.join(a.out, f"episode_{saved:04d}.npz"),
                                    rgb=np.stack(rec["rgb"]), pose=np.stack(rec["pose"]).astype(np.float32),
                                    gripper=np.array(rec["gripper"], np.float32),
                                    gripper_cmd=np.array(rec["gripper_cmd"], np.float32))
                saved += 1
                k += 1
        print(f"{mode}: {n} episodes saved from {tries} tries")
    assert seed < 10_000, "demo seeds must stay below the eval seeds (10000+)"
    print(f"saved {saved} episodes to {a.out}")


if __name__ == "__main__":
    main()
