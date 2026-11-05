"""Train a diffusion policy on episode_*.npz data (real UMI-style or simulated).

    python -m umi_lite.train --data data/sim_pick --out runs/sim_pick
    python -m umi_lite.train --data data/sim_pick --out runs/sim_cpu --encoder small --batch 32  # CPU

Low-compute defaults: 96x96 images (88 crop), obs horizon 2, U-Net (64,128,256) ~ 4M params
+ ResNet-18 11M params, batch 64 -> fits in 2 GB VRAM.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from umi_lite.data.dataset import ACTION_DIM, LOWDIM_DIM, EpisodeDataset, load_episodes
from umi_lite.policy import EMA, DiffusionPolicy
from umi_lite.paths import resolve_args


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def eval_action_error(policy, ds, device, n_batches=4, batch=64):
    """Offline metric: L2 error of predicted vs. demonstrated future positions (first 8 steps)."""
    if len(ds) == 0:
        return float("nan")
    errs = []
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    for k, b in enumerate(loader):
        if k >= n_batches:
            break
        pred = policy.predict(b["rgb"].to(device), b["lowdim"].to(device)).cpu()
        errs.append((pred[:, :8, :3] - b["action"][:, :8, :3]).norm(dim=-1).mean().item())
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--obs-horizon", type=int, default=2)
    ap.add_argument("--pred-horizon", type=int, default=16)
    ap.add_argument("--encoder", choices=["resnet18", "small"], default="resnet18")
    ap.add_argument("--crop", type=int, default=88)
    ap.add_argument("--down-dims", type=int, nargs=3, default=[64, 128, 256])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke tests)")
    ap.add_argument("--save-every", type=int, default=5, help="checkpoint every N epochs")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/last.pt if it exists (Colab disconnects)")
    ap.add_argument("--amp", action="store_true", help="fp16 mixed precision (CUDA only, e.g. T4)")
    a = ap.parse_args()
    resolve_args(a, "data", "out")   # data/, runs/, ... -> data store (umi_lite/paths.py)
    os.makedirs(a.out, exist_ok=True)
    dev = pick_device(a.device)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    use_amp = a.amp and dev.type == "cuda"

    eps = load_episodes(a.data)
    n_val = max(1, int(len(eps) * a.val_frac)) if len(eps) > 1 else 0
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(eps))
    val_eps = [eps[i] for i in perm[:n_val]]
    tr_eps = [eps[i] for i in perm[n_val:]]
    ds = EpisodeDataset(tr_eps, a.obs_horizon, a.pred_horizon)
    ds_val = EpisodeDataset(val_eps, a.obs_horizon, a.pred_horizon)
    print(f"{len(tr_eps)} train / {len(val_eps)} val episodes, {len(ds)} samples, device {dev}")

    policy = DiffusionPolicy(LOWDIM_DIM, ACTION_DIM, a.obs_horizon, a.pred_horizon, a.encoder,
                             a.crop, a.down_dims)
    st = ds.stats()
    policy.norm_low.fit(*st["lowdim"])
    policy.norm_act.fit(*st["action"])
    policy.to(dev)
    n_par = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"policy: {n_par:.1f} M parameters")

    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=0,
                        pin_memory=dev.type == "cuda")
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=1e-6)
    total = a.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * min(s / max(total, 1), 1))))
    ema = EMA(policy)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    ck_path = os.path.join(a.out, "last.pt")

    def save(epoch, step):
        tmp = ck_path + ".tmp"  # write-then-rename: a disconnect mid-save never corrupts last.pt
        torch.save({"cfg": policy.cfg, "model": policy.state_dict(), "ema": ema.model.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "ema_step": ema.step, "epoch": epoch, "step": step, "args": vars(a)}, tmp)
        os.replace(tmp, ck_path)

    step, start_epoch, t0 = 0, 0, time.time()
    if a.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        policy.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        ema.step = ck.get("ema_step", 0)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
        start_epoch, step = ck.get("epoch", -1) + 1, ck.get("step", 0)
        print(f"resumed from {ck_path}: epoch {start_epoch}, step {step}")

    epoch = start_epoch - 1
    for epoch in range(start_epoch, a.epochs):
        policy.train()
        losses = []
        for b in loader:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = policy.loss(b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(policy)
            losses.append(loss.item())
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        rec = {"epoch": epoch, "step": step, "loss": float(np.mean(losses)),
               "lr": sched.get_last_lr()[0], "time_min": (time.time() - t0) / 60}
        if epoch % 10 == 9 or epoch == a.epochs - 1 or a.max_steps:
            rec["val_pos_err_m"] = eval_action_error(ema.model, ds_val, dev)
        if epoch % a.save_every == a.save_every - 1 or epoch == a.epochs - 1:
            save(epoch, step)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if a.max_steps and step >= a.max_steps:
            break
    save(epoch, step)
    print("saved", ck_path)


if __name__ == "__main__":
    main()
