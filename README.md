# umi_lite — UMI-style data collection + diffusion policy training on a budget

A small framework that follows the **Universal Manipulation Interface (UMI)** recipe
(Chi et al., RSS 2024): demonstrations are recorded with a **handheld gripper carrying a
wrist camera**, the gripper's 6-DoF trajectory and finger width are recovered from the video,
and a **Diffusion Policy** learns `wrist image + relative proprioception → future relative
end-effector poses + gripper width`. A policy trained this way runs on any arm that carries
the same gripper/camera, because no robot is involved in data collection.

Differences from the original UMI:

| | Original UMI | umi_lite |
|---|---|---|
| Camera | GoPro + 155° fisheye, built-in IMU | any USB webcam (wide-angle recommended) |
| Pose tracking | ORB-SLAM3 monocular-**inertial** SLAM + map | **ArUco world board on the table + PnP** (camera only) |
| Gripper width | ArUco markers on fingers | same |
| Policy | CNN Diffusion Policy with CLIP ViT-B/16 encoder, multi-GPU | CNN Diffusion Policy with ResNet-18 (GroupNorm) or tiny CNN, 2 GB GPU / CPU |
| Without robot | – | MuJoCo sim with a floating UMI gripper for closed-loop testing |

```
umi_lite/
  markers.py              marker layout (edit after measuring your prints)
  geometry.py             pose math, 6-D rotations, relative poses
  capture/make_markers.py printable world board, finger markers, ChArUco board
  capture/calibrate.py    camera intrinsics (pinhole or fisheye)
  capture/record.py       record episodes from the webcam (+ per-frame timestamps)
  tracking/track_video.py camera pose (PnP on world board) + gripper width (finger markers)
  data/build_dataset.py   resample to 10 Hz, 96x96 images -> episode_XXXX.npz
  data/dataset.py         UMI relative-pose obs/action representation
  policy.py               diffusion policy (encoder, 1-D U-Net, DDPM/DDIM, EMA)
  train.py                training
  eval_offline.py         open-loop error on held-out real demos
  sim/env.py              MuJoCo pick-and-place with floating gripper + wrist camera
  sim/expert.py           scripted demonstrator
  sim/collect_demos.py    generate sim demos in the same format as real data
  sim/eval_policy.py      closed-loop success rate + videos
docs/HARDWARE.md          shopping list
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

For an NVIDIA GPU install the matching PyTorch build from pytorch.org first. Note: older
Pascal GPUs (e.g. GeForce MX250, compute capability 6.1) are dropped from the newest CUDA
builds of PyTorch; use a build that still lists sm_61 (e.g. a CUDA 11.8/12.6 wheel) or train
on CPU with `--encoder small`.

## Repository vs. data store

The git repo holds only code, docs, the notebook and small configs. Everything large lives in a
**data store** outside the repo, by default the sibling folder `<repo>_store`:

```
workspace/
├── diffpol/            git repo (umi_lite/, scripts/, notebooks/, docs/, configs/)
└── diffpol_store/      not versioned; override with the environment variable UMI_STORE
    ├── data/           episode_*.npz datasets
    ├── runs/           checkpoints, logs, eval videos/JSON
    ├── recordings/     raw webcam videos (+ calibration images)
    ├── markers/        generated marker PNGs
    └── dist/           umi_lite.zip for Colab
```

Command-line paths that start with `data/`, `runs/`, `recordings/`, `markers/` or `dist/` are
resolved inside the store (`umi_lite/paths.py`), so all commands below work unchanged on any machine.
Absolute paths and other relative paths (e.g. `configs/camera.json`, versioned in the repo) are used
as given. On Colab the store is `My Drive/diffpol_store`.

## Quick start 1 — full pipeline in simulation (no hardware)

```bash
python -m umi_lite.sim.collect_demos --n 100 --n-perturb 60 --n-regrasp 40 --out data/sim_pick_v2
python -m umi_lite.train --data data/sim_pick_v2 --out runs/sim_pick_v2 --epochs 100
python -m umi_lite.sim.eval_policy --ckpt runs/sim_pick_v2/last.pt --episodes 50 --ddim-steps 50 --smooth 3 --video runs/sim_pick_v2/eval
# visual robustness: same seeds on other floors (presets: plain dark light blue fine coarse noise wood, or an image file)
python -m umi_lite.sim.eval_policy --ckpt runs/sim_pick_v2/last.pt --episodes 50 --ddim-steps 50 --smooth 3 --floor default plain dark wood my_table.jpg
```

`--n-perturb` adds DART-style recovery episodes (pushed off course during the approach), `--n-regrasp`
adds missed-grasp → re-grasp episodes. `--ddim-steps 50 --smooth 3` removes jitter from the sampled
action chunks at evaluation time.

CPU only: add `--encoder small --batch 32` to `train`.

## Training on Google Colab (free tier, T4 GPU)

Develop locally, push to GitHub, train remotely: [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb)
clones the code, generates or loads the demos, trains with `--amp --resume`, saves checkpoints to the
data store on Drive (`My Drive/diffpol_store/runs/`), then evaluates in MuJoCo (headless EGL rendering).

Two ways to work:

* **VS Code + Google's Colab extension** (closest to "local development, remote GPU"): open the
  notebook in VS Code → *Select Kernel* → *Colab* → new server with a T4. Notebook cells run on Colab.
  The remote kernel does **not** see your local files, so the code reaches it through git
  (the first cells of the notebook).
* **Browser**: upload/open the notebook on colab.research.google.com.

Getting the code there (pick one):

* **git (default):** `git push`, then *Run all*; the notebook clones `GIT_URL`. For a private repo add a
  fine-grained, read-only GitHub token as Colab secret `GITHUB_TOKEN` (key icon in the left bar).
* **Zip upload (fallback):** `python scripts/pack_for_colab.py` writes `diffpol_store/dist/umi_lite.zip`;
  upload it to `My Drive/diffpol_store/dist/` and set `CODE_SOURCE = "zip"`.

Then open `notebooks/colab_train.ipynb` in Colab (*File → Upload notebook*), select
*Runtime → Change runtime type → T4 GPU*, and *Run all*. The notebook is generated by
`scripts/make_colab_notebook.py`; edit that script rather than the .ipynb.

If mounting Drive fails with *"credential propagation was unsuccessful"*: re-run the cell in the
browser, allow pop-ups, choose the same Google account and tick **all** permissions in the consent
screen (Drive access is a separate checkbox). It also happens in the VS Code Colab extension and
with Workspace accounts whose admin blocks Drive. The notebook then continues **without Drive**:
the store becomes `/content/diffpol_store` (lost on disconnect, so no resume across sessions), and the
last cell downloads the run as a zip.

Free-tier facts that shape the design: GPUs are not guaranteed, sessions end after at most
~12 h and earlier when idle, so checkpoints go to Drive every few epochs and `--resume`
continues where it stopped. SSH/remote-shell tricks into Colab are disallowed on the free tier.

**T4 vs. TPU v5e-1: use the T4.** This code is PyTorch/CUDA. On a TPU it would need PyTorch/XLA
(device handling, graph compilation that recompiles on shape changes, no CUDA-only paths), and
for a ~20 M-parameter model with small batches the TPU gives no speed advantage. The T4
(16 GB, fp16 tensor cores via `--amp`) easily handles the ResNet-18 encoder.

## Quick start 2 — real data with a webcam

```bash
# 0. print markers, measure them, update umi_lite/markers.py
python -m umi_lite.capture.make_markers --out markers
# 1. intrinsics (same resolution as recording!)  add --fisheye for lenses > ~120 deg
python -m umi_lite.capture.calibrate --camera 0 --width 1280 --height 720 --out configs/camera.json
# 2. gripper calibration: record a clip opening/closing the gripper fully, then
python -m umi_lite.capture.record --camera 0 --out recordings/gripper_calib
python -m umi_lite.tracking.track_video --intrinsics configs/camera.json --calib-gripper recordings/gripper_calib/ep_000.mp4 --max-width 0.08
# 3. record demonstrations (SPACE start/stop, Q quit)
python -m umi_lite.capture.record --camera 0 --out recordings/session1 --exposure -7
# 4. track, build dataset, train, evaluate offline
python -m umi_lite.tracking.track_video --intrinsics configs/camera.json --session recordings/session1 --show
python -m umi_lite.data.build_dataset --sessions recordings/session1 --out data/real_task
python -m umi_lite.train --data data/real_task --out runs/real_task
python -m umi_lite.eval_offline --ckpt runs/real_task/last.pt --data data/real_task_holdout
```

---

## Is an IMU needed, or can it be purely camera-based?

**For this framework: no IMU is needed.** In the original UMI the IMU is there because the
pose comes from *monocular SLAM*, and monocular vision alone has three problems the IMU fixes:

1. **Metric scale** — a single camera cannot tell a 10 cm move from a 1 m move in an unknown
   scene. The accelerometer provides metric scale.
2. **Robustness to fast motion / motion blur** — the gyro keeps tracking alive for the
   fraction of a second when the image is blurry.
3. **Gravity direction** — absolute "down" for consistent orientation.

umi_lite gets the same things from the scene instead: an **ArUco board of known size** fixed
on the table. Every frame where ≥ 2 board markers are visible gives a *metric, absolute,
drift-free* 6-DoF camera pose by PnP. Scale comes from the printed marker size, "down" from
the board normal, and there is no drift because there is no integration over time.

The trade-offs of camera-only marker tracking:

* The board (or several boards) must be **in view in every frame** → the workspace is
  limited to the table area around the board, and you should not look away from it.
* **Motion blur** is now fatal (no gyro to bridge it): move smoothly, use bright light and a
  short manual exposure, prefer a global-shutter camera. Short dropouts (≤ 6 frames) are
  interpolated; longer ones split the episode.
* Accuracy is a few mm / < 1° at 20–50 cm with a calibrated 720p camera — sufficient for
  diffusion policies (UMI's SLAM is in the same range).

**When you would want an IMU:** markerless tracking in arbitrary scenes (e.g. ORB-SLAM3
mono-inertial, as in UMI), large workspaces, or very fast/dynamic motions. Then you also need
camera-IMU **time synchronisation and extrinsic calibration** (e.g. Kalibr), which is the
main reason UMI uses a GoPro (camera + IMU on one clock). With a webcam + separate IMU board
this is doable but is the hardest part to get right — so start camera-only.

## Can the policy be evaluated in simulation without a robot arm?

**Yes, with an important caveat about *what* you can evaluate.**

1. **Closed-loop evaluation of the method (included).** `umi_lite/sim` contains a MuJoCo
   scene with a *floating UMI gripper*: two fingers + wrist camera, driven by a mocap body.
   This is exactly the UMI abstraction — the policy outputs end-effector poses, whatever arm
   carries the gripper just has to follow them — so no arm model is needed. A scripted
   demonstrator plays the role of the human, demos are stored in the same `episode_*.npz`
   format as real data, and `eval_policy.py` measures the closed-loop success rate and writes
   videos. This validates the whole representation/training/inference stack, hyperparameters
   and amount of data, all on your laptop.

2. **A policy trained on *real* webcam data cannot be meaningfully run closed-loop in this
   simulator**, because the rendered wrist images look nothing like your real camera
   (visual sim-to-real gap) — the policy would be out of distribution. For real-data
   policies, use:
   * `eval_offline.py`: open-loop prediction error on held-out real demonstrations (necessary,
     not sufficient — compounding errors are invisible open-loop);
   * real-to-sim reconstruction (photogrammetry / Gaussian-splat digital twin of your table)
     if you want closed-loop testing with realistic images — a research-level effort;
   * kinematic replay: send the recorded/predicted end-effector trajectories to a simulated arm
     (e.g. Franka/UR5e models from MuJoCo Menagerie + an IK solver) to check reachability and
     joint limits before you ever touch a real robot.

3. **Recommended path:** develop and debug in sim → collect real data with the webcam
   gripper → check offline error → deploy on a (borrowed / low-cost DIY, e.g. SO-101) arm.

## Which diffusion-policy framework for low compute?

**Recommendation: start with this repo's compact implementation, then move to
[LeRobot](https://github.com/huggingface/lerobot)'s `DiffusionPolicy` when you need more.**

* **umi_lite (this repo)** — CNN-based Diffusion Policy (the variant UMI uses), ~20 M
  parameters with ResNet-18, ~5 M with `--encoder small`. 96×96 images, 10 DDIM inference
  steps, batch 64 fits in **2 GB VRAM**. ~100 demos × ~100 steps trains in roughly an hour on
  an entry-level GPU; CPU training with the small encoder takes several hours. Everything
  lives in ~300 lines (`policy.py`), easy to read and modify.
* **LeRobot (Hugging Face)** — maintained, well-tested Diffusion Policy (same architecture,
  ResNet-18 + 1-D U-Net), dataset tooling, sim benchmarks (PushT, ALOHA), and alternatives
  that are even cheaper to train (ACT). Runs on a single consumer GPU. Good next step.
* **Original `real-stanford/diffusion_policy` / `universal_manipulation_interface`** — the
  reference, but UMI's config uses a CLIP ViT-B/16 encoder at 224×224 and was trained on
  multi-GPU servers. Not suitable for low compute without changing the config.

Levers for even less compute (most effective first):

1. Lower image resolution (96 → 64) and `--encoder small`.
2. Freeze a pretrained encoder and **pre-compute image features once**; then only the U-Net
   trains (minutes instead of hours).
3. Fewer DDIM inference steps (10 → 4–5) — affects inference speed only.
4. Smaller U-Net (`--down-dims 32 64 128`).
5. Faster samplers (flow matching / consistency distillation) for 1–2-step inference.

## Practical notes for deployment on a real arm later

* Actions are poses of the **camera frame** relative to the current camera pose. On the robot,
  convert with the fixed camera→TCP (or camera→flange) transform of the mount, and mount the
  camera at the same place relative to the fingers as on the handheld gripper.
* Measured finger width is used as the gripper action for real data. Command slightly less
  than predicted when grasping so the robot gripper actually squeezes.
* Observation/command latency matters on real hardware (UMI compensates for it explicitly);
  start by executing 4–8 of the 16 predicted steps per inference.
