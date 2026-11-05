"""Regenerates notebooks/colab_train.ipynb:  python scripts/make_colab_notebook.py"""
import json, os, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md(r"""
# umi_lite on Colab (T4 GPU)

Trains the diffusion policy on a Colab GPU. The code is developed locally and pushed to GitHub; this
notebook only **clones the code, (re)creates data, trains with resume, and evaluates in MuJoCo**.
Large files live in the data store `My Drive/diffpol_store/` (`data/`, `runs/`; same layout as the local
`diffpol_store` folder next to the repo, see `umi_lite/paths.py`).

1. `Runtime → Change runtime type → T4 GPU`
2. Edit the config cell, then `Runtime → Run all`.
3. If the free-tier session disconnects: reconnect and *Run all* again — training resumes from the
   last checkpoint in the store (saved every `SAVE_EVERY` epochs).

Private GitHub repo: add a fine-grained, read-only token as Colab secret `GITHUB_TOKEN` (key icon in the
left bar, enable notebook access). Works the same from **VS Code** with Google's Colab extension.
""")
code(r"""
# ---------------- config ----------------
CODE_SOURCE = "git"            # "git": clone GIT_URL | "zip": STORE/dist/umi_lite.zip (scripts/pack_for_colab.py)
GIT_URL = "https://github.com/teiband/diffpol.git"
GIT_BRANCH = "main"
DRIVE_STORE = "/content/drive/MyDrive/diffpol_store"   # UMI_STORE on Google Drive

TASK = "sim_pick_v2"             # v2 demos: clean + DART-style perturbed + missed-grasp re-grasp episodes
N_CLEAN, N_PERTURB, N_REGRASP = 100, 60, 40
N_DEMOS = N_CLEAN + N_PERTURB + N_REGRASP
RUN_NAME = "sim_pick_v2_resnet18"

ENCODER = "resnet18"           # T4: resnet18 ; CPU: small
EPOCHS = 300
BATCH = 64
SAVE_EVERY = 5
EVAL_EPISODES = 50
""")
code(r"""
# ---------------- data store: Google Drive, or local disk as fallback ----------------
import os, subprocess
USE_DRIVE = True
try:
    from google.colab import drive
    drive.mount("/content/drive")
    STORE = DRIVE_STORE
except Exception as e:
    # "credential propagation was unsuccessful": consent popup closed/blocked, Drive permission not ticked,
    # different account chosen, VS Code Colab extension, or Workspace admin restriction.
    print(f"Drive not available ({e}).\n-> Fix: re-run this cell in the browser, allow pop-ups, same account, "
          "tick ALL permissions.\n-> Continuing WITHOUT Drive: data/checkpoints live in /content/diffpol_store "
          "and are LOST on disconnect (download them with the last cell).")
    USE_DRIVE = False
    STORE = "/content/diffpol_store"
os.environ["UMI_STORE"] = STORE                       # umi_lite resolves data/..., runs/... inside it
DATA_DIR = f"{STORE}/data/{TASK}"
RUN_DIR = f"{STORE}/runs/{RUN_NAME}"
for d in (DATA_DIR, RUN_DIR, f"{STORE}/dist"):
    os.makedirs(d, exist_ok=True)
print("store:", STORE)
""")
code(r"""
# ---------------- fetch code (fresh copy every session) ----------------
import shutil
CODE = "/content/diffpol"
shutil.rmtree(CODE, ignore_errors=True)
if CODE_SOURCE == "git":
    url, why = GIT_URL, ""
    try:                                              # private repo: token from Colab secrets
        from google.colab import userdata
        url = GIT_URL.replace("https://", f"https://x-access-token:{userdata.get('GITHUB_TOKEN').strip()}@")
        print("using GITHUB_TOKEN secret")
    except Exception as e:                            # no secret / notebook access not granted -> public clone
        why = f"GITHUB_TOKEN not used ({type(e).__name__}: {e})"
        print(why)
    r = subprocess.run(["git", "clone", "--depth", "1", "-b", GIT_BRANCH, url, CODE], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git clone failed. {why or 'Token rejected (expired? no Contents:read on this repo?)'}\n"
                           "Private repo: key icon (left bar) -> Add new secret GITHUB_TOKEN = fine-grained token with "
                           "Contents: read-only on this repo, and switch 'Notebook access' on.\n"
                           + r.stderr.replace(url, GIT_URL))
else:
    os.makedirs(CODE)
    zip_path = f"{STORE}/dist/umi_lite.zip"
    if not os.path.exists(zip_path):
        from google.colab import files
        print("Select umi_lite.zip (scripts/pack_for_colab.py -> diffpol_store/dist/) from your computer:")
        up = files.upload()
        open(zip_path, "wb").write(up[next(iter(up))])
    shutil.unpack_archive(zip_path, CODE)
os.chdir(CODE)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout or "(zip)",
      sorted(os.listdir(os.path.join(CODE, "umi_lite"))))
""")
code(r"""
# ---------------- dependencies (torch/torchvision/opencv are preinstalled on Colab) ----------------
!pip -q install mujoco
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
""")
code(r"""
# ---------------- data: reuse cached episodes from the store, otherwise generate and cache ----------------
import glob
LOCAL_DATA = f"/content/data/{TASK}"                  # local disk is much faster than Drive for training
if len(glob.glob(f"{DATA_DIR}/episode_*.npz")) >= N_DEMOS:
    shutil.copytree(DATA_DIR, LOCAL_DATA, dirs_exist_ok=True)
else:
    !python -m umi_lite.sim.collect_demos --n {N_CLEAN} --n-perturb {N_PERTURB} --n-regrasp {N_REGRASP} --out {LOCAL_DATA} | grep -E "^(clean|perturb|regrasp):|saved"
    shutil.copytree(LOCAL_DATA, DATA_DIR, dirs_exist_ok=True)
print(len(glob.glob(f"{LOCAL_DATA}/episode_*.npz")), "episodes in", LOCAL_DATA)
""")
code(r"""
# ---------------- train (resumes automatically from RUN_DIR/last.pt) ----------------
!python -u -m umi_lite.train --data {LOCAL_DATA} --out {RUN_DIR} --encoder {ENCODER} \
    --batch {BATCH} --epochs {EPOCHS} --save-every {SAVE_EVERY} --amp --resume
""")
code(r"""
# ---------------- closed-loop evaluation in MuJoCo ----------------
!python -m umi_lite.sim.eval_policy --ckpt {RUN_DIR}/last.pt --episodes {EVAL_EPISODES} \
    --ddim-steps 50 --smooth 3 --video {RUN_DIR}/eval_videos | grep -v "^episode"   # summary only; per-episode details: eval_grasp.json next to last.pt
""")
code(r"""
# ---------------- show an evaluation video (re-encode to H.264 so the browser can play it) ----------------
from IPython.display import HTML, display
from base64 import b64encode
VIDEO_EP = 0                                          # first 5 episodes are recorded (ep_000 .. ep_004)
src = f"{RUN_DIR}/eval_videos/ep_{VIDEO_EP:03d}.mp4"
if not os.path.exists(src):
    print(f"no video at {src} - did the evaluation cell above finish? (its output shows errors, if any)")
else:
    !ffmpeg -y -loglevel error -i {src} -vcodec libx264 -pix_fmt yuv420p /content/ep_h264.mp4
    display(HTML(f'<video width=480 controls><source src="data:video/mp4;base64,{b64encode(open("/content/ep_h264.mp4","rb").read()).decode()}" type="video/mp4"></video>'))
""")
code(r"""
# ---------------- download the run (checkpoint + logs) - essential when running without Drive ----------------
shutil.make_archive(f"/content/{RUN_NAME}", "zip", RUN_DIR)
from google.colab import files
files.download(f"/content/{RUN_NAME}.zip")
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks", "colab_train.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
print("wrote", os.path.normpath(out))
