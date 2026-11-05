"""Where large files live: a data store outside the git repo.

Layout of the store (default: a sibling folder of the repo, `<repo>_store`, e.g. D:/workspace/diffpol_store;
override with the environment variable UMI_STORE, e.g. a Google Drive folder on Colab):

    data/        episode_*.npz datasets       runs/      checkpoints, logs, eval videos
    recordings/  raw webcam videos            markers/   generated marker PNGs
    dist/        umi_lite.zip for Colab

Command-line paths that start with one of these folder names are resolved inside the store, so
`--data data/sim_pick --out runs/sim_pick` works unchanged from any machine. Absolute paths and all
other relative paths (e.g. `configs/camera.json`, which is versioned in the repo) are used as given.
"""
import os
from pathlib import Path

STORE_DIRS = ("data", "runs", "recordings", "markers", "dist")
REPO = Path(__file__).resolve().parent.parent


def store_root():
    env = os.environ.get("UMI_STORE")
    return Path(env).expanduser() if env else REPO.parent / f"{REPO.name}_store"


def resolve(p):
    """map 'data/x', 'runs/y', ... into the store; leave everything else untouched"""
    if p is None:
        return None
    q = Path(os.path.expanduser(str(p)))
    if q.is_absolute() or not q.parts or q.parts[0] not in STORE_DIRS:
        return str(p)
    return str(store_root() / q)


def resolve_args(ns, *names):
    """resolve the named argparse attributes in place (str or list of str); returns ns"""
    for n in names:
        v = getattr(ns, n, None)
        setattr(ns, n, [resolve(x) for x in v] if isinstance(v, list) else resolve(v))
    return ns
