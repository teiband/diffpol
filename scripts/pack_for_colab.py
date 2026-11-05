"""Pack the code for Colab (fallback to git clone): code only, no data/runs/venv.

    python scripts/pack_for_colab.py

Writes <store>/dist/umi_lite.zip (see umi_lite/paths.py). Only needed when the notebook uses
CODE_SOURCE="zip" (no GitHub access from Colab): upload it to My Drive/diffpol_store/dist/.
"""
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from umi_lite.paths import resolve  # noqa: E402

OUT = resolve("dist/umi_lite.zip")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for base, dirs, files in os.walk(os.path.join(ROOT, "umi_lite")):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(base, f)
                    z.write(p, os.path.relpath(p, ROOT))
                    n += 1
        z.write(os.path.join(ROOT, "requirements.txt"), "requirements.txt")
    print(f"wrote {OUT} ({n} python files, {os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
