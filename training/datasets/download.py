"""
Download the external datasets into data/external/ (gitignored).

    python -m training.datasets.download              # all
    python -m training.datasets.download sparkov      # one

Needs a Kaggle API token (~/.kaggle/access_token or ~/.kaggle/kaggle.json). Competition
datasets also need their rules accepted on kaggle.com once.
"""
import subprocess
import sys
import zipfile

from training.datasets import EXTERNAL_DIR

# name -> (kaggle sub-command, slug, files to fetch; None = whole dataset)
SOURCES = {
    "sparkov": ("datasets", "kartik2112/fraud-detection", None),
}


def _unzip_all(folder):
    for z in folder.glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(folder)
        z.unlink()


def download(name):
    kind, slug, files = SOURCES[name]
    out = EXTERNAL_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    for f in files or [None]:
        if f and (out / f).exists():
            print(f"{name}: {f} already present")
            continue
        cmd = ["kaggle", kind, "download", slug, "-p", str(out)]
        if f:
            cmd += ["-f", f]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)
    _unzip_all(out)
    print(f"{name}: {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        download(name)
