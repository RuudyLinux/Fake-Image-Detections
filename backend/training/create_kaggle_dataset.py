"""
Create + upload the FakeGuard dataset to your Kaggle account.

Usage (after setting KAGGLE_API_TOKEN env var or ~/.kaggle/access_token):
    python training/create_kaggle_dataset.py \\
        --username your_kaggle_username \\
        --data    ../data \\
        --create          # first time: creates new dataset
        # or
        --update          # subsequent: pushes new version

Flow:
    1. Download 140k Real and Fake Faces → real/ + ai_generated/
    2. Download CASIA v2               → edited/
    3. Zip and upload to Kaggle as:
       {username}/fake-image-detection-dataset
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import random
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DATASET_SLUG = "fake-image-detection-dataset"
METADATA_TEMPLATE = Path(__file__).parent.parent.parent / "kaggle_dataset" / "dataset-metadata.json"


# ─── helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=False)


def copy_imgs(src: Path, dst: Path, limit: int | None = None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    files = [f for f in src.rglob("*") if f.suffix.lower() in EXTS]
    if limit:
        files = random.sample(files, min(limit, len(files)))
    for f in files:
        shutil.copy2(f, dst / f.name)
    print(f"    {dst.relative_to(dst.parent.parent.parent)}: {len(files):,} images")
    return len(files)


# ─── download sources ──────────────────────────────────────────────────────────

def download_140k(tmp: Path, data: Path, max_per: int):
    print("\n[1/3] Downloading 140k Real and Fake Faces...")
    dl = tmp / "dl_140k"
    dl.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "datasets", "download",
          "-d", "xhlulu/140k-real-and-fake-faces",
          "-p", str(dl), "--unzip"], check=False)

    base = dl / "real_vs_fake"
    # Kaggle folder name varies by download time
    inner = base / "real-vs-fake" if (base / "real-vs-fake").exists() else base / "real-or-fake"
    base = inner
    for src_sp, dst_sp in [("train", "train"), ("test", "val")]:
        for src_cls, dst_cls in [("real", "real"), ("fake", "ai_generated")]:
            src = base / src_sp / src_cls
            if src.exists():
                copy_imgs(src, data / dst_sp / dst_cls, max_per)
            else:
                print(f"    WARNING: {src} not found")


def download_casia(tmp: Path, data: Path, max_per: int):
    print("\n[2/3] Downloading CASIA v2 (edited images)...")
    dl = tmp / "dl_casia"
    dl.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "datasets", "download",
          "-d", "sophatvathana/casia-dataset",
          "-p", str(dl), "--unzip"], check=False)

    tampered = []
    for name in ["Tp", "tampered", "CASIA2", "fake"]:
        for d in dl.rglob(name):
            if d.is_dir():
                tampered += [f for f in d.rglob("*") if f.suffix.lower() in EXTS]

    if tampered:
        random.shuffle(tampered)
        sp = int(len(tampered) * 0.8)
        for split, files in [("train", tampered[:sp]), ("val", tampered[sp:])]:
            dst = data / split / "edited"
            dst.mkdir(parents=True, exist_ok=True)
            chosen = files[:max_per]
            for f in chosen:
                shutil.copy2(f, dst / f.name)
            print(f"    {split}/edited: {len(chosen):,}")
    else:
        print("    CASIA Tp not found — using synthetic fallback (splice)")
        _make_synthetic_edited(data, max_per)


def _make_synthetic_edited(data: Path, n: int):
    from PIL import Image
    import numpy as np
    real_imgs = list((data / "train" / "real").glob("*.jpg"))[:n * 2]
    random.shuffle(real_imgs)
    sp = int(len(real_imgs) * 0.8)
    for split, files in [("train", real_imgs[:sp]), ("val", real_imgs[sp:])]:
        dst = data / split / "edited"
        dst.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(files) - 1, 2):
            a = np.array(Image.open(files[i]).convert("RGB").resize((256, 256)))
            b = np.array(Image.open(files[i + 1]).convert("RGB").resize((256, 256)))
            spliced = a.copy()
            spliced[:, 128:] = b[:, 128:]
            Image.fromarray(spliced).save(dst / f"splice_{i:05d}.jpg", quality=85)
        print(f"    {split}/edited: {len(files)//2} synthetic")


# ─── verify ────────────────────────────────────────────────────────────────────

def verify(data: Path):
    print("\n─── Dataset Summary ──────────────────────────────────────────")
    total = 0
    for split in ["train", "val"]:
        for cls in ["real", "ai_generated", "edited"]:
            d = data / split / cls
            n = len([f for f in d.rglob("*") if f.suffix.lower() in EXTS]) if d.exists() else 0
            s = "✓" if n >= 500 else ("⚠" if n > 0 else "✗")
            print(f"  {s}  {split:5}/{cls:14}: {n:6,}")
            total += n
    print(f"\n  Total: {total:,} images")
    print("─────────────────────────────────────────────────────────────")
    return total


# ─── kaggle upload ─────────────────────────────────────────────────────────────

def _write_metadata(upload_dir: Path, username: str):
    """Write dataset-metadata.json with correct username."""
    meta = {
        "title": "Fake Image Detection Dataset",
        "id": f"{username}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            "3-class fake image detection dataset: REAL | AI_GENERATED | EDITED.\n"
            "Structure: train/ + val/ each with real/, ai_generated/, edited/ subfolders.\n"
            "Sources: Flickr-Faces (real), StyleGAN/SD (AI), CASIA v2 (edited).\n"
            "Used by FakeGuard AI: https://github.com/RuudyLinux/Fake-Image-Detections"
        ),
    }
    meta_path = upload_dir / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata written: {meta_path}")


def create_dataset(data: Path, username: str):
    """First-time dataset creation."""
    print(f"\n[3/3] Creating Kaggle dataset: {username}/{DATASET_SLUG}")
    upload_dir = data.parent / "_kaggle_upload"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    shutil.copytree(data, upload_dir)
    _write_metadata(upload_dir, username)

    _run(["kaggle", "datasets", "create", "-p", str(upload_dir),
          "--dir-mode", "zip"])
    print(f"\n  Dataset live at: https://www.kaggle.com/datasets/{username}/{DATASET_SLUG}")
    shutil.rmtree(upload_dir)


def update_dataset(data: Path, username: str, message: str = "Updated dataset"):
    """Push a new version to existing dataset."""
    print(f"\n[3/3] Updating Kaggle dataset: {username}/{DATASET_SLUG}")
    upload_dir = data.parent / "_kaggle_upload"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    shutil.copytree(data, upload_dir)
    _write_metadata(upload_dir, username)

    _run(["kaggle", "datasets", "version", "-p", str(upload_dir),
          "-m", message, "--dir-mode", "zip"])
    print(f"\n  Updated: https://www.kaggle.com/datasets/{username}/{DATASET_SLUG}")
    shutil.rmtree(upload_dir)


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True,  help="Your Kaggle username (lowercase)")
    parser.add_argument("--data",     default="../data", help="Local data directory")
    parser.add_argument("--max",      type=int, default=8000, help="Max images per class per split")
    parser.add_argument("--create",   action="store_true", help="Create new dataset (first time)")
    parser.add_argument("--update",   action="store_true", help="Push new version to existing dataset")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading sources (use existing data/)")
    parser.add_argument("--message",  default="Updated dataset", help="Version message for --update")
    args = parser.parse_args()

    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    tmp  = data.parent / "_tmp_dl"
    tmp.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_140k(tmp, data, args.max)
        download_casia(tmp, data, args.max)
        shutil.rmtree(tmp, ignore_errors=True)

    total = verify(data)
    if total == 0:
        print("ERROR: No images found. Run without --skip-download or populate data/ manually.")
        sys.exit(1)

    if args.create:
        create_dataset(data, args.username)
    elif args.update:
        update_dataset(data, args.username, args.message)
    else:
        print("\nDataset organized locally. Use --create to upload or --update to push new version.")


if __name__ == "__main__":
    main()
