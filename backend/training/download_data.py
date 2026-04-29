"""
Dataset acquisition helper for 3-class fake image detection.
  Classes: real | ai_generated | edited

─── Quick Start ─────────────────────────────────────────────────────────────
  pip install kaggle
  # Put kaggle.json in ~/.kaggle/
  python training/download_data.py --source all --output ../data

─── Source Overview ─────────────────────────────────────────────────────────
  REAL        → 140k dataset (Flickr faces) or FFHQ
  AI_GENERATED→ 140k dataset (StyleGAN faces) + optional Stable Diffusion
  EDITED      → CASIA v2 (splicing + copy-move) from Kaggle
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import subprocess
import shutil
import random
import zipfile
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _copy_images(src: Path, dst: Path, limit: int | None = None):
    dst.mkdir(parents=True, exist_ok=True)
    files = [f for f in src.rglob("*") if f.suffix.lower() in EXTENSIONS]
    if limit:
        files = random.sample(files, min(limit, len(files)))
    for f in files:
        shutil.copy2(f, dst / f.name)
    print(f"  → {dst.relative_to(dst.parent.parent.parent)}: {len(files)} images")
    return len(files)


# ─── 140k Real and Fake Faces ──────────────────────────────────────────────────

def download_140k(output_dir: Path, max_per_class: int = 10000):
    """
    Kaggle: xhlulu/140k-real-and-fake-faces
    70k real (Flickr) + 70k fake (StyleGAN). ~2.5 GB.
    Populates: real/ and ai_generated/
    """
    print("\n[1/3] Downloading 140k Real and Fake Faces (Kaggle)...")
    dl_dir = output_dir / "_tmp_140k"
    dl_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "xhlulu/140k-real-and-fake-faces",
         "-p", str(dl_dir), "--unzip"],
        check=False,
    )

    if result.returncode != 0:
        print("\n  Kaggle download failed. Manual steps:")
        print("  1. Visit https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
        print(f"  2. Download + extract to: {dl_dir}")
        print("  3. Re-run with --skip-140k-download")
        return

    for split in ["train", "val"]:
        src_split = "train" if split == "train" else "test"
        for src_cls, dst_cls in [("real", "real"), ("fake", "ai_generated")]:
            src = dl_dir / "real_vs_fake" / "real-or-fake" / src_split / src_cls
            dst = output_dir / split / dst_cls
            if src.exists():
                _copy_images(src, dst, limit=max_per_class)
            else:
                print(f"  WARNING: {src} not found")

    shutil.rmtree(dl_dir, ignore_errors=True)
    print("  140k dataset organized.")


# ─── CASIA v2 — Edited/Manipulated images ─────────────────────────────────────

def download_casia(output_dir: Path, max_per_split: int = 5000):
    """
    Kaggle: sophatvathana/casia-dataset
    CASIA v2: copy-move + splicing manipulations. ~1 GB.
    Populates: edited/

    Alternative: https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset
    """
    print("\n[2/3] Downloading CASIA v2 (edited/manipulated images)...")
    dl_dir = output_dir / "_tmp_casia"
    dl_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "sophatvathana/casia-dataset",
         "-p", str(dl_dir), "--unzip"],
        check=False,
    )

    if result.returncode != 0:
        print("\n  Kaggle download failed. Manual steps:")
        print("  1. Visit https://www.kaggle.com/datasets/sophatvathana/casia-dataset")
        print(f"  2. Download + extract to: {dl_dir}")
        print("  3. Re-run with --skip-casia-download")
        _print_casia_alternatives()
        return

    _organize_casia(dl_dir, output_dir, max_per_split)
    shutil.rmtree(dl_dir, ignore_errors=True)


def _organize_casia(dl_dir: Path, output_dir: Path, max_per_split: int):
    """CASIA v2 contains Tp (tampered) and Au (authentic) folders."""
    tampered_dirs = list(dl_dir.rglob("Tp")) + list(dl_dir.rglob("tampered"))
    if not tampered_dirs:
        print("  WARNING: Could not find tampered images folder. Check extraction.")
        return

    all_tampered = []
    for d in tampered_dirs:
        all_tampered.extend([f for f in d.rglob("*") if f.suffix.lower() in EXTENSIONS])

    random.shuffle(all_tampered)
    split_idx = int(len(all_tampered) * 0.8)
    splits = {
        "train": all_tampered[:split_idx],
        "val": all_tampered[split_idx:],
    }

    for split, files in splits.items():
        dst = output_dir / split / "edited"
        dst.mkdir(parents=True, exist_ok=True)
        chosen = files[:max_per_split]
        for f in chosen:
            shutil.copy2(f, dst / f.name)
        print(f"  → {split}/edited: {len(chosen)} images")

    print("  CASIA organized.")


def _print_casia_alternatives():
    print("\n  Alternative edited image datasets:")
    print("  • CoMoFoD: https://www.vcl.fer.hr/comofod/")
    print("  • Coverage: https://github.com/wenbihan/coverage")
    print("  • Columbia Uncompressed: https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/")
    print("  • NIST16: https://www.nist.gov/system/files/documents/2019/01/17/nist16_0.zip")
    print(f"\n  Place edited images in: data/train/edited/ and data/val/edited/")


# ─── Optional: Stable Diffusion generated ─────────────────────────────────────

def generate_sd_images(output_dir: Path, n: int = 3000):
    """
    Generate additional AI images via Stable Diffusion (optional, needs GPU).
    pip install diffusers accelerate
    """
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        from PIL import Image

        print(f"\n[Optional] Generating {n} SD images for ai_generated/...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        ).to("cuda")
        pipe.safety_checker = None

        dst = output_dir / "train" / "ai_generated"
        dst.mkdir(parents=True, exist_ok=True)

        prompts = [
            "professional headshot portrait, studio lighting",
            "corporate portrait photo, business attire",
            "linkedin profile photo, smiling professional",
            "passport style photo, neutral expression",
            "candid photo of a person outdoors",
        ]
        for i in range(n):
            image = pipe(prompts[i % len(prompts)], num_inference_steps=25).images[0]
            image.save(dst / f"sd_{i:06d}.jpg")
            if (i + 1) % 200 == 0:
                print(f"  Generated {i+1}/{n}")
        print(f"  Done. {n} SD images → {dst}")

    except ImportError:
        print("  diffusers not installed: pip install diffusers accelerate")


# ─── Verify dataset structure ──────────────────────────────────────────────────

def verify(output_dir: Path):
    print("\n─── Dataset Summary ──────────────────────────────────────────")
    total = 0
    for split in ["train", "val"]:
        for cls in ["real", "ai_generated", "edited"]:
            d = output_dir / split / cls
            n = len([f for f in d.rglob("*") if f.suffix.lower() in EXTENSIONS]) if d.exists() else 0
            status = "✓" if n >= 500 else ("⚠" if n > 0 else "✗")
            print(f"  {status}  {split:5}/{cls:14}: {n:6,} images")
            total += n
    print(f"\n  Total: {total:,} images")
    print("  Recommended minimum: 1000 per class per split")
    print("─────────────────────────────────────────────────────────────")


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  choices=["140k", "casia", "sd", "all", "verify"], default="all",
                        help="Data source to download, or 'verify' to just show dataset stats")
    parser.add_argument("--output",  default="../data")
    parser.add_argument("--max",     type=int, default=10000, help="Max images per class per split")
    parser.add_argument("--sd-n",    type=int, default=3000,  help="SD images to generate")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.source in ("140k", "all"):
        download_140k(output, max_per_class=args.max)
    if args.source in ("casia", "all"):
        download_casia(output, max_per_split=args.max)
    if args.source == "sd":
        generate_sd_images(output, n=args.sd_n)

    verify(output)


if __name__ == "__main__":
    main()
