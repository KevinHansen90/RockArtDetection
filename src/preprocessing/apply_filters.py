#!/usr/bin/env python3
import os, sys, argparse, shutil
import numpy as np
import cv2
from PIL import Image, ImageFilter
from tqdm import tqdm


# ────────────────────────────── filters ──────────────────────────────
def apply_bilateral(pil_img, d=11, sigmaColor=40, sigmaSpace=75):
    """Edge-preserving denoise (recommended defaults tuned for rock-art JPEGs)."""
    arr = cv2.bilateralFilter(np.asarray(pil_img), d, sigmaColor, sigmaSpace)
    return Image.fromarray(arr)


def apply_unsharp(pil_img, radius=2.0, percent=250, threshold=0):
    """Small-radius, high-strength un-sharp masking to pop faint strokes."""
    return pil_img.filter(ImageFilter.UnsharpMask(radius, percent, threshold))


def apply_laplacian(pil_img, alpha=0.8, beta=0.2):
    """
    Laplacian edge-boost blended with the original (alpha*orig + beta*edges).
    """
    rgb  = np.asarray(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap  = cv2.convertScaleAbs(lap)
    lap3 = cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)
    blend = cv2.addWeighted(rgb, alpha, lap3, beta, 0)
    return Image.fromarray(blend)


def apply_clahe(pil_img, clipLimit=4.0, tileGrid=(8, 8)):
    """CLAHE on the L channel of LAB colour space."""
    lab  = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit, tileGrid)
    l2    = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


def apply_dstretch(pil_img, exaggeration=1.2):
    """
    Minimal DStretch-style decorrelation stretch.

    • Rotates RGB to PCA space
    • Scales each principal component so its variance equals the largest one,
      then multiplies by 'exaggeration' to push colours further apart.
    """
    rgb  = np.asarray(pil_img.convert('RGB')).astype(np.float32)
    flat = rgb.reshape(-1, 3)
    mean = flat.mean(0, keepdims=True)
    cov  = np.cov(flat - mean, rowvar=False)

    eigval, eigvec = np.linalg.eigh(cov)
    order  = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    scale = exaggeration / np.sqrt(eigval + 1e-5)
    T     = eigvec @ np.diag(scale) @ eigvec.T

    stretched = ((flat - mean) @ T + mean).clip(0, 255).astype(np.uint8)
    return Image.fromarray(stretched.reshape(rgb.shape))


# ───────────────────────────── helpers ───────────────────────────────
def copy_labels(src_labels_dir, dst_labels_dir):
    """Copy YOLO-format .txt label files, preserving directory tree."""
    if os.path.exists(dst_labels_dir):
        shutil.rmtree(dst_labels_dir)
    shutil.copytree(src_labels_dir, dst_labels_dir)


def process_subset(base_dir, output_dir, subset, filter_fn):
    src_img = os.path.join(base_dir, subset, 'images')
    src_lbl = os.path.join(base_dir, subset, 'labels')
    dst_img = os.path.join(output_dir, subset, 'images')
    dst_lbl = os.path.join(output_dir, subset, 'labels')

    os.makedirs(dst_img, exist_ok=True)
    copy_labels(src_lbl, dst_lbl)

    img_files = [f for f in os.listdir(src_img)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        print(f"Warning: no images in {src_img}", file=sys.stderr)
        return

    print(f"[{subset}] {len(img_files)} images")
    for fname in tqdm(img_files, desc=f"{subset}"):
        src = os.path.join(src_img, fname)
        dst = os.path.join(dst_img, fname)
        try:
            img = Image.open(src).convert('RGB')
            filt = filter_fn(img)
            filt.save(dst, quality=95)
        except Exception as e:
            print(f"Error {src}: {e}", file=sys.stderr)


# ───────────────────────── main CLI ─────────────────────────
def main():
    fns = {
        'bilateral': apply_bilateral,
        'unsharp'  : apply_unsharp,
        'laplacian': apply_laplacian,
        'clahe'    : apply_clahe,
        'dstretch' : apply_dstretch,
    }

    ap = argparse.ArgumentParser(
        description="Apply contrast/denoise filters to train/val/test dataset.")
    ap.add_argument('--base_dir',   required=True,
                    help='Root dir with train/, val/, test/ splits')
    ap.add_argument('--output_dir', required=True,
                    help='Destination root (will be created)')
    ap.add_argument('--filter_type', choices=fns.keys(), required=True,
                    help='Which filter to apply')
    args = ap.parse_args()

    for subset in ['train', 'val', 'test']:
        process_subset(args.base_dir, args.output_dir,
                       subset, fns[args.filter_type])

    print(f"Done ➜ {args.output_dir}")


if __name__ == '__main__':
    main()


