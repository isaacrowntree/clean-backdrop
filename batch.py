#!/usr/bin/env python3
"""
Clean Backdrop - Batch Processor
Runs shadow lift + optional LaMa marks removal on all images in a folder.

Usage:
    python batch.py /path/to/folder [options]

Examples:
    python batch.py "D:\\Photos\\Export\\My Shoot"
    python batch.py /mnt/d/Photos/Export/shoot --lift 80 --marks
    python batch.py "D:\\Photos\\shoot" --output cleaned --lift 100 --sensitivity 8
"""

import argparse
import os
import sys
import re
import glob
import time
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session
import torch

# Import techniques from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import compute_shadow_lift, detect_floor, detect_marks, get_lama, run_lama


def to_wsl_path(path):
    """Convert Windows path to WSL if needed."""
    path = path.strip().strip('"')
    m = re.match(r'^([A-Za-z]):\\', path)
    if m:
        path = '/mnt/' + m.group(1).lower() + path[2:].replace('\\', '/')
    return path


def process_image(fpath, out_path, session, lift_strength=0.7, do_marks=False, sensitivity=10):
    img = cv2.imread(fpath)
    if img is None:
        return False, "can't read"
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]

    # Segment subject
    pil_img = Image.open(fpath).convert("RGB")
    sm = np.array(remove(pil_img, session=session, only_mask=True))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, k, iterations=3)
    sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    pk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    protected = cv2.dilate(sm, pk, iterations=2)
    bg_mask = (protected < 128).astype(np.uint8)
    fm, fs = detect_floor(img, bg_mask)

    current = img.copy()

    # Shadow lift
    if lift_strength > 0:
        current = compute_shadow_lift(current, bg_mask, fm, sm, strength=lift_strength)

    # Subject composite
    sf = sm.astype(np.float32) / 255.0
    sf = cv2.GaussianBlur(sf, (3, 3), 0)[:, :, np.newaxis]
    final = (sf * img.astype(np.float32) + (1 - sf) * current.astype(np.float32))
    final = np.clip(final, 0, 255).astype(np.uint8)

    # LaMa marks
    if do_marks:
        marks_mask = detect_marks(final, bg_mask, fm, sensitivity=sensitivity)
        marks_mask[sm > 128] = 0
        cov = np.sum(marks_mask > 0) / (h * w) * 100
        if 0.1 < cov <= 5:
            model, device = get_lama()
            max_dim = 3072
            scale = min(max_dim / max(h, w), 1.0)
            if scale < 1.0:
                sh, sw = int(h * scale), int(w * scale)
                lr = run_lama(cv2.resize(final, (sw, sh), interpolation=cv2.INTER_AREA),
                              cv2.resize(marks_mask, (sw, sh), interpolation=cv2.INTER_NEAREST),
                              model, device)
                lr = cv2.resize(lr, (w, h), interpolation=cv2.INTER_LANCZOS4)
            else:
                lr = run_lama(final, marks_mask, model, device)
            mf = (marks_mask > 128).astype(np.float32)
            mf = cv2.GaussianBlur(mf, (3, 3), 0)[:, :, np.newaxis]
            final = (mf * lr.astype(np.float32) + (1 - mf) * final.astype(np.float32))
            final = np.clip(final, 0, 255).astype(np.uint8)

    # Save with ICC + EXIF
    orig_pil = Image.open(fpath)
    icc = orig_pil.info.get('icc_profile')
    exif = orig_pil.info.get('exif')
    rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(rgb)
    kw = {}
    if icc: kw['icc_profile'] = icc
    if exif: kw['exif'] = exif

    ext = os.path.splitext(out_path)[1].lower()
    if ext in ('.tif', '.tiff'):
        pil_out.save(out_path, **kw)
    else:
        pil_out.save(out_path, quality=98, subsampling=0, **kw)

    return True, f"{w}x{h}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch process studio photos - shadow lift + optional marks removal"
    )
    parser.add_argument("folder", help="Input folder path (Windows or WSL path)")
    parser.add_argument("--output", "-o", default="cleaned",
                        help="Output subfolder name (default: 'cleaned')")
    parser.add_argument("--lift", type=int, default=70,
                        help="Shadow lift strength 0-100 (default: 70)")
    parser.add_argument("--marks", action="store_true",
                        help="Also run LaMa on small marks/blemishes")
    parser.add_argument("--sensitivity", type=int, default=10,
                        help="Mark detection sensitivity (lower=more aggressive, default: 10)")
    parser.add_argument("--ext", default="jpg",
                        help="Image file extension to process (default: jpg)")

    args = parser.parse_args()

    folder = to_wsl_path(args.folder)
    if not os.path.isdir(folder):
        print(f"Error: folder not found: {folder}")
        sys.exit(1)

    out_dir = os.path.join(folder, args.output)
    os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(folder, f"*.{args.ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        # Try case insensitive
        files = sorted(glob.glob(os.path.join(folder, f"*.{args.ext.upper()}")))
    if not files:
        print(f"No .{args.ext} files found in {folder}")
        sys.exit(1)

    print(f"Clean Backdrop - Batch Processor")
    print(f"  Input:  {folder} ({len(files)} images)")
    print(f"  Output: {out_dir}")
    print(f"  Shadow lift: {args.lift}%")
    print(f"  Marks: {'on (sensitivity {})'.format(args.sensitivity) if args.marks else 'off'}")
    print()

    # Load segmentation model once
    print("Loading birefnet-portrait model...")
    session = new_session("birefnet-portrait")
    print("Ready.\n")

    lift = args.lift / 100.0
    t0 = time.time()
    ok = 0
    fail = 0

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        out_path = os.path.join(out_dir, fname)
        print(f"[{i+1}/{len(files)}] {fname} ", end="", flush=True)

        try:
            success, info = process_image(
                fpath, out_path, session,
                lift_strength=lift,
                do_marks=args.marks,
                sensitivity=args.sensitivity,
            )
            if success:
                print(f"-> {info}")
                ok += 1
            else:
                print(f"SKIP: {info}")
                fail += 1
        except Exception as e:
            print(f"ERROR: {e}")
            fail += 1

    elapsed = time.time() - t0
    per_img = elapsed / max(ok, 1)
    print(f"\nDone! {ok}/{len(files)} processed in {elapsed:.0f}s ({per_img:.1f}s/image)")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
