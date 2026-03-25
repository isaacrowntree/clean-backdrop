#!/usr/bin/env python3
"""
Clean Backdrop - Batch Processor
Shadow lift + frequency separation on all images in a folder.

Usage:
    python batch.py /path/to/folder [options]

Examples:
    python batch.py "D:\\Photos\\Export\\My Shoot"
    python batch.py /mnt/d/Photos/shoot --lift 80 --texture 50
    python batch.py "D:\\Photos\\shoot" --output cleaned --lift 100 --texture 70
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import shadow_lift, freq_separation, detect_floor


def to_wsl_path(path):
    path = path.strip().strip('"')
    m = re.match(r'^([A-Za-z]):\\', path)
    if m:
        path = '/mnt/' + m.group(1).lower() + path[2:].replace('\\', '/')
    return path


def process_image(fpath, out_path, session, lift_strength=0.7, tex_strength=0.5, no_floor=False):
    img = cv2.imread(fpath)
    if img is None:
        return False, "can't read"
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]

    # Segment
    pil_img = Image.open(fpath).convert("RGB")
    sm = np.array(remove(pil_img, session=session, only_mask=True))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, k, iterations=3)
    sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    pk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    protected = cv2.dilate(sm, pk, iterations=2)
    bg_mask = (protected < 128).astype(np.uint8)
    if no_floor:
        fm = np.zeros((h, w), dtype=np.uint8)
    else:
        fm, fs = detect_floor(img, bg_mask)

    current = img.copy()

    # Shadow lift
    if lift_strength > 0:
        current = shadow_lift(current, bg_mask, fm, sm, strength=lift_strength)

    # Frequency separation
    if tex_strength > 0:
        current = freq_separation(current, bg_mask, fm, sm, strength=tex_strength)

    # Save with ICC + EXIF
    orig_pil = Image.open(fpath)
    icc = orig_pil.info.get('icc_profile')
    exif = orig_pil.info.get('exif')
    rgb = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
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
        description="Batch process studio photos - shadow lift + texture smoothing"
    )
    parser.add_argument("folder", help="Input folder (Windows or WSL path)")
    parser.add_argument("--output", "-o", default="cleaned", help="Output subfolder (default: cleaned)")
    parser.add_argument("--lift", type=int, default=70, help="Shadow lift 0-100 (default: 70)")
    parser.add_argument("--texture", type=int, default=50, help="Texture smoothing 0-100 (default: 50)")
    parser.add_argument("--ext", default="jpg", help="File extension (default: jpg)")
    parser.add_argument("--no-floor", action="store_true", help="Disable floor detection (treat everything as wall)")

    args = parser.parse_args()

    folder = to_wsl_path(args.folder)
    if not os.path.isdir(folder):
        print(f"Error: not found: {folder}")
        sys.exit(1)

    out_dir = os.path.join(folder, args.output)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(folder, f"*.{args.ext}")))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, f"*.{args.ext.upper()}")))
    if not files:
        print(f"No .{args.ext} files in {folder}")
        sys.exit(1)

    print(f"Clean Backdrop - Batch")
    print(f"  Input:   {folder} ({len(files)} images)")
    print(f"  Output:  {out_dir}")
    print(f"  Shadow:  {args.lift}%")
    print(f"  Texture: {args.texture}%")
    print()

    print("Loading birefnet-portrait model (GPU)...")
    session = new_session("birefnet-portrait")
    print("Ready.\n")

    t0 = time.time()
    ok = 0

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        out_path = os.path.join(out_dir, fname)
        print(f"[{i+1}/{len(files)}] {fname} ", end="", flush=True)

        try:
            success, info = process_image(fpath, out_path, session,
                                           lift_strength=args.lift / 100,
                                           tex_strength=args.texture / 100,
                                           no_floor=args.no_floor)
            if success:
                print(f"-> {info}")
                ok += 1
            else:
                print(f"SKIP: {info}")
        except Exception as e:
            print(f"ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\nDone! {ok}/{len(files)} in {elapsed:.0f}s ({elapsed/max(ok,1):.1f}s/img)")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
