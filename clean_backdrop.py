#!/usr/bin/env python3
"""
Clean Backdrop - AI studio background retouching tool.

Detects and heals wrinkles, scuffs, dust, and cast shadows on studio backdrops
using LaMa inpainting. Subject and watermarks are never touched.
Works with both seamless paper backdrops and wall+floor setups.

Optionally uses Stable Diffusion for removing large foreign objects (--sd flag).

Usage:
    python clean_backdrop.py input.jpg [output.jpg] [--preview] [--sensitivity N] [--sd]
"""

import argparse
import sys
import os
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session
import torch

LAMA_MODEL_URL = "https://huggingface.co/fashn-ai/LaMa/resolve/main/big-lama.pt"
LAMA_MODEL_DIR = os.path.expanduser("~/.cache/clean-backdrop")
LAMA_MODEL_PATH = os.path.join(LAMA_MODEL_DIR, "big-lama.pt")


def download_lama_model():
    if os.path.exists(LAMA_MODEL_PATH):
        return LAMA_MODEL_PATH
    os.makedirs(LAMA_MODEL_DIR, exist_ok=True)
    print("  Downloading LaMa model (~200MB)...")
    import urllib.request
    urllib.request.urlretrieve(LAMA_MODEL_URL, LAMA_MODEL_PATH)
    return LAMA_MODEL_PATH


def pad_to_modulo(img, mod):
    h, w = img.shape[:2]
    pad_h = (mod - h % mod) % mod
    pad_w = (mod - w % mod) % mod
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    if img.ndim == 3:
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect'), (pad_h, pad_w)
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect'), (pad_h, pad_w)


def inpaint_lama(image, mask, model, device):
    """LaMa inpainting. mask: 255=inpaint, 0=keep."""
    h, w = image.shape[:2]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask_f = (mask > 128).astype(np.float32)

    img_rgb, (pad_h, pad_w) = pad_to_modulo(img_rgb, 8)
    mask_f, _ = pad_to_modulo(mask_f, 8)

    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t, mask_t)

    out = out[0].permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    if pad_h > 0:
        out = out[:-pad_h]
    if pad_w > 0:
        out = out[:, :-pad_w]
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def inpaint_sd(image_bgr, mask, device):
    """Stable Diffusion inpainting for large foreign object removal."""
    from diffusers import StableDiffusionInpaintPipeline

    h, w = image_bgr.shape[:2]
    print("  Loading Stable Diffusion inpainting model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.enable_xformers_memory_efficient_attention()

    sd_size = 768
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize((sd_size, sd_size), Image.LANCZOS)
    pil_mask = Image.fromarray(mask).resize((sd_size, sd_size), Image.NEAREST)

    print("  Running SD inpainting...")
    result = pipe(
        prompt="clean smooth studio photography backdrop, seamless, professional photo studio, even soft lighting",
        negative_prompt="wrinkles, creases, seams, tape, equipment, rigging, dark spots, blemishes, text, watermark, person, people",
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=35,
        guidance_scale=7.5,
        strength=0.9,
        height=sd_size,
        width=sd_size,
    ).images[0]

    result = result.resize((w, h), Image.LANCZOS)
    sd_result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    # Correct color drift: match chrominance to original
    sd_lab = cv2.cvtColor(sd_result, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_ref = mask < 128  # untouched areas as reference
    if np.sum(bg_ref) > 100:
        for ch in [1, 2]:
            orig_mean = np.mean(orig_lab[:, :, ch][bg_ref])
            sd_mean = np.mean(sd_lab[:, :, ch][mask > 128])
            sd_lab[:, :, ch] += (orig_mean - sd_mean)
        sd_lab = np.clip(sd_lab, 0, 255)
        sd_result = cv2.cvtColor(sd_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    del pipe
    torch.cuda.empty_cache()
    return sd_result


def detect_watermark_regions(img):
    """Detect watermark/logo regions in corners to protect them."""
    h, w = img.shape[:2]
    margin_h, margin_w = int(h * 0.12), int(w * 0.12)
    mask = np.zeros((h, w), dtype=np.uint8)

    corners = [
        (0, 0, margin_h, margin_w),
        (0, w - margin_w, margin_h, w),
        (h - margin_h, 0, h, margin_w),
        (h - margin_h, w - margin_w, h, w),
    ]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for y1, x1, y2, x2 in corners:
        region = gray[y1:y2, x1:x2]
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density > 0.02:
            _, thresh = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            thresh = cv2.dilate(thresh, kernel, iterations=3)
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], thresh)

    return mask


def segment_subject(input_path):
    """Segment subject with rembg + protect watermarks."""
    pil_img = Image.open(input_path).convert("RGB")
    session = new_session("u2net_human_seg")
    mask = np.array(remove(pil_img, session=session, only_mask=True))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    img = cv2.imread(input_path)
    if img is not None:
        watermark = detect_watermark_regions(img)
        mask = np.maximum(mask, watermark)

    return mask


def detect_floor_boundary(img, bg_mask):
    """
    Detect if there's a distinct floor surface (different color/material from wall).
    Returns a mask of the floor region, or zeros if it's all one surface.
    """
    h, w = img.shape[:2]
    floor_mask = np.zeros((h, w), dtype=np.uint8)

    # Sample wall color from upper-mid background
    wall_top, wall_bot = h // 4, h // 2
    wall_sample = bg_mask[wall_top:wall_bot, :] > 0
    if np.sum(wall_sample) < 100:
        return floor_mask
    wall_color = np.median(img[wall_top:wall_bot][wall_sample], axis=0)

    # Sample floor color from bottom background
    floor_top = int(h * 0.85)
    floor_sample = bg_mask[floor_top:, :] > 0
    if np.sum(floor_sample) < 100:
        return floor_mask
    floor_color = np.median(img[floor_top:][floor_sample], axis=0)

    # Only flag as different floor if colors are clearly different
    color_diff = np.sqrt(np.sum((wall_color.astype(float) - floor_color.astype(float)) ** 2))
    if color_diff < 60:
        return floor_mask  # Same surface, no floor boundary

    print(f"  Detected different floor surface (color diff: {color_diff:.0f})")

    # Find the transition row
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bottom_half = gray[h // 2:, :]
    bg_bottom = bg_mask[h // 2:, :]

    row_means = []
    for r in range(bottom_half.shape[0]):
        pixels = bottom_half[r][bg_bottom[r] > 0]
        row_means.append(np.mean(pixels) if len(pixels) > 0 else 0)
    row_means = np.array(row_means)

    if len(row_means) > 20:
        smoothed = cv2.GaussianBlur(row_means.reshape(-1, 1), (1, 21), 0).flatten()
        gradient = np.diff(smoothed)
        if len(gradient) > 0 and np.min(gradient) < -0.5:
            threshold = np.min(gradient) * 0.3
            candidates = np.where(gradient < threshold)[0]
            if len(candidates) > 0:
                floor_start = h // 2 + candidates[0]
                floor_mask[floor_start:, :] = bg_mask[floor_start:, :]

    return floor_mask


def detect_blemishes(img, bg_mask, floor_mask, sensitivity=15):
    """
    Detect blemishes on the backdrop for LaMa healing:
    - Wrinkles, creases, scuffs on wall/paper
    - Dust and sensor spots
    - Cast shadows from the subject on the wall
    - Does NOT touch the floor if it's a different surface
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Work area: background minus floor (if floor is different surface)
    work_mask = bg_mask.copy()
    work_mask[floor_mask > 0] = 0

    # === Wrinkles & texture (high-frequency detail that shouldn't be on a smooth backdrop) ===
    low_freq = cv2.GaussianBlur(gray, (0, 0), sigmaX=8)
    high_freq = np.abs(gray - low_freq)
    wrinkles = ((high_freq > sensitivity) & (work_mask > 0)).astype(np.uint8)

    # === Cast shadows: areas darker than their local neighborhood ===
    work_float = gray * work_mask.astype(np.float32)
    work_weight = work_mask.astype(np.float32)
    blur_k = min(max(h, w) // 6, 501)
    blur_k = blur_k + (1 - blur_k % 2)
    blurred = cv2.GaussianBlur(work_float, (blur_k, blur_k), 0)
    blurred_w = cv2.GaussianBlur(work_weight, (blur_k, blur_k), 0)
    blurred_w = np.maximum(blurred_w, 1e-6)
    expected = blurred / blurred_w

    shadow_deviation = expected - gray
    shadows = ((shadow_deviation > 20) & (work_mask > 0)).astype(np.uint8)

    # === Dark bars/stripes: baseboards, edges, trim, dark borders ===
    wall_brightness = np.percentile(gray[work_mask > 0], 70) if np.sum(work_mask) > 100 else 200
    dark_bars = ((gray < wall_brightness * 0.6) & (work_mask > 0)).astype(np.uint8)

    # Catch baseboards/bars at the wall-floor boundary
    if np.sum(floor_mask) > 0:
        floor_edge_up = cv2.dilate(floor_mask,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80)), iterations=1)
        floor_edge_up = np.clip(floor_edge_up.astype(np.int16) - floor_mask.astype(np.int16), 0, 1).astype(np.uint8)
        boundary_dark = ((gray < wall_brightness * 0.7) & (floor_edge_up > 0) & (bg_mask > 0)).astype(np.uint8)
        dark_bars = np.clip(dark_bars + boundary_dark, 0, 1).astype(np.uint8)

        # Check the TOP of the floor zone for dark bars/baseboards
        floor_pixels = gray[floor_mask > 0]
        if len(floor_pixels) > 100:
            floor_brightness = np.median(floor_pixels)
            floor_top_band = cv2.erode(floor_mask,
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=1)
            floor_top_band = np.clip(floor_mask.astype(np.int16) - floor_top_band.astype(np.int16), 0, 1).astype(np.uint8)
            floor_top_band = cv2.dilate(floor_top_band,
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40)), iterations=1)
            floor_top_band = floor_top_band & floor_mask
            baseboard = ((gray < min(wall_brightness * 0.5, floor_brightness * 0.6)) &
                         (floor_top_band > 0) & (bg_mask > 0)).astype(np.uint8)
            dark_bars = np.clip(dark_bars + baseboard, 0, 1).astype(np.uint8)

    # Top edge dark areas (rigging, dark borders)
    edge_zone_h = max(int(h * 0.05), 20)
    top_dark = ((gray[:edge_zone_h, :] < wall_brightness * 0.7) &
                (bg_mask[:edge_zone_h, :] > 0)).astype(np.uint8)
    dark_bars[:edge_zone_h, :] = np.clip(
        dark_bars[:edge_zone_h, :] + top_dark, 0, 1).astype(np.uint8)

    # === Scuffs, marks, and dirt: localized dark spots at multiple scales ===
    local_mean_small = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
    scuffs_small = ((local_mean_small - gray > 12) & (work_mask > 0)).astype(np.uint8)

    local_mean_large = cv2.GaussianBlur(gray, (0, 0), sigmaX=40)
    scuffs_large = ((local_mean_large - gray > 15) & (work_mask > 0)).astype(np.uint8)

    # === Bright marks/scuffs (lighter than surroundings) ===
    bright_marks = ((gray - local_mean_small > 12) & (work_mask > 0)).astype(np.uint8)

    # === Edge detection for seam lines and hard marks ===
    edges = cv2.Canny(gray.astype(np.uint8), 15, 50)
    edge_marks = ((edges > 0) & (work_mask > 0)).astype(np.uint8)
    # Vertical seams specifically (common in wall backdrops)
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    vert_seams = ((sobelx > 15) & (work_mask > 0)).astype(np.uint8)
    edge_marks = np.clip(edge_marks + vert_seams, 0, 1).astype(np.uint8)

    # Combine all detections
    combined = np.clip(
        wrinkles + shadows + scuffs_small + scuffs_large + bright_marks + edge_marks + dark_bars,
        0, 1
    ).astype(np.uint8)

    # Dilate to cover edges of detected areas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=2)
    # Close gaps between nearby marks
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)

    # Never touch floor
    combined[floor_mask > 0] = 0

    # Safety cap at 45% - walls can genuinely have a lot of marks
    coverage = np.sum(combined > 0) / (h * w)
    if coverage > 0.45:
        print(f"  Warning: {coverage*100:.0f}% detected, tightening to strongest blemishes...")
        wrinkles_strict = ((high_freq > sensitivity * 2) & (work_mask > 0)).astype(np.uint8)
        shadows_strict = ((shadow_deviation > 35) & (work_mask > 0)).astype(np.uint8)
        scuffs_strict = ((local_mean_small - gray > 20) & (work_mask > 0)).astype(np.uint8)
        combined = np.clip(wrinkles_strict + shadows_strict + scuffs_strict, 0, 1).astype(np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        combined[floor_mask > 0] = 0

    return combined * 255


def detect_foreign_objects(img, bg_mask, floor_mask):
    """
    Detect large foreign objects for SD removal (--sd flag only):
    rigging, light stands, tape, cables.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)

    # Work on wall only
    work_mask = bg_mask.copy()
    work_mask[floor_mask > 0] = 0

    # Get expected backdrop brightness
    wall_pixels = gray[work_mask > 0]
    if len(wall_pixels) < 100:
        return np.zeros((h, w), dtype=np.uint8)
    clean_brightness = np.percentile(wall_pixels, 85)

    # Foreign: highly saturated OR extremely dark objects on the backdrop
    foreign_colored = ((saturation > 50) & (work_mask > 0)).astype(np.uint8)
    foreign_dark = ((gray < min(clean_brightness * 0.3, 40)) & (work_mask > 0)).astype(np.uint8)
    combined = np.clip(foreign_colored + foreign_dark, 0, 1).astype(np.uint8)

    # Only keep substantial connected components
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    min_area = (h * w) * 0.001
    filtered = np.zeros_like(combined)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 1

    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    filtered = cv2.dilate(filtered, kernel_med, iterations=2)
    filtered[floor_mask > 0] = 0

    return filtered * 255


def clean_backdrop(input_path, output_path=None, sensitivity=15, preview=False, use_sd=False):
    print(f"Loading {input_path}...")
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load {input_path}")
        sys.exit(1)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Step 1: Segment subject ──
    print("\n[Step 1] Segmenting subject...")
    subject_mask = segment_subject(input_path)
    protect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    protected = cv2.dilate(subject_mask, protect_kernel, iterations=2)
    bg_mask = (protected < 128).astype(np.uint8)

    # ── Step 2: Detect floor boundary ──
    print("\n[Step 2] Analyzing backdrop surfaces...")
    floor_mask = detect_floor_boundary(img, bg_mask)
    has_floor = np.sum(floor_mask) > 0
    if not has_floor:
        print("  Single surface detected (seamless backdrop or same material)")
    current = img.copy()

    # ── Step 3 (optional): SD for foreign object removal ──
    sd_mask = np.zeros((h, w), dtype=np.uint8)
    if use_sd:
        print("\n[Step 3] Detecting foreign objects for SD removal...")
        sd_mask = detect_foreign_objects(img, bg_mask, floor_mask)
        sd_mask[protected > 128] = 0
        sd_coverage = np.sum(sd_mask > 0) / (h * w) * 100

        if sd_coverage > 0.3:
            print(f"  Found {sd_coverage:.1f}% foreign objects")
            sd_result = inpaint_sd(current, sd_mask, device)
            mask_f = (sd_mask > 128).astype(np.float32)
            mask_f = cv2.GaussianBlur(mask_f, (21, 21), 0)
            mask_3ch = mask_f[:, :, np.newaxis]
            current = (mask_3ch * sd_result.astype(np.float32) +
                       (1 - mask_3ch) * current.astype(np.float32))
            current = np.clip(current, 0, 255).astype(np.uint8)
            print("  SD pass complete.")
        else:
            print("  No significant foreign objects found.")
    else:
        print("\n[Step 3] Skipping SD (use --sd to remove foreign objects)")

    # ── Step 4: LaMa for blemish healing ──
    print("\n[Step 4] Detecting blemishes...")
    lama_mask = detect_blemishes(current, bg_mask, floor_mask, sensitivity)
    lama_mask[protected > 128] = 0
    lama_mask[sd_mask > 128] = 0
    lama_coverage = np.sum(lama_mask > 0) / (h * w) * 100

    if lama_coverage > 0.1:
        print(f"  Found {lama_coverage:.1f}% blemishes to heal")
        model_path = download_lama_model()
        print(f"  Loading LaMa on {device}...")
        lama_model = torch.jit.load(model_path, map_location=device)
        lama_model.eval()

        # For large images, process at a reasonable resolution
        max_lama_dim = 2048
        scale = min(max_lama_dim / max(h, w), 1.0)
        if scale < 1.0:
            sh, sw = int(h * scale), int(w * scale)
            print(f"  Downscaling to {sw}x{sh} for LaMa...")
            img_small = cv2.resize(current, (sw, sh), interpolation=cv2.INTER_AREA)
            mask_small = cv2.resize(lama_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            lama_result_small = inpaint_lama(img_small, mask_small, lama_model, device)
            lama_result = cv2.resize(lama_result_small, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            print("  Running LaMa inpainting...")
            lama_result = inpaint_lama(current, lama_mask, lama_model, device)

        # Blend only where mask is active
        mask_f = (lama_mask > 128).astype(np.float32)
        mask_f = cv2.GaussianBlur(mask_f, (7, 7), 0)
        mask_3ch = mask_f[:, :, np.newaxis]
        current = (mask_3ch * lama_result.astype(np.float32) +
                   (1 - mask_3ch) * current.astype(np.float32))
        current = np.clip(current, 0, 255).astype(np.uint8)

        del lama_model
        torch.cuda.empty_cache()
        print("  LaMa pass complete.")
    else:
        print("  No blemishes found, backdrop looks clean.")

    # ── Step 5: Light polish ──
    print("\n[Step 5] Final polish...")
    smoothed = current.copy()
    for _ in range(2):
        smoothed = cv2.bilateralFilter(smoothed, 9, 25, 25)

    # Only smooth background, leave subject pristine
    smooth_blend = bg_mask.astype(np.float32)
    # Don't smooth the floor if it's a different surface
    if has_floor:
        smooth_blend[floor_mask > 0] = 0
    smooth_blend = cv2.GaussianBlur(smooth_blend, (11, 11), 0)
    smooth_3ch = smooth_blend[:, :, np.newaxis]
    current = (smooth_3ch * smoothed.astype(np.float32) +
               (1 - smooth_3ch) * current.astype(np.float32))
    current = np.clip(current, 0, 255).astype(np.uint8)

    # Final composite: guarantee original subject pixels
    subject_f = subject_mask.astype(np.float32) / 255.0
    subject_f = cv2.GaussianBlur(subject_f, (3, 3), 0)
    subject_3ch = subject_f[:, :, np.newaxis]
    final = (subject_3ch * img.astype(np.float32) +
             (1 - subject_3ch) * current.astype(np.float32))
    final = np.clip(final, 0, 255).astype(np.uint8)

    # ── Output ──
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_clean{ext}"

    if preview:
        overlay = img.copy()
        if use_sd:
            overlay[sd_mask > 128] = [0, 0, 255]       # red = SD areas
        overlay[lama_mask > 128] = [0, 165, 255]        # orange = LaMa areas
        blended_overlay = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        comparison = np.hstack([img, blended_overlay, final])
        preview_path = output_path.replace('.', '_preview.')
        # Save preview with ICC profile too
        comp_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        comp_pil = Image.fromarray(comp_rgb)
        original_pil_p = Image.open(input_path)
        icc = original_pil_p.info.get('icc_profile')
        comp_pil.save(preview_path, quality=95, subsampling=0,
                      **({"icc_profile": icc} if icc else {}))
        print(f"\nPreview saved: {preview_path}")
        print("  Left=original | Middle=detected | Right=result")

    # Save with original color profile and metadata preserved
    # cv2 strips ICC profiles which causes color shifts in viewers
    final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    final_pil = Image.fromarray(final_rgb)

    # Copy ICC profile and EXIF from original
    original_pil = Image.open(input_path)
    icc_profile = original_pil.info.get('icc_profile')
    exif_data = original_pil.info.get('exif')

    save_kwargs = {}
    if icc_profile:
        save_kwargs['icc_profile'] = icc_profile
    if exif_data:
        save_kwargs['exif'] = exif_data

    ext_out = os.path.splitext(output_path)[1].lower()
    if ext_out in ('.tif', '.tiff'):
        final_pil.save(output_path, **save_kwargs)
    elif ext_out == '.png':
        final_pil.save(output_path, **save_kwargs)
    else:
        final_pil.save(output_path, quality=98, subsampling=0, **save_kwargs)
    print(f"\nCleaned image saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="AI studio backdrop retouching - heals blemishes, wrinkles, and shadows"
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", nargs="?", help="Output path (default: input_clean.ext)")
    parser.add_argument("--sensitivity", type=int, default=15,
                        help="Blemish detection sensitivity: lower=more aggressive (default: 15)")
    parser.add_argument("--preview", action="store_true",
                        help="Save a 3-up comparison (original | detected | result)")
    parser.add_argument("--sd", action="store_true",
                        help="Also use Stable Diffusion to remove large foreign objects (rigging, stands)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    clean_backdrop(args.input, args.output, args.sensitivity, args.preview, args.sd)


if __name__ == "__main__":
    main()
