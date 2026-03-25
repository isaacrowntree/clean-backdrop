#!/usr/bin/env python3
"""
Clean Backdrop - Studio background cleanup tool.
Removes wrinkles, shadows, seams, and uneven lighting from studio backdrops
while preserving the subject.

Usage:
    python clean_backdrop.py input.jpg [output.jpg] [--color WHITE_HEX] [--feather PIXELS] [--preview]

Examples:
    python clean_backdrop.py photo.jpg
    python clean_backdrop.py photo.jpg cleaned.jpg --color "#F0F0F0"
    python clean_backdrop.py photo.jpg --preview
"""

import argparse
import sys
import os
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session


def detect_backdrop_color(image, mask, sample_margin=50):
    """Sample the backdrop color from areas known to be background."""
    h, w = image.shape[:2]
    bg_mask = (mask < 30)  # definite background

    # Sample from edges (likely pure backdrop)
    edge_samples = []
    regions = [
        image[sample_margin:h//4, sample_margin:w-sample_margin],  # top
        image[sample_margin:h-sample_margin, sample_margin:w//6],  # left
        image[sample_margin:h-sample_margin, w*5//6:w-sample_margin],  # right
    ]
    mask_regions = [
        bg_mask[sample_margin:h//4, sample_margin:w-sample_margin],
        bg_mask[sample_margin:h-sample_margin, sample_margin:w//6],
        bg_mask[sample_margin:h-sample_margin, w*5//6:w-sample_margin],
    ]

    for region, m_region in zip(regions, mask_regions):
        if m_region.any():
            pixels = region[m_region]
            if len(pixels) > 0:
                edge_samples.append(pixels)

    if edge_samples:
        all_samples = np.concatenate(edge_samples, axis=0)
        # Use the brightest quartile to get the "intended" backdrop color
        brightness = np.mean(all_samples, axis=1)
        bright_threshold = np.percentile(brightness, 75)
        bright_pixels = all_samples[brightness >= bright_threshold]
        if len(bright_pixels) > 0:
            return np.median(bright_pixels, axis=0).astype(np.uint8)

    return np.array([240, 240, 240], dtype=np.uint8)


def clean_backdrop(input_path, output_path=None, target_color=None, feather=15, preview=False):
    """
    Clean the backdrop of a studio photo.

    1. Use rembg to segment subject from background
    2. Detect the intended backdrop color
    3. Replace background with clean, uniform color
    4. Feather the edges for natural blending
    5. Preserve shadows near the subject's feet for realism
    """
    print(f"Loading {input_path}...")
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load {input_path}")
        sys.exit(1)

    # Ensure 3-channel BGR
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # Step 1: Segment subject
    print("Segmenting subject (this may take a moment on first run)...")
    pil_img = Image.open(input_path).convert("RGB")
    session = new_session("u2net_human_seg")
    result = remove(pil_img, session=session, only_mask=True)
    mask = np.array(result)  # 0=background, 255=subject

    # Step 2: Refine mask with morphological operations
    print("Refining mask...")
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Close small gaps in the subject
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_med, iterations=2)
    # Remove small noise in background
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Step 3: Detect or use target backdrop color
    if target_color is not None:
        # Parse hex color
        hex_color = target_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bg_color = np.array([b, g, r], dtype=np.uint8)  # BGR
        print(f"Using specified color: #{hex_color}")
    else:
        bg_color = detect_backdrop_color(img, refined_mask)
        hex_detected = f"#{bg_color[2]:02x}{bg_color[1]:02x}{bg_color[0]:02x}"
        print(f"Detected backdrop color: {hex_detected}")

    # Step 4: Create the clean background
    clean_bg = np.full_like(img, bg_color)

    # Step 5: Create feathered alpha mask for natural blending
    # Gaussian blur the mask edges
    feather_amount = max(1, feather) * 2 + 1  # must be odd
    alpha = refined_mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (feather_amount, feather_amount), 0)

    # Step 6: Preserve contact shadows
    # Detect shadow areas near the bottom of the subject
    print("Preserving contact shadows...")
    subject_bottom = 0
    for row in range(h - 1, 0, -1):
        if refined_mask[row].max() > 128:
            subject_bottom = row
            break

    # Create shadow preservation zone (area below and around subject's base)
    shadow_zone = np.zeros((h, w), dtype=np.float32)
    shadow_height = min(int(h * 0.15), 200)  # shadow zone height
    shadow_start = max(0, subject_bottom - 20)
    shadow_end = min(h, subject_bottom + shadow_height)

    if shadow_start < shadow_end:
        # Gradient that fades out with distance from subject
        for y in range(shadow_start, shadow_end):
            dist = (y - subject_bottom) / max(shadow_height, 1)
            shadow_zone[y, :] = max(0, 1.0 - dist) * 0.6

    # Detect actual shadow pixels (darker than backdrop)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg_brightness = int(np.mean(bg_color))
    shadow_pixels = (gray < bg_brightness * 0.85).astype(np.float32)
    shadow_pixels = cv2.GaussianBlur(shadow_pixels, (21, 21), 0)

    # Combine: only preserve shadows in the shadow zone that are actual shadows
    shadow_preserve = shadow_zone * shadow_pixels
    shadow_preserve = cv2.GaussianBlur(shadow_preserve, (31, 31), 0)

    # Step 7: Composite
    print("Compositing clean backdrop...")
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
    shadow_3ch = np.stack([shadow_preserve, shadow_preserve, shadow_preserve], axis=-1)

    # Blend: subject over clean background, with shadow areas partially preserved
    result = (alpha_3ch * img.astype(np.float32) +
              (1 - alpha_3ch) * (
                  shadow_3ch * img.astype(np.float32) +
                  (1 - shadow_3ch) * clean_bg.astype(np.float32)
              ))

    result = np.clip(result, 0, 255).astype(np.uint8)

    # Step 8: Final cleanup - smooth any remaining imperfections in background
    # Only smooth areas that are definitively background
    bg_only_mask = (refined_mask < 10).astype(np.uint8)
    bg_only_mask = cv2.erode(bg_only_mask, kernel_med, iterations=2)

    # Light bilateral filter on background areas for extra smoothness
    smoothed = cv2.bilateralFilter(result, 15, 75, 75)
    bg_blend = bg_only_mask.astype(np.float32)[:, :, np.newaxis]
    result = (bg_blend * smoothed.astype(np.float32) +
              (1 - bg_blend) * result.astype(np.float32)).astype(np.uint8)

    # Output
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_clean{ext}"

    if preview:
        # Save a side-by-side comparison
        comparison = np.hstack([img, result])
        preview_path = output_path.replace('.', '_preview.')
        cv2.imwrite(preview_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Preview saved: {preview_path}")

    ext = os.path.splitext(output_path)[1].lower()
    if ext in ('.tif', '.tiff'):
        cv2.imwrite(output_path, result)
    elif ext == '.png':
        cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"Cleaned image saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Clean studio backdrops - remove wrinkles, shadows, and imperfections"
    )
    parser.add_argument("input", help="Input image path (JPEG or TIFF)")
    parser.add_argument("output", nargs="?", help="Output path (default: input_clean.jpg)")
    parser.add_argument("--color", help="Target backdrop color as hex (e.g. '#FFFFFF'). Auto-detected if not specified.")
    parser.add_argument("--feather", type=int, default=15, help="Edge feather amount in pixels (default: 15)")
    parser.add_argument("--preview", action="store_true", help="Save a side-by-side comparison image")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    clean_backdrop(args.input, args.output, args.color, args.feather, args.preview)


if __name__ == "__main__":
    main()
