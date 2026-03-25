#!/usr/bin/env python3
"""
Clean Backdrop - Web UI
Local web interface with sliders for tuning detection and previewing results.
"""

import os
import io
import base64
import json
import tempfile
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template_string, request, jsonify
from rembg import remove, new_session
import torch

LAMA_MODEL_URL = "https://huggingface.co/fashn-ai/LaMa/resolve/main/big-lama.pt"
LAMA_MODEL_DIR = os.path.expanduser("~/.cache/clean-backdrop")
LAMA_MODEL_PATH = os.path.join(LAMA_MODEL_DIR, "big-lama.pt")

app = Flask(__name__)

# Global state for loaded models and current image
state = {
    "img": None,
    "img_path": None,
    "subject_mask": None,
    "bg_mask": None,
    "floor_mask": None,
    "lama_model": None,
    "device": None,
    "preview_size": 1200,  # max dimension for preview
    "icc_profile": None,
}


def download_lama():
    if os.path.exists(LAMA_MODEL_PATH):
        return
    os.makedirs(LAMA_MODEL_DIR, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(LAMA_MODEL_URL, LAMA_MODEL_PATH)


def get_lama_model():
    if state["lama_model"] is None:
        download_lama()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state["device"] = device
        state["lama_model"] = torch.jit.load(LAMA_MODEL_PATH, map_location=device)
        state["lama_model"].eval()
    return state["lama_model"], state["device"]


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


def resize_for_preview(img, max_dim):
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale
    return img, 1.0


def img_to_b64(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    save_kwargs = {}
    if state.get("icc_profile"):
        save_kwargs["icc_profile"] = state["icc_profile"]
    pil.save(buf, format='JPEG', quality=90, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode()


def detect_floor_boundary(img, bg_mask):
    h, w = img.shape[:2]
    floor_mask = np.zeros((h, w), dtype=np.uint8)
    wall_top, wall_bot = h // 4, h // 2
    wall_sample = bg_mask[wall_top:wall_bot, :] > 0
    if np.sum(wall_sample) < 100:
        return floor_mask
    wall_color = np.median(img[wall_top:wall_bot][wall_sample], axis=0)
    floor_top = int(h * 0.85)
    floor_sample = bg_mask[floor_top:, :] > 0
    if np.sum(floor_sample) < 100:
        return floor_mask
    floor_color = np.median(img[floor_top:][floor_sample], axis=0)
    color_diff = np.sqrt(np.sum((wall_color.astype(float) - floor_color.astype(float)) ** 2))
    if color_diff < 60:
        return floor_mask

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


def detect_blemishes(img, bg_mask, floor_mask, sensitivity=15, shadow_strength=20,
                     scuff_strength=12, detect_dark_bars=True, detect_edges=True):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    work_mask = bg_mask.copy()
    work_mask[floor_mask > 0] = 0

    # Wrinkles
    low_freq = cv2.GaussianBlur(gray, (0, 0), sigmaX=8)
    high_freq = np.abs(gray - low_freq)
    wrinkles = ((high_freq > sensitivity) & (work_mask > 0)).astype(np.uint8)

    # Shadows
    work_float = gray * work_mask.astype(np.float32)
    work_weight = work_mask.astype(np.float32)
    blur_k = min(max(h, w) // 6, 501)
    blur_k = blur_k + (1 - blur_k % 2)
    blurred = cv2.GaussianBlur(work_float, (blur_k, blur_k), 0)
    blurred_w = cv2.GaussianBlur(work_weight, (blur_k, blur_k), 0)
    blurred_w = np.maximum(blurred_w, 1e-6)
    expected = blurred / blurred_w
    shadow_deviation = expected - gray
    shadows = ((shadow_deviation > shadow_strength) & (work_mask > 0)).astype(np.uint8)

    # Scuffs
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
    scuffs = ((local_mean - gray > scuff_strength) & (work_mask > 0)).astype(np.uint8)
    local_mean_large = cv2.GaussianBlur(gray, (0, 0), sigmaX=40)
    scuffs_large = ((local_mean_large - gray > scuff_strength * 1.3) & (work_mask > 0)).astype(np.uint8)
    bright_marks = ((gray - local_mean > scuff_strength) & (work_mask > 0)).astype(np.uint8)

    # Dark bars / baseboards / dark edges
    dark_bars = np.zeros((h, w), dtype=np.uint8)
    if detect_dark_bars:
        wall_brightness = np.percentile(gray[work_mask > 0], 70) if np.sum(work_mask) > 100 else 200

        # Anything much darker than the wall
        dark_bars = ((gray < wall_brightness * 0.6) & (work_mask > 0)).astype(np.uint8)

        # Catch baseboards/bars at the wall-floor boundary
        # These sit IN the floor zone but are actually wall imperfections
        if np.sum(floor_mask) > 0:
            # Expand upward from floor into wall zone
            floor_edge_up = cv2.dilate(floor_mask,
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80)), iterations=1)
            floor_edge_up = np.clip(floor_edge_up.astype(np.int16) - floor_mask.astype(np.int16), 0, 1).astype(np.uint8)
            boundary_dark = ((gray < wall_brightness * 0.7) & (floor_edge_up > 0) & (bg_mask > 0)).astype(np.uint8)
            dark_bars = np.clip(dark_bars + boundary_dark, 0, 1).astype(np.uint8)

            # Scan the entire floor zone for dark horizontal bars/baseboards
            # These are much darker than the floor and span the width
            floor_pixels = gray[floor_mask > 0]
            if len(floor_pixels) > 100:
                floor_brightness = np.median(floor_pixels)
                # Anything in the floor zone that's very dark = baseboard/bar
                baseboard = ((gray < min(wall_brightness * 0.5, floor_brightness * 0.5)) &
                             (floor_mask > 0) & (bg_mask > 0)).astype(np.uint8)
                # Dilate to cover full bar width
                baseboard = cv2.dilate(baseboard,
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)), iterations=2)
                baseboard = baseboard & bg_mask  # still only on background
                dark_bars = np.clip(dark_bars + baseboard, 0, 1).astype(np.uint8)

        # Top and bottom edges of the background (rigging, dark borders)
        edge_zone_h = max(int(h * 0.05), 20)
        top_dark = ((gray[:edge_zone_h, :] < wall_brightness * 0.7) &
                    (bg_mask[:edge_zone_h, :] > 0)).astype(np.uint8)
        dark_bars[:edge_zone_h, :] = np.clip(
            dark_bars[:edge_zone_h, :] + top_dark, 0, 1).astype(np.uint8)

    # Seam lines / edges
    edge_marks = np.zeros((h, w), dtype=np.uint8)
    if detect_edges:
        # Lower thresholds to catch subtle seams
        edges = cv2.Canny(gray.astype(np.uint8), 15, 50)
        edge_marks = ((edges > 0) & (work_mask > 0)).astype(np.uint8)

        # Also use Sobel for vertical seams specifically (common in wall backdrops)
        sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        vert_seams = ((sobelx > 15) & (work_mask > 0)).astype(np.uint8)
        edge_marks = np.clip(edge_marks + vert_seams, 0, 1).astype(np.uint8)

    combined = np.clip(
        wrinkles + shadows + scuffs + scuffs_large + bright_marks + edge_marks + dark_bars,
        0, 1
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)

    # Exclude floor EXCEPT for detected dark bars (baseboards)
    floor_exclude = floor_mask.copy()
    if detect_dark_bars:
        floor_exclude[dark_bars > 0] = 0  # don't exclude bars from the mask
    combined[floor_exclude > 0] = 0

    return combined * 255


def generate_seamless_backdrop(img, bg_mask, subject_mask, floor_mask, backdrop_color=None, floor_darken=0.75):
    """
    Generate a synthetic seamless paper backdrop.
    - Clean wall color with subtle center-bright vignette
    - Smooth gradient to darker floor (simulates light falloff on paper)
    - No visible wall-floor seam (simulates the paper curve)
    - Preserves subject contact shadows
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Determine the clean wall color
    if backdrop_color is None:
        wall_mask = bg_mask.copy()
        wall_mask[floor_mask > 0] = 0
        wall_pixels = img[wall_mask > 0]
        if len(wall_pixels) > 100:
            brightness = np.mean(wall_pixels, axis=1)
            bright = wall_pixels[brightness > np.percentile(brightness, 75)]
            if len(bright) > 10:
                backdrop_color = np.median(bright, axis=0).astype(np.float32)
            else:
                backdrop_color = np.array([240, 240, 240], dtype=np.float32)
        else:
            backdrop_color = np.array([240, 240, 240], dtype=np.float32)

    # Create the synthetic backdrop
    backdrop = np.zeros((h, w, 3), dtype=np.float32)

    # Vertical gradient: wall color at top, gradually darker toward bottom
    for y in range(h):
        t = y / h  # 0 at top, 1 at bottom
        # Smooth S-curve: stays bright for wall, then darkens for floor
        # The "curve" point is where wall meets floor
        curve_point = 0.65  # 65% of image is wall, bottom 35% is floor gradient
        if t < curve_point:
            darken = 1.0  # wall: full brightness
        else:
            floor_t = (t - curve_point) / (1.0 - curve_point)
            darken = 1.0 - (1.0 - floor_darken) * (floor_t ** 0.8)  # gradual darkening
        backdrop[y, :] = backdrop_color * darken

    # Subtle radial vignette: brighter in center, slightly darker at edges
    cy, cx = h * 0.4, w * 0.5  # center slightly above middle (where light hits)
    Y, X = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((X - cx) / (w * 0.6)) ** 2 + ((Y - cy) / (h * 0.7)) ** 2)
    vignette = 1.0 - np.clip(dist, 0, 1) * 0.15  # subtle: max 15% darkening at edges
    backdrop *= vignette[:, :, np.newaxis]
    backdrop = np.clip(backdrop, 0, 255)

    # Composite subject over synthetic backdrop
    # Use the original subject mask (not dilated) for tight edge
    subject_f = subject_mask.astype(np.float32) / 255.0
    # Small feather for natural edge
    subject_f = cv2.GaussianBlur(subject_f, (5, 5), 0)
    subject_3ch = subject_f[:, :, np.newaxis]

    result = (subject_3ch * img.astype(np.float32) +
              (1 - subject_3ch) * backdrop)

    # Preserve contact shadows: near the subject's feet, blend some of the
    # original floor darkness to simulate natural shadow
    # Find the bottom of the subject
    subject_rows = np.where(np.any(subject_mask > 128, axis=1))[0]
    if len(subject_rows) > 0:
        subject_bottom = subject_rows[-1]
        shadow_zone_h = int(h * 0.08)  # shadow zone below subject

        for y in range(max(0, subject_bottom - 10), min(h, subject_bottom + shadow_zone_h)):
            if y < subject_bottom:
                shadow_strength = 0.3
            else:
                dist_from_bottom = (y - subject_bottom) / max(shadow_zone_h, 1)
                shadow_strength = max(0, 0.4 * (1 - dist_from_bottom ** 0.5))

            # Only apply shadow where original was darker than backdrop
            orig_row = gray[y, :]
            backdrop_brightness = np.mean(backdrop[y, :], axis=1) if backdrop.ndim == 3 else backdrop[y, :]
            backdrop_row_bright = np.mean(backdrop[y, :].reshape(-1, 3), axis=1)
            shadow_mask_row = (orig_row < backdrop_row_bright * 0.9) & (subject_f[y, :] < 0.5)
            shadow_mask_f = shadow_mask_row.astype(np.float32) * shadow_strength
            shadow_mask_f = cv2.GaussianBlur(shadow_mask_f.reshape(1, -1), (51, 1), 0).flatten()
            s3 = shadow_mask_f[:, np.newaxis]
            result[y] = result[y] * (1 - s3) + img[y].astype(np.float32) * s3

    return np.clip(result, 0, 255).astype(np.uint8)


HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Clean Backdrop</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a1a; color: #e0e0e0; }
.header { padding: 12px 20px; background: #222; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 20px; }
.header h1 { font-size: 18px; font-weight: 600; }
.main { display: flex; height: calc(100vh - 49px); }
.sidebar { width: 320px; background: #222; padding: 16px; overflow-y: auto; border-right: 1px solid #333; flex-shrink: 0; }
.viewer { flex: 1; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }
.viewer img { max-width: 100%; max-height: 100%; object-fit: contain; }
.section { margin-bottom: 20px; }
.section h3 { font-size: 13px; text-transform: uppercase; color: #888; margin-bottom: 10px; letter-spacing: 0.5px; }
.control { margin-bottom: 14px; }
.control label { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
.control label span { color: #aaa; }
.control input[type=range] { width: 100%; accent-color: #4a9eff; }
.control input[type=checkbox] { accent-color: #4a9eff; }
.checkbox-row { display: flex; align-items: center; gap: 8px; font-size: 13px; margin-bottom: 8px; }
.btn { padding: 10px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; width: 100%; margin-bottom: 8px; }
.btn-primary { background: #4a9eff; color: white; }
.btn-primary:hover { background: #3a8eef; }
.btn-secondary { background: #333; color: #ccc; }
.btn-secondary:hover { background: #444; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.status { font-size: 12px; color: #888; padding: 8px 0; }
.view-toggle { display: flex; gap: 4px; margin-bottom: 12px; }
.view-btn { flex: 1; padding: 6px; border: 1px solid #444; background: #2a2a2a; color: #aaa; border-radius: 4px; cursor: pointer; font-size: 12px; text-align: center; }
.view-btn.active { background: #4a9eff; color: white; border-color: #4a9eff; }
.drop-zone { border: 2px dashed #444; border-radius: 8px; padding: 40px; text-align: center; cursor: pointer; transition: border-color 0.2s; }
.drop-zone:hover, .drop-zone.dragover { border-color: #4a9eff; }
.drop-zone p { color: #888; font-size: 14px; }
.file-input { display: none; }
.slider-value { min-width: 28px; text-align: right; }
#loading { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.7); z-index: 100; align-items: center; justify-content: center; }
#loading.active { display: flex; }
#loading .spinner { font-size: 18px; color: white; }
</style>
</head>
<body>
<div class="header">
    <h1>Clean Backdrop</h1>
    <span class="status" id="imageInfo"></span>
</div>
<div class="main">
    <div class="sidebar">
        <div class="section">
            <h3>Image</h3>
            <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                <p>Drop image here or click to browse</p>
                <p style="font-size:11px;margin-top:8px;color:#666">Or paste a file path below</p>
            </div>
            <input type="file" class="file-input" id="fileInput" accept="image/*">
            <input type="text" id="pathInput" placeholder="/path/to/image.jpg"
                   style="width:100%;padding:8px;margin-top:8px;background:#2a2a2a;border:1px solid #444;border-radius:4px;color:#e0e0e0;font-size:12px;">
            <button class="btn btn-secondary" onclick="loadFromPath()" style="margin-top:4px">Load from path</button>
        </div>

        <div class="section">
            <h3>View</h3>
            <div class="view-toggle">
                <div class="view-btn active" data-view="original" onclick="setView('original')">Original</div>
                <div class="view-btn" data-view="mask" onclick="setView('mask')">Mask</div>
                <div class="view-btn" data-view="result" onclick="setView('result')">Result</div>
            </div>
        </div>

        <div class="section">
            <h3>Detection</h3>
            <div class="control">
                <label>Wrinkle sensitivity <span class="slider-value" id="sensVal">15</span></label>
                <input type="range" id="sensitivity" min="5" max="40" value="15" oninput="updateVal('sensVal', this.value)">
            </div>
            <div class="control">
                <label>Shadow strength <span class="slider-value" id="shadowVal">20</span></label>
                <input type="range" id="shadow" min="10" max="50" value="20" oninput="updateVal('shadowVal', this.value)">
            </div>
            <div class="control">
                <label>Scuff sensitivity <span class="slider-value" id="scuffVal">12</span></label>
                <input type="range" id="scuff" min="5" max="30" value="12" oninput="updateVal('scuffVal', this.value)">
            </div>
            <div class="checkbox-row">
                <input type="checkbox" id="darkBars" checked>
                <label for="darkBars">Detect dark bars/edges</label>
            </div>
            <div class="checkbox-row">
                <input type="checkbox" id="edgeDetect" checked>
                <label for="edgeDetect">Detect seam lines</label>
            </div>
            <div class="checkbox-row">
                <input type="checkbox" id="includeFloor">
                <label for="includeFloor">Include floor (seamless backdrop)</label>
            </div>
            <button class="btn btn-secondary" onclick="detectOnly()">Update Detection Preview</button>
        </div>

        <div class="section">
            <h3>Mode</h3>
            <div class="view-toggle">
                <div class="view-btn active" data-mode="retouch" onclick="setMode('retouch')">Retouch</div>
                <div class="view-btn" data-mode="replace" onclick="setMode('replace')">Replace Backdrop</div>
            </div>
            <div id="replaceOpts" style="display:none;margin-top:8px;">
                <div class="control">
                    <label>Floor darken <span class="slider-value" id="floorDarkenVal">75</span>%</label>
                    <input type="range" id="floorDarken" min="40" max="100" value="75" oninput="updateVal('floorDarkenVal', this.value)">
                </div>
            </div>
        </div>

        <div class="section">
            <h3>Process</h3>
            <button class="btn btn-primary" id="processBtn" onclick="process()" disabled>Run</button>
            <button class="btn btn-secondary" id="saveBtn" onclick="saveResult()" disabled>Save Full Resolution</button>
            <div class="status" id="status"></div>
        </div>
    </div>
    <div class="viewer" id="viewerArea">
        <img id="preview" src="" style="display:none">
        <div id="placeholder" style="color:#555;font-size:14px;">Drop an image anywhere or paste a path</div>
    </div>
</div>

<div id="loading"><div class="spinner">Processing...</div></div>

<script>
let currentView = 'original';
let currentMode = 'retouch';
let images = { original: null, mask: null, result: null };

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('[data-mode]').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
    document.getElementById('replaceOpts').style.display = mode === 'replace' ? 'block' : 'none';
}

function updateVal(id, val) { document.getElementById(id).textContent = val; }

function setView(view) {
    currentView = view;
    document.querySelectorAll('.view-btn').forEach(b => b.classList.toggle('active', b.dataset.view === view));
    if (images[view]) {
        document.getElementById('preview').src = 'data:image/jpeg;base64,' + images[view];
        document.getElementById('preview').style.display = 'block';
        document.getElementById('placeholder').style.display = 'none';
    }
}

function showLoading(msg) {
    document.querySelector('#loading .spinner').textContent = msg || 'Processing...';
    document.getElementById('loading').classList.add('active');
}
function hideLoading() { document.getElementById('loading').classList.remove('active'); }

function setStatus(msg) { document.getElementById('status').textContent = msg; }

// File upload
document.getElementById('fileInput').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        showLoading('Finding ' + file.name + ' on disk...');
        fetch('/find_and_load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: file.name})
        })
        .then(r => r.json())
        .then(data => {
            hideLoading();
            if (data.error) { alert(data.error); return; }
            images.original = data.image;
            images.mask = null;
            images.result = null;
            document.getElementById('imageInfo').textContent = data.info;
            document.getElementById('pathInput').value = data.original_path || '';
            document.getElementById('processBtn').disabled = false;
            setView('original');
        })
        .catch(err => { hideLoading(); alert('Error: ' + err); });
    }
});

const dropZone = document.getElementById('dropZone');
const viewerArea = document.getElementById('viewerArea');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

// Also allow dropping on the viewer area
viewerArea.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
viewerArea.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
viewerArea.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFileDrop(e.dataTransfer.files[0]);
    }
});
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFileDrop(e.dataTransfer.files[0]);
});

function handleFileDrop(file) {
    showLoading('Finding ' + file.name + ' on disk...');
    fetch('/find_and_load', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: file.name})
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.error) { alert(data.error); return; }
        images.original = data.image;
        images.mask = null;
        images.result = null;
        document.getElementById('imageInfo').textContent = data.info;
        document.getElementById('pathInput').value = data.original_path || '';
        document.getElementById('processBtn').disabled = false;
        setView('original');
    })
    .catch(err => { hideLoading(); alert('Error: ' + err); });
}

function uploadFile(file) {
    showLoading('Uploading and segmenting subject...');
    const formData = new FormData();
    formData.append('file', file);
    fetch('/upload', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            hideLoading();
            if (data.error) { alert(data.error); return; }
            images.original = data.image;
            images.mask = null;
            images.result = null;
            document.getElementById('imageInfo').textContent = data.info;
            document.getElementById('processBtn').disabled = false;
            setView('original');
        })
        .catch(err => { hideLoading(); alert('Error: ' + err); });
}

function loadFromPath() {
    const path = document.getElementById('pathInput').value.trim();
    if (!path) return;
    showLoading('Loading and segmenting subject...');
    fetch('/load_path', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path: path})
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.error) { alert(data.error); return; }
        images.original = data.image;
        images.mask = null;
        images.result = null;
        document.getElementById('imageInfo').textContent = data.info;
        document.getElementById('processBtn').disabled = false;
        setView('original');
    })
    .catch(err => { hideLoading(); alert('Error: ' + err); });
}

function detectOnly() {
    showLoading('Detecting blemishes...');
    const params = getParams();
    fetch('/detect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.error) { alert(data.error); return; }
        images.mask = data.mask_overlay;
        setStatus(data.info);
        setView('mask');
    })
    .catch(err => { hideLoading(); alert('Error: ' + err); });
}

function process() {
    showLoading('Running LaMa inpainting (may take a minute)...');
    const params = getParams();
    fetch('/process', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.error) { alert(data.error); return; }
        images.result = data.result;
        images.mask = data.mask_overlay;
        document.getElementById('saveBtn').disabled = false;
        setStatus(data.info);
        setView('result');
    })
    .catch(err => { hideLoading(); alert('Error: ' + err); });
}

function saveResult() {
    showLoading('Saving full resolution...');
    const params = getParams();
    fetch('/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.error) { alert(data.error); return; }
        setStatus('Saved: ' + data.path);
    })
    .catch(err => { hideLoading(); alert('Error: ' + err); });
}

function getParams() {
    return {
        mode: currentMode,
        sensitivity: parseInt(document.getElementById('sensitivity').value),
        shadow: parseInt(document.getElementById('shadow').value),
        scuff: parseInt(document.getElementById('scuff').value),
        dark_bars: document.getElementById('darkBars').checked,
        edges: document.getElementById('edgeDetect').checked,
        include_floor: document.getElementById('includeFloor').checked,
        floor_darken: parseInt(document.getElementById('floorDarken').value) / 100,
    };
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


def find_file_on_disk(filename):
    """Search common photo directories for the original file by name."""
    search_roots = [
        "/mnt/c/Users",
        "/mnt/d/Photos",
        "/mnt/d",
        "/mnt/e",
    ]
    import subprocess
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        try:
            result = subprocess.run(
                ["find", root, "-name", filename, "-type", "f", "-maxdepth", "8"],
                capture_output=True, text=True, timeout=10
            )
            matches = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            if matches:
                return matches[0]  # Return first match
        except:
            continue
    return None


@app.route('/find_and_load', methods=['POST'])
def find_and_load():
    data = request.json
    filename = data.get('filename', '')
    if not filename:
        return jsonify({"error": "No filename provided"})

    path = find_file_on_disk(filename)
    if not path:
        return jsonify({"error": f"Could not find '{filename}' on disk. Try pasting the full path instead."})

    result = _load_image(path)
    resp = result.get_json()
    resp['original_path'] = path
    return jsonify(resp)


@app.route('/load_path', methods=['POST'])
def load_path():
    data = request.json
    path = data.get('path', '').strip().strip('"')

    # Convert Windows paths to WSL
    import re
    if re.match(r'^[A-Za-z]:\\', path):
        drive = path[0].lower()
        path = '/mnt/' + drive + path[2:].replace('\\', '/')

    if not os.path.exists(path):
        return jsonify({"error": f"File not found: {path}"})

    result = _load_image(path)
    data = result.get_json()
    data['original_path'] = path
    return jsonify(data)


def _load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return jsonify({"error": "Could not read image"})

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]

    # Store ICC profile
    try:
        pil_img = Image.open(path)
        state["icc_profile"] = pil_img.info.get('icc_profile')
    except:
        state["icc_profile"] = None

    state["img"] = img
    state["img_path"] = path

    # Segment subject
    pil_img = Image.open(path).convert("RGB")
    session = new_session("u2net_human_seg")
    subject_mask = np.array(remove(pil_img, session=session, only_mask=True))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    subject_mask = cv2.morphologyEx(subject_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    subject_mask = cv2.morphologyEx(subject_mask, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    state["subject_mask"] = subject_mask

    protect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    protected = cv2.dilate(subject_mask, protect_kernel, iterations=2)
    state["bg_mask"] = (protected < 128).astype(np.uint8)
    state["floor_mask"] = detect_floor_boundary(img, state["bg_mask"])

    # Preview
    preview, _ = resize_for_preview(img, state["preview_size"])
    b64 = img_to_b64(preview)

    floor_info = "floor detected" if np.sum(state["floor_mask"]) > 0 else "single surface"
    return jsonify({
        "image": b64,
        "info": f"{w}x{h} | {floor_info}"
    })


@app.route('/detect', methods=['POST'])
def detect():
    if state["img"] is None:
        return jsonify({"error": "No image loaded"})

    params = request.json
    floor = state["floor_mask"] if not params.get("include_floor", False) else np.zeros_like(state["floor_mask"])
    mask = detect_blemishes(
        state["img"], state["bg_mask"], floor,
        sensitivity=params.get("sensitivity", 15),
        shadow_strength=params.get("shadow", 20),
        scuff_strength=params.get("scuff", 12),
        detect_dark_bars=params.get("dark_bars", True),
        detect_edges=params.get("edges", True),
    )
    mask[state["subject_mask"] > 128] = 0  # protect subject

    h, w = state["img"].shape[:2]
    coverage = np.sum(mask > 0) / (h * w) * 100

    # Create overlay visualization
    overlay = state["img"].copy()
    overlay[mask > 128] = [0, 165, 255]  # orange
    blended = cv2.addWeighted(state["img"], 0.5, overlay, 0.5, 0)

    preview, _ = resize_for_preview(blended, state["preview_size"])
    return jsonify({
        "mask_overlay": img_to_b64(preview),
        "info": f"Detected {coverage:.1f}% blemishes"
    })


@app.route('/process', methods=['POST'])
def process():
    if state["img"] is None:
        return jsonify({"error": "No image loaded"})

    params = request.json
    img = state["img"]
    h, w = img.shape[:2]

    # Replace mode: synthetic seamless backdrop
    if params.get("mode") == "replace":
        final = generate_seamless_backdrop(
            img, state["bg_mask"], state["subject_mask"], state["floor_mask"],
            floor_darken=params.get("floor_darken", 0.75),
        )
        state["last_result"] = final
        result_preview, _ = resize_for_preview(final, state["preview_size"])
        return jsonify({
            "result": img_to_b64(result_preview),
            "mask_overlay": img_to_b64(result_preview),
            "info": "Backdrop replaced with synthetic seamless"
        })

    # Detect
    mask = detect_blemishes(
        img, state["bg_mask"], state["floor_mask"],
        sensitivity=params.get("sensitivity", 15),
        shadow_strength=params.get("shadow", 20),
        scuff_strength=params.get("scuff", 12),
        detect_dark_bars=params.get("dark_bars", True),
        detect_edges=params.get("edges", True),
    )
    mask[state["subject_mask"] > 128] = 0
    coverage = np.sum(mask > 0) / (h * w) * 100

    if coverage < 0.1:
        return jsonify({"error": "No blemishes detected with current settings"})

    # Iterative LaMa inpainting:
    # Large masks smudge at low res, so we do multiple passes:
    # Pass 1: small blemishes (dilated less) at higher res
    # Pass 2: remaining larger areas
    model, device = get_lama_model()
    current = img.copy()

    # Work at as high a resolution as VRAM allows
    max_dim = 3072  # higher res = cleaner result
    scale = min(max_dim / max(h, w), 1.0)

    # If mask coverage is very high (>25%), do two passes
    if coverage > 25:
        # Pass 1: just wrinkles and small marks (erode mask to shrink to cores)
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        small_mask = cv2.erode(mask, small_kernel, iterations=2)
        small_mask = cv2.dilate(small_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

        if np.sum(small_mask > 0) > 0:
            if scale < 1.0:
                sh, sw = int(h * scale), int(w * scale)
                r1 = inpaint_lama(cv2.resize(current, (sw, sh), interpolation=cv2.INTER_AREA),
                                   cv2.resize(small_mask, (sw, sh), interpolation=cv2.INTER_NEAREST),
                                   model, device)
                r1 = cv2.resize(r1, (w, h), interpolation=cv2.INTER_LANCZOS4)
            else:
                r1 = inpaint_lama(current, small_mask, model, device)
            m1 = (small_mask > 128).astype(np.float32)
            m1 = cv2.GaussianBlur(m1, (7, 7), 0)[:, :, np.newaxis]
            current = (m1 * r1.astype(np.float32) + (1 - m1) * current.astype(np.float32))
            current = np.clip(current, 0, 255).astype(np.uint8)

        # Pass 2: remaining areas (shadows, bars) on the now-cleaner image
        remaining = mask.copy()
        remaining[small_mask > 128] = 0
        if np.sum(remaining > 0) > 0:
            if scale < 1.0:
                sh, sw = int(h * scale), int(w * scale)
                r2 = inpaint_lama(cv2.resize(current, (sw, sh), interpolation=cv2.INTER_AREA),
                                   cv2.resize(remaining, (sw, sh), interpolation=cv2.INTER_NEAREST),
                                   model, device)
                r2 = cv2.resize(r2, (w, h), interpolation=cv2.INTER_LANCZOS4)
            else:
                r2 = inpaint_lama(current, remaining, model, device)
            m2 = (remaining > 128).astype(np.float32)
            m2 = cv2.GaussianBlur(m2, (7, 7), 0)[:, :, np.newaxis]
            current = (m2 * r2.astype(np.float32) + (1 - m2) * current.astype(np.float32))
            current = np.clip(current, 0, 255).astype(np.uint8)
    else:
        # Single pass for small coverage
        if scale < 1.0:
            sh, sw = int(h * scale), int(w * scale)
            lama_result = inpaint_lama(cv2.resize(current, (sw, sh), interpolation=cv2.INTER_AREA),
                                        cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST),
                                        model, device)
            lama_result = cv2.resize(lama_result, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            lama_result = inpaint_lama(current, mask, model, device)
        mask_f = (mask > 128).astype(np.float32)
        mask_f = cv2.GaussianBlur(mask_f, (7, 7), 0)[:, :, np.newaxis]
        current = (mask_f * lama_result.astype(np.float32) +
                   (1 - mask_f) * current.astype(np.float32))
        current = np.clip(current, 0, 255).astype(np.uint8)

    # Light polish
    smoothed = current.copy()
    for _ in range(2):
        smoothed = cv2.bilateralFilter(smoothed, 9, 25, 25)
    bg_f = state["bg_mask"].astype(np.float32)
    if np.sum(state["floor_mask"]) > 0:
        bg_f[state["floor_mask"] > 0] = 0
    bg_f = cv2.GaussianBlur(bg_f, (11, 11), 0)[:, :, np.newaxis]
    current = (bg_f * smoothed.astype(np.float32) + (1 - bg_f) * current.astype(np.float32))
    current = np.clip(current, 0, 255).astype(np.uint8)

    # Final composite
    subj_f = state["subject_mask"].astype(np.float32) / 255.0
    subj_f = cv2.GaussianBlur(subj_f, (3, 3), 0)[:, :, np.newaxis]
    final = (subj_f * img.astype(np.float32) + (1 - subj_f) * current.astype(np.float32))
    final = np.clip(final, 0, 255).astype(np.uint8)

    state["last_result"] = final

    # Preview images
    result_preview, _ = resize_for_preview(final, state["preview_size"])
    overlay = img.copy()
    overlay[mask > 128] = [0, 165, 255]
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    mask_preview, _ = resize_for_preview(blended, state["preview_size"])

    return jsonify({
        "result": img_to_b64(result_preview),
        "mask_overlay": img_to_b64(mask_preview),
        "info": f"Healed {coverage:.1f}% blemishes"
    })


@app.route('/save', methods=['POST'])
def save():
    if state.get("last_result") is None:
        return jsonify({"error": "No result to save. Run processing first."})

    # Determine output path
    input_path = state["img_path"]
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_clean{ext}"

    final = state["last_result"]
    final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(final_rgb)

    save_kwargs = {}
    if state["icc_profile"]:
        save_kwargs["icc_profile"] = state["icc_profile"]

    try:
        original_pil = Image.open(input_path)
        exif = original_pil.info.get("exif")
        if exif:
            save_kwargs["exif"] = exif
    except:
        pass

    ext_out = ext.lower()
    if ext_out in ('.tif', '.tiff'):
        pil.save(output_path, **save_kwargs)
    elif ext_out == '.png':
        pil.save(output_path, **save_kwargs)
    else:
        pil.save(output_path, quality=98, subsampling=0, **save_kwargs)

    return jsonify({"path": output_path})


if __name__ == '__main__':
    print("\n  Clean Backdrop Web UI")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
