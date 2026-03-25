#!/usr/bin/env python3
"""
Clean Backdrop v2 - Three-technique approach:
1. LaMa: small blemishes only (marks, wrinkles, scuffs)
2. Shadow lift: brighten cast shadows to match surrounding wall
3. Floor replace: synthetic gradient from wall color

Each has independent controls. Preview shows all three layered.
"""

import os
import io
import base64
import json
import re
import subprocess
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

state = {
    "img": None,
    "img_path": None,
    "subject_mask": None,
    "bg_mask": None,
    "floor_mask": None,
    "floor_start_row": None,
    "lama_model": None,
    "device": None,
    "preview_size": 1200,
    "icc_profile": None,
    "last_result": None,
}


def download_lama():
    if os.path.exists(LAMA_MODEL_PATH):
        return
    os.makedirs(LAMA_MODEL_DIR, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(LAMA_MODEL_URL, LAMA_MODEL_PATH)


def get_lama():
    if state["lama_model"] is None:
        download_lama()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state["device"] = device
        state["lama_model"] = torch.jit.load(LAMA_MODEL_PATH, map_location=device)
        state["lama_model"].eval()
    return state["lama_model"], state["device"]


def pad8(img):
    h, w = img.shape[:2]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph == 0 and pw == 0:
        return img, (0, 0)
    if img.ndim == 3:
        return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect'), (ph, pw)
    return np.pad(img, ((0, ph), (0, pw)), mode='reflect'), (ph, pw)


def run_lama(image, mask, model, device):
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mf = (mask > 128).astype(np.float32)
    rgb, (ph, pw) = pad8(rgb)
    mf, _ = pad8(mf)
    with torch.no_grad():
        out = model(
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device),
            torch.from_numpy(mf).unsqueeze(0).unsqueeze(0).to(device),
        )
    out = out[0].permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    if ph > 0: out = out[:-ph]
    if pw > 0: out = out[:, :-pw]
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def preview_resize(img):
    h, w = img.shape[:2]
    s = min(state["preview_size"] / max(h, w), 1.0)
    if s < 1.0:
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def to_b64(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    kw = {}
    if state.get("icc_profile"):
        kw["icc_profile"] = state["icc_profile"]
    pil.save(buf, format='JPEG', quality=92, **kw)
    return base64.b64encode(buf.getvalue()).decode()


def find_file(filename):
    for root in ["/mnt/c/Users", "/mnt/d/Photos", "/mnt/d", "/mnt/e"]:
        if not os.path.isdir(root):
            continue
        try:
            r = subprocess.run(["find", root, "-name", filename, "-type", "f", "-maxdepth", "8"],
                               capture_output=True, text=True, timeout=10)
            matches = [l.strip() for l in r.stdout.strip().split('\n') if l.strip()]
            if matches:
                return matches[0]
        except:
            continue
    return None


def detect_floor(img, bg_mask):
    h, w = img.shape[:2]
    floor_mask = np.zeros((h, w), dtype=np.uint8)
    floor_start = h

    wt, wb = h // 4, h // 2
    ws = bg_mask[wt:wb, :] > 0
    if np.sum(ws) < 100:
        return floor_mask, floor_start
    wall_color = np.median(img[wt:wb][ws], axis=0)

    ft = int(h * 0.85)
    fs = bg_mask[ft:, :] > 0
    if np.sum(fs) < 100:
        return floor_mask, floor_start
    floor_color = np.median(img[ft:][fs], axis=0)

    diff = np.sqrt(np.sum((wall_color.astype(float) - floor_color.astype(float)) ** 2))
    if diff < 60:
        return floor_mask, floor_start

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bh = gray[h // 2:, :]
    bm = bg_mask[h // 2:, :]
    means = []
    for r in range(bh.shape[0]):
        px = bh[r][bm[r] > 0]
        means.append(np.mean(px) if len(px) > 0 else 0)
    means = np.array(means)
    if len(means) > 20:
        sm = cv2.GaussianBlur(means.reshape(-1, 1), (1, 21), 0).flatten()
        grad = np.diff(sm)
        if len(grad) > 0 and np.min(grad) < -0.5:
            cands = np.where(grad < np.min(grad) * 0.3)[0]
            if len(cands) > 0:
                floor_start = h // 2 + cands[0]
                floor_mask[floor_start:, :] = bg_mask[floor_start:, :]

    return floor_mask, floor_start


# ── Technique 1: Detect small blemishes for LaMa ──
def detect_marks(img, bg_mask, floor_mask, sensitivity=15):
    """Only wrinkles, scuffs, marks - NOT shadows, NOT floor."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    work = bg_mask.copy()
    work[floor_mask > 0] = 0

    # High-freq wrinkles
    lo = cv2.GaussianBlur(gray, (0, 0), sigmaX=8)
    hi = np.abs(gray - lo)
    marks = ((hi > sensitivity) & (work > 0)).astype(np.uint8)

    # Scuffs (local dark spots)
    local = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
    scuffs = ((local - gray > sensitivity * 0.8) & (work > 0)).astype(np.uint8)
    bright = ((gray - local > sensitivity * 0.8) & (work > 0)).astype(np.uint8)

    # Seam lines
    edges = cv2.Canny(gray.astype(np.uint8), 15, 50)
    seams = ((edges > 0) & (work > 0)).astype(np.uint8)
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    vseams = ((sobelx > 15) & (work > 0)).astype(np.uint8)

    combined = np.clip(marks + scuffs + bright + seams + vseams, 0, 1).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.dilate(combined, k, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    combined[floor_mask > 0] = 0
    return combined * 255


# ── Technique 2: Shadow lift ──
def compute_shadow_lift(img, bg_mask, floor_mask, subject_mask, strength=0.7):
    """
    Lift shadows by blending toward clean wall color.
    Compares every wall pixel directly to the clean wall brightness -
    NOT to a blur that includes the shadow.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    work = bg_mask.copy()
    work[floor_mask > 0] = 0

    if np.sum(work) < 100:
        return img.copy()

    # Find the clean wall color from the brightest background pixels
    wall_pixels = img[work > 0]
    brightness = np.mean(wall_pixels, axis=1)
    bright_pixels = wall_pixels[brightness > np.percentile(brightness, 80)]
    if len(bright_pixels) < 10:
        return img.copy()
    clean_wall = np.median(bright_pixels, axis=0).astype(np.float32)  # BGR
    clean_brightness = np.mean(clean_wall)

    # Shadow amount: how much darker is each pixel compared to clean wall
    # This is a direct comparison, not relative to a blur
    darkness = clean_brightness - gray
    max_shadow_depth = clean_brightness * 0.5  # shadows up to 50% darker than clean wall
    shadow_amount = np.clip(darkness / max(max_shadow_depth, 1), 0, 1)
    shadow_amount = shadow_amount * strength

    # Only on wall background, not subject or floor
    shadow_amount[work == 0] = 0

    # Smooth to avoid harsh edges
    shadow_amount = cv2.GaussianBlur(shadow_amount, (31, 31), 0)

    # Blend each pixel toward clean wall color
    result = img.astype(np.float32)
    blend = shadow_amount[:, :, np.newaxis]
    clean_bg = np.full_like(result, clean_wall)
    result = result * (1 - blend) + clean_bg * blend

    # Subject stays original (sharp edge)
    subj_f = (subject_mask > 128).astype(np.float32)
    subj_f = cv2.GaussianBlur(subj_f, (3, 3), 0)[:, :, np.newaxis]
    result = subj_f * img.astype(np.float32) + (1 - subj_f) * result

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Technique 3: Floor replacement ──
def replace_floor(img, bg_mask, subject_mask, floor_mask, floor_start, darken=0.8):
    """Replace the floor area with a gradient from wall color."""
    h, w = img.shape[:2]
    if np.sum(floor_mask) == 0:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Sample wall color from clean area just above floor
    wall_zone = bg_mask.copy()
    wall_zone[floor_mask > 0] = 0
    wall_zone[:max(0, floor_start - int(h * 0.15)), :] = 0  # just above the floor
    wall_pixels = img[wall_zone > 0]
    if len(wall_pixels) < 50:
        wall_zone = bg_mask.copy()
        wall_zone[floor_mask > 0] = 0
        wall_pixels = img[wall_zone > 0]
    if len(wall_pixels) < 50:
        return img

    brightness = np.mean(wall_pixels, axis=1)
    clean_wall = np.median(wall_pixels[brightness > np.percentile(brightness, 60)], axis=0).astype(np.float32)

    # Build gradient for floor area
    result = img.astype(np.float32).copy()
    floor_h = h - floor_start
    if floor_h <= 0:
        return img

    for y in range(floor_start, h):
        t = (y - floor_start) / max(floor_h, 1)
        # Smooth gradient: starts at wall color, darkens toward bottom
        color = clean_wall * (1.0 - (1.0 - darken) * (t ** 0.6))
        # Only replace floor background pixels
        floor_row = (floor_mask[y, :] > 0) & (subject_mask[y, :] < 128)
        result[y, floor_row] = color

    # Also fill the baseboard/bar zone (dark area right at transition)
    bar_zone = max(0, floor_start - int(h * 0.02))
    for y in range(bar_zone, min(floor_start + int(h * 0.02), h)):
        t = max(0, (y - floor_start)) / max(floor_h, 1)
        color = clean_wall * (1.0 - (1.0 - darken) * max(0, t ** 0.6))
        dark_here = (gray[y, :] < np.mean(clean_wall) * 0.6)
        replace_px = dark_here & (bg_mask[y, :] > 0) & (subject_mask[y, :] < 128)
        result[y, replace_px] = color

    # Feather the transition
    result_u8 = np.clip(result, 0, 255).astype(np.uint8)
    blended = cv2.GaussianBlur(result_u8, (1, 31), 0)

    # Only apply blur at the transition zone
    trans_mask = np.zeros((h, w), dtype=np.float32)
    trans_top = max(0, floor_start - 15)
    trans_bot = min(h, floor_start + 15)
    for y in range(trans_top, trans_bot):
        trans_mask[y, :] = 1.0 - abs(y - floor_start) / 15.0
    trans_mask = np.clip(trans_mask, 0, 1)
    trans_3 = trans_mask[:, :, np.newaxis]
    result_u8 = (trans_3 * blended.astype(np.float32) +
                  (1 - trans_3) * result_u8.astype(np.float32))

    # Protect subject
    subj_f = (subject_mask > 128).astype(np.float32)
    subj_f = cv2.GaussianBlur(subj_f, (3, 3), 0)[:, :, np.newaxis]
    final = subj_f * img.astype(np.float32) + (1 - subj_f) * result_u8
    return np.clip(final, 0, 255).astype(np.uint8)


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
.sidebar { width: 340px; background: #222; padding: 16px; overflow-y: auto; border-right: 1px solid #333; flex-shrink: 0; }
.viewer { flex: 1; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }
.viewer img { max-width: 100%; max-height: 100%; object-fit: contain; }
.section { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #333; }
.section:last-child { border-bottom: none; }
.section h3 { font-size: 11px; text-transform: uppercase; color: #666; margin-bottom: 8px; letter-spacing: 1px; }
.control { margin-bottom: 10px; }
.control label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px; }
.control label span { color: #888; }
input[type=range] { width: 100%; accent-color: #4a9eff; }
input[type=checkbox] { accent-color: #4a9eff; }
.check-row { display: flex; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 6px; }
.btn { padding: 8px 12px; border: none; border-radius: 5px; cursor: pointer; font-size: 12px; font-weight: 500; width: 100%; margin-bottom: 6px; }
.btn-blue { background: #4a9eff; color: white; }
.btn-blue:hover { background: #3a8eef; }
.btn-grey { background: #333; color: #bbb; }
.btn-grey:hover { background: #444; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.status { font-size: 11px; color: #777; padding: 4px 0; }
.tabs { display: flex; gap: 3px; margin-bottom: 8px; }
.tab { flex: 1; padding: 5px; border: 1px solid #444; background: #2a2a2a; color: #888; border-radius: 3px; cursor: pointer; font-size: 11px; text-align: center; }
.tab.on { background: #4a9eff; color: white; border-color: #4a9eff; }
.drop-zone { border: 2px dashed #444; border-radius: 8px; padding: 30px; text-align: center; cursor: pointer; }
.drop-zone:hover, .drop-zone.dragover { border-color: #4a9eff; }
.drop-zone p { color: #666; font-size: 13px; }
input[type=text] { width:100%; padding:6px 8px; background:#2a2a2a; border:1px solid #444; border-radius:4px; color:#e0e0e0; font-size:11px; }
#loading { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:100; align-items:center; justify-content:center; }
#loading.on { display:flex; }
.technique-box { background: #2a2a2a; border-radius: 6px; padding: 10px; margin-bottom: 8px; }
.technique-box h4 { font-size: 12px; margin-bottom: 6px; color: #ccc; }
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
            <div class="drop-zone" id="dropZone" onclick="document.getElementById('fi').click()">
                <p>Drop image or click to browse</p>
            </div>
            <input type="file" id="fi" accept="image/*" style="display:none">
            <input type="text" id="pathInput" placeholder="Paste file path..." style="margin-top:6px">
            <button class="btn btn-grey" onclick="loadPath()" style="margin-top:4px">Load path</button>
        </div>

        <div class="section">
            <h3>View</h3>
            <div class="tabs">
                <div class="tab on" data-v="original" onclick="view('original')">Original</div>
                <div class="tab" data-v="preview" onclick="view('preview')">Preview</div>
                <div class="tab" data-v="marks" onclick="view('marks')">Marks</div>
                <div class="tab" data-v="shadows" onclick="view('shadows')">Shadows</div>
            </div>
        </div>

        <div class="section">
            <h3>1. Marks &amp; Blemishes (LaMa)</h3>
            <div class="check-row"><input type="checkbox" id="doMarks" checked><label for="doMarks">Enable</label></div>
            <div class="control">
                <label>Sensitivity <span id="sensV">10</span></label>
                <input type="range" id="sens" min="3" max="30" value="10" oninput="document.getElementById('sensV').textContent=this.value">
            </div>
        </div>

        <div class="section">
            <h3>2. Shadow Lift</h3>
            <div class="check-row"><input type="checkbox" id="doShadow" checked><label for="doShadow">Enable</label></div>
            <div class="control">
                <label>Lift strength <span id="liftV">70</span>%</label>
                <input type="range" id="lift" min="0" max="100" value="70" oninput="document.getElementById('liftV').textContent=this.value">
            </div>
        </div>

        <!-- Floor replacement removed - keeping original floor -->


        <div class="section">
            <h3>Process</h3>
            <button class="btn btn-blue" id="prevBtn" onclick="doPreview()" disabled>Preview All</button>
            <button class="btn btn-blue" id="runBtn" onclick="doProcess()" disabled>Apply (Full Res)</button>
            <button class="btn btn-grey" id="saveBtn" onclick="doSave()" disabled>Save</button>
            <div class="status" id="status"></div>
        </div>
    </div>
    <div class="viewer" id="viewer">
        <img id="img" src="" style="display:none">
        <div id="ph" style="color:#555;font-size:14px;">Drop an image anywhere</div>
    </div>
</div>
<div id="loading"><div style="color:white;font-size:16px;" id="loadMsg">Processing...</div></div>

<script>
let imgs = {};
let curView = 'original';

function view(v) {
    curView = v;
    document.querySelectorAll('[data-v]').forEach(t => t.classList.toggle('on', t.dataset.v === v));
    if (imgs[v]) { document.getElementById('img').src = 'data:image/jpeg;base64,' + imgs[v]; document.getElementById('img').style.display='block'; document.getElementById('ph').style.display='none'; }
}

function loading(msg) { document.getElementById('loadMsg').textContent=msg||'Processing...'; document.getElementById('loading').classList.add('on'); }
function loaded() { document.getElementById('loading').classList.remove('on'); }
function status(s) { document.getElementById('status').textContent=s; }

function params() { return {
    do_marks: document.getElementById('doMarks').checked,
    sensitivity: +document.getElementById('sens').value,
    do_shadow: document.getElementById('doShadow').checked,
    lift: +document.getElementById('lift').value / 100,
};}

// Drop handling
function handleDrop(file) {
    loading('Finding ' + file.name + '...');
    fetch('/find_load', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:file.name}) })
    .then(r=>r.json()).then(d => { loaded(); if(d.error){alert(d.error);return;}
        imgs = {original: d.image}; document.getElementById('imageInfo').textContent=d.info;
        document.getElementById('pathInput').value=d.path||'';
        document.getElementById('prevBtn').disabled=false; document.getElementById('runBtn').disabled=false;
        view('original');
    }).catch(e=>{loaded();alert(e);});
}

const dz = document.getElementById('dropZone');
const vw = document.getElementById('viewer');
[dz, vw].forEach(el => {
    el.addEventListener('dragover', e=>{e.preventDefault();dz.classList.add('dragover');});
    el.addEventListener('dragleave', ()=>dz.classList.remove('dragover'));
    el.addEventListener('drop', e=>{e.preventDefault();dz.classList.remove('dragover');if(e.dataTransfer.files.length)handleDrop(e.dataTransfer.files[0]);});
});
document.getElementById('fi').addEventListener('change', e=>{if(e.target.files.length)handleDrop(e.target.files[0]);});

function loadPath() {
    let p = document.getElementById('pathInput').value.trim().replace(/"/g,'');
    if(!p) return;
    loading('Loading...');
    fetch('/load_path', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path:p})})
    .then(r=>r.json()).then(d=>{loaded();if(d.error){alert(d.error);return;}
        imgs={original:d.image}; document.getElementById('imageInfo').textContent=d.info;
        document.getElementById('pathInput').value=d.path||p;
        document.getElementById('prevBtn').disabled=false; document.getElementById('runBtn').disabled=false;
        view('original');
    }).catch(e=>{loaded();alert(e);});
}

function doPreview() {
    loading('Building preview...');
    fetch('/preview', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params())})
    .then(r=>r.json()).then(d=>{loaded();if(d.error){alert(d.error);return;}
        imgs.preview=d.preview; imgs.marks=d.marks; imgs.shadows=d.shadows;
        status(d.info); view('preview');
    }).catch(e=>{loaded();alert(e);});
}

function doProcess() {
    loading('Applying full resolution (may take a minute)...');
    fetch('/process', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params())})
    .then(r=>r.json()).then(d=>{loaded();if(d.error){alert(d.error);return;}
        imgs.preview=d.preview; document.getElementById('saveBtn').disabled=false;
        status(d.info); view('preview');
    }).catch(e=>{loaded();alert(e);});
}

function doSave() {
    loading('Saving...');
    fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({})})
    .then(r=>r.json()).then(d=>{loaded();if(d.error){alert(d.error);return;} status('Saved: '+d.path);})
    .catch(e=>{loaded();alert(e);});
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/find_load', methods=['POST'])
def find_load():
    fn = request.json.get('filename', '')
    if not fn:
        return jsonify({"error": "No filename"})
    path = find_file(fn)
    if not path:
        return jsonify({"error": f"'{fn}' not found on disk. Paste the full path."})
    return _load(path)


@app.route('/load_path', methods=['POST'])
def load_path():
    path = request.json.get('path', '').strip().strip('"')
    m = re.match(r'^([A-Za-z]):\\', path)
    if m:
        path = '/mnt/' + m.group(1).lower() + path[2:].replace('\\', '/')
    if not os.path.exists(path):
        return jsonify({"error": f"Not found: {path}"})
    return _load(path)


def _load(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return jsonify({"error": "Can't read image"})
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    try:
        pil = Image.open(path)
        state["icc_profile"] = pil.info.get('icc_profile')
    except:
        state["icc_profile"] = None

    state["img"] = img
    state["img_path"] = path

    # Segment
    pil_img = Image.open(path).convert("RGB")
    session = new_session("u2net_human_seg")
    sm = np.array(remove(pil_img, session=session, only_mask=True))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, k, iterations=3)
    sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    state["subject_mask"] = sm

    pk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    protected = cv2.dilate(sm, pk, iterations=2)
    state["bg_mask"] = (protected < 128).astype(np.uint8)

    fm, fs = detect_floor(img, state["bg_mask"])
    state["floor_mask"] = fm
    state["floor_start_row"] = fs

    preview = preview_resize(img)
    floor_info = f"floor at row {fs}" if np.sum(fm) > 0 else "no floor boundary"
    return jsonify({"image": to_b64(preview), "info": f"{w}x{h} | {floor_info}", "path": path})


@app.route('/preview', methods=['POST'])
def preview():
    if state["img"] is None:
        return jsonify({"error": "No image loaded"})

    p = request.json
    img = state["img"]
    h, w = img.shape[:2]
    current = img.copy()
    info_parts = []

    # Technique 2: Shadow lift (do before marks so LaMa sees lifted image)
    shadow_vis = None
    if p.get("do_shadow"):
        current = compute_shadow_lift(current, state["bg_mask"], state["floor_mask"],
                                       state["subject_mask"], strength=p.get("lift", 0.7))
        # Visualization: diff between original and lifted
        diff = np.abs(current.astype(np.float32) - img.astype(np.float32))
        diff_vis = img.copy()
        changed = np.mean(diff, axis=2) > 5
        diff_vis[changed] = [0, 165, 255]
        shadow_vis = preview_resize(cv2.addWeighted(img, 0.5, diff_vis, 0.5, 0))
        info_parts.append("shadows lifted")

    # Technique 1: Marks detection (show mask, don't inpaint yet for preview speed)
    marks_vis = None
    if p.get("do_marks"):
        marks_mask = detect_marks(current, state["bg_mask"], state["floor_mask"],
                                   sensitivity=p.get("sensitivity", 15))
        marks_mask[state["subject_mask"] > 128] = 0
        cov = np.sum(marks_mask > 0) / (h * w) * 100
        overlay = current.copy()
        overlay[marks_mask > 128] = [0, 165, 255]
        marks_vis = preview_resize(cv2.addWeighted(current, 0.5, overlay, 0.5, 0))
        info_parts.append(f"marks: {cov:.1f}%")

    # Final subject composite
    sf = state["subject_mask"].astype(np.float32) / 255.0
    sf = cv2.GaussianBlur(sf, (3, 3), 0)[:, :, np.newaxis]
    final = (sf * img.astype(np.float32) + (1 - sf) * current.astype(np.float32))
    final = np.clip(final, 0, 255).astype(np.uint8)

    return jsonify({
        "preview": to_b64(preview_resize(final)),
        "marks": to_b64(marks_vis) if marks_vis is not None else to_b64(preview_resize(img)),
        "shadows": to_b64(shadow_vis) if shadow_vis is not None else to_b64(preview_resize(img)),
        "info": " | ".join(info_parts) if info_parts else "No techniques enabled",
    })


@app.route('/process', methods=['POST'])
def process_full():
    if state["img"] is None:
        return jsonify({"error": "No image loaded"})

    p = request.json
    img = state["img"]
    h, w = img.shape[:2]
    current = img.copy()
    info_parts = []

    # Technique 2: Shadow lift (same as preview)
    if p.get("do_shadow"):
        current = compute_shadow_lift(current, state["bg_mask"], state["floor_mask"],
                                       state["subject_mask"], strength=p.get("lift", 0.7))
        info_parts.append("shadows lifted")

    # Subject composite (sharp - minimal feather)
    sf = state["subject_mask"].astype(np.float32) / 255.0
    sf = cv2.GaussianBlur(sf, (3, 3), 0)[:, :, np.newaxis]
    final = (sf * img.astype(np.float32) + (1 - sf) * current.astype(np.float32))
    final = np.clip(final, 0, 255).astype(np.uint8)

    # Technique 1: LaMa on marks ONLY if enabled - very conservative
    if p.get("do_marks"):
        marks_mask = detect_marks(final, state["bg_mask"], state["floor_mask"],
                                   sensitivity=p.get("sensitivity", 10))
        marks_mask[state["subject_mask"] > 128] = 0
        cov = np.sum(marks_mask > 0) / (h * w) * 100

        # Hard cap at 5% - LaMa should only touch tiny isolated marks
        if cov > 5:
            marks_mask = detect_marks(final, state["bg_mask"], state["floor_mask"],
                                       sensitivity=max(p.get("sensitivity", 10) * 3, 25))
            marks_mask[state["subject_mask"] > 128] = 0
            cov = np.sum(marks_mask > 0) / (h * w) * 100

        if cov > 0.1 and cov <= 5:
            print(f"  LaMa on {cov:.1f}% marks")
            model, device = get_lama()
            max_dim = 3072
            scale = min(max_dim / max(h, w), 1.0)
            if scale < 1.0:
                sh, sw = int(h * scale), int(w * scale)
                lr = run_lama(cv2.resize(final, (sw, sh), interpolation=cv2.INTER_AREA),
                              cv2.resize(marks_mask, (sw, sh), interpolation=cv2.INTER_NEAREST), model, device)
                lr = cv2.resize(lr, (w, h), interpolation=cv2.INTER_LANCZOS4)
            else:
                lr = run_lama(final, marks_mask, model, device)
            mf = (marks_mask > 128).astype(np.float32)
            mf = cv2.GaussianBlur(mf, (3, 3), 0)[:, :, np.newaxis]
            final = (mf * lr.astype(np.float32) + (1 - mf) * final.astype(np.float32))
            final = np.clip(final, 0, 255).astype(np.uint8)
            info_parts.append(f"marks healed ({cov:.1f}%)")
        else:
            info_parts.append(f"marks skipped ({cov:.1f}% - {'too many' if cov > 5 else 'none found'})")

    state["last_result"] = final
    return jsonify({
        "preview": to_b64(preview_resize(final)),
        "info": " | ".join(info_parts) if info_parts else "Processed",
    })


@app.route('/save', methods=['POST'])
def save():
    if state.get("last_result") is None:
        return jsonify({"error": "Run process first"})
    path = state["img_path"]
    base, ext = os.path.splitext(path)
    out = f"{base}_clean{ext}"

    rgb = cv2.cvtColor(state["last_result"], cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    kw = {}
    if state["icc_profile"]:
        kw["icc_profile"] = state["icc_profile"]
    try:
        orig = Image.open(path)
        exif = orig.info.get("exif")
        if exif:
            kw["exif"] = exif
    except:
        pass

    if ext.lower() in ('.tif', '.tiff'):
        pil.save(out, **kw)
    else:
        pil.save(out, quality=98, subsampling=0, **kw)
    return jsonify({"path": out})


if __name__ == '__main__':
    print("\n  Clean Backdrop v2")
    print("  http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
