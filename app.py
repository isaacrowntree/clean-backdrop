#!/usr/bin/env python3
"""
Clean Backdrop - Studio background retouching tool.
1. Shadow lift: blends cast shadows toward clean wall color
2. Frequency separation: smooths texture/marks while keeping lighting gradient
Both run on GPU. No AI inpainting needed.
"""

import os
import io
import base64
import re
import subprocess
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template_string, request, jsonify
from rembg import remove, new_session

app = Flask(__name__)

state = {
    "img": None,
    "img_path": None,
    "subject_mask": None,
    "bg_mask": None,
    "floor_mask": None,
    "floor_start_row": None,
    "preview_size": 1200,
    "icc_profile": None,
    "last_result": None,
}


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


# ── Technique 1: Shadow lift ──
def shadow_lift(img, bg_mask, floor_mask, subject_mask, strength=0.7):
    """Blend shadows toward clean wall color. Preserves gradient shape."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    work = bg_mask.copy()
    work[floor_mask > 0] = 0

    if np.sum(work) < 100:
        return img.copy()

    # Clean wall color from brightest bg pixels
    wall_pixels = img[work > 0]
    brightness = np.mean(wall_pixels, axis=1)
    bright_pixels = wall_pixels[brightness > np.percentile(brightness, 80)]
    if len(bright_pixels) < 10:
        return img.copy()
    clean_wall = np.median(bright_pixels, axis=0).astype(np.float32)
    clean_brightness = np.mean(clean_wall)

    # Shadow amount: how much darker vs clean wall
    darkness = clean_brightness - gray
    max_depth = clean_brightness * 0.5
    shadow_amount = np.clip(darkness / max(max_depth, 1), 0, 1) * strength

    shadow_amount[work == 0] = 0

    # Wide blur for smooth transition
    blur_size = max(int(min(h, w) * 0.06), 51)
    blur_size = blur_size + (1 - blur_size % 2)
    shadow_amount = cv2.GaussianBlur(shadow_amount, (blur_size, blur_size), 0)

    # Blend toward clean wall
    result = img.astype(np.float32)
    blend = shadow_amount[:, :, np.newaxis]
    clean_bg = np.full_like(result, clean_wall)
    result = result * (1 - blend) + clean_bg * blend

    # Subject composite - tight edge
    subj_f = (subject_mask > 128).astype(np.float32)
    subj_f = cv2.GaussianBlur(subj_f, (3, 3), 0)[:, :, np.newaxis]
    result = subj_f * img.astype(np.float32) + (1 - subj_f) * result

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Technique 2: Frequency separation ──
def freq_separation(img, bg_mask, floor_mask, subject_mask, strength=0.7):
    """
    Remove high-frequency detail (marks, scuffs, wrinkles, texture) from the wall
    while keeping the low-frequency lighting gradient intact.
    strength: 0 = no smoothing, 1 = full texture removal
    """
    h, w = img.shape[:2]
    work = bg_mask.copy()
    work[floor_mask > 0] = 0

    if np.sum(work) < 100:
        return img.copy()

    # Low-freq = heavy Gaussian blur (the gradient/lighting)
    # Sigma scales with image size - larger image = need larger sigma
    sigma = max(min(h, w) // 40, 10)

    # Weighted blur: only average from background pixels so subject doesn't bleed in
    img_f = img.astype(np.float32)
    wm = work.astype(np.float32)
    wm3 = wm[:, :, np.newaxis]

    weighted = img_f * wm3
    weight = wm3.copy()

    ksize = int(sigma * 6) | 1  # 6*sigma, make odd
    low_freq = cv2.GaussianBlur(weighted, (ksize, ksize), sigma)
    low_weight = cv2.GaussianBlur(weight, (ksize, ksize), sigma)
    low_weight = np.maximum(low_weight, 1e-6)
    low_freq = low_freq / low_weight

    # High-freq = original - low_freq (the texture/marks/detail)
    # Removing high-freq = just using low_freq on the background

    # Blend: strength controls how much texture is removed
    # 0 = original, 1 = fully smoothed (low_freq only)
    result = img_f.copy()
    blend = (wm * strength)[:, :, np.newaxis]
    result = result * (1 - blend) + low_freq * blend

    # Subject composite - tight edge
    subj_f = (subject_mask > 128).astype(np.float32)
    subj_f = cv2.GaussianBlur(subj_f, (3, 3), 0)[:, :, np.newaxis]
    result = subj_f * img_f + (1 - subj_f) * result

    return np.clip(result, 0, 255).astype(np.uint8)


HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Clean Backdrop</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a1a; color: #e0e0e0; }
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
                <div class="tab" data-v="shadows" onclick="view('shadows')">Shadows</div>
                <div class="tab" data-v="texture" onclick="view('texture')">Texture</div>
                <div class="tab" data-v="preview" onclick="view('preview')">Preview</div>
            </div>
        </div>

        <div class="section">
            <h3>1. Shadow Lift</h3>
            <div class="check-row"><input type="checkbox" id="doShadow" checked><label for="doShadow">Enable</label></div>
            <div class="control">
                <label>Strength <span id="liftV">70</span>%</label>
                <input type="range" id="lift" min="0" max="100" value="70"
                    oninput="document.getElementById('liftV').textContent=this.value">
            </div>
        </div>

        <div class="section">
            <h3>2. Texture Smoothing</h3>
            <div class="check-row"><input type="checkbox" id="doTexture" checked><label for="doTexture">Enable</label></div>
            <div class="control">
                <label>Smoothing <span id="texV">50</span>%</label>
                <input type="range" id="tex" min="0" max="100" value="50"
                    oninput="document.getElementById('texV').textContent=this.value">
            </div>
        </div>

        <div class="section">
            <h3>Process</h3>
            <button class="btn btn-blue" id="prevBtn" onclick="doPreview()" disabled>Preview</button>
            <button class="btn btn-grey" id="saveBtn" onclick="doSave()" disabled>Save Full Resolution</button>
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
    if (imgs[v]) {
        document.getElementById('img').src = 'data:image/jpeg;base64,' + imgs[v];
        document.getElementById('img').style.display='block';
        document.getElementById('ph').style.display='none';
    }
}

function loading(msg) { document.getElementById('loadMsg').textContent=msg||'Processing...'; document.getElementById('loading').classList.add('on'); }
function loaded() { document.getElementById('loading').classList.remove('on'); }
function status(s) { document.getElementById('status').textContent=s; }

function params() { return {
    do_shadow: document.getElementById('doShadow').checked,
    lift: +document.getElementById('lift').value / 100,
    do_texture: document.getElementById('doTexture').checked,
    tex_strength: +document.getElementById('tex').value / 100,
};}

function handleDrop(file) {
    loading('Finding ' + file.name + '...');
    fetch('/find_load', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:file.name}) })
    .then(r=>r.json()).then(d => { loaded(); if(d.error){alert(d.error);return;}
        imgs = {original: d.image}; document.getElementById('imageInfo').textContent=d.info;
        document.getElementById('pathInput').value=d.path||'';
        document.getElementById('prevBtn').disabled=false;
        view('original');
        doPreview();
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
        document.getElementById('prevBtn').disabled=false;
        view('original');
        doPreview();
    }).catch(e=>{loaded();alert(e);});
}

function doPreview() {
    loading('Processing...');
    fetch('/preview', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params())})
    .then(r=>r.json()).then(d=>{loaded();if(d.error){alert(d.error);return;}
        imgs.preview=d.preview; imgs.shadows=d.shadows; imgs.texture=d.texture;
        document.getElementById('saveBtn').disabled=false;
        status(d.info); view('preview');
    }).catch(e=>{loaded();alert(e);});
}

function doSave() {
    loading('Saving full resolution...');
    fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params())})
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

    # Segment subject on GPU via birefnet-portrait
    print(f"Segmenting {os.path.basename(path)} ({w}x{h})...")
    pil_img = Image.open(path).convert("RGB")
    session = new_session("birefnet-portrait")
    sm = np.array(remove(pil_img, session=session, only_mask=True))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, k, iterations=3)
    sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    state["subject_mask"] = sm

    pk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    protected = cv2.dilate(sm, pk, iterations=2)
    state["bg_mask"] = (protected < 128).astype(np.uint8)

    fm, fs = detect_floor(img, state["bg_mask"])
    state["floor_mask"] = fm
    state["floor_start_row"] = fs
    print(f"  Done. Floor: {'row ' + str(fs) if np.sum(fm) > 0 else 'none'}")

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

    # Step 1: Shadow lift
    shadows_vis = None
    if p.get("do_shadow"):
        current = shadow_lift(current, state["bg_mask"], state["floor_mask"],
                              state["subject_mask"], strength=p.get("lift", 0.7))
        # Visualization
        diff = np.abs(current.astype(np.float32) - img.astype(np.float32))
        diff_vis = current.copy()
        changed = np.mean(diff, axis=2) > 3
        diff_vis[changed] = [0, 165, 255]
        shadows_vis = preview_resize(cv2.addWeighted(img, 0.4, diff_vis, 0.6, 0))
        info_parts.append(f"shadows {int(p.get('lift', 0.7)*100)}%")

    # Step 2: Frequency separation (texture smoothing)
    texture_vis = None
    if p.get("do_texture"):
        current = freq_separation(current, state["bg_mask"], state["floor_mask"],
                                   state["subject_mask"], strength=p.get("tex_strength", 0.5))
        # Visualization: show what was removed (high-freq component)
        removed = np.abs(current.astype(np.float32) - (img if not p.get("do_shadow") else
                         shadow_lift(img, state["bg_mask"], state["floor_mask"],
                                     state["subject_mask"], strength=p.get("lift", 0.7))).astype(np.float32))
        tex_vis = current.copy()
        tex_changed = np.mean(removed, axis=2) > 2
        tex_vis[tex_changed] = [255, 165, 0]
        texture_vis = preview_resize(cv2.addWeighted(current, 0.4, tex_vis, 0.6, 0))
        info_parts.append(f"texture {int(p.get('tex_strength', 0.5)*100)}%")

    state["last_result"] = current

    return jsonify({
        "preview": to_b64(preview_resize(current)),
        "shadows": to_b64(shadows_vis) if shadows_vis is not None else to_b64(preview_resize(img)),
        "texture": to_b64(texture_vis) if texture_vis is not None else to_b64(preview_resize(img)),
        "info": " | ".join(info_parts) if info_parts else "No processing",
    })


@app.route('/save', methods=['POST'])
def save():
    if state.get("last_result") is None:
        return jsonify({"error": "Run preview first"})

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
    print("\n  Clean Backdrop")
    print("  http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
