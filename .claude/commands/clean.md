Run clean-backdrop on a single image file.

Arguments: $ARGUMENTS should be a path to an image file (Windows or WSL path).

Steps:
1. Convert the path to WSL format if it's a Windows path (e.g. D:\foo\bar -> /mnt/d/foo/bar)
2. Run the following Python script via Bash, substituting the resolved WSL input path and output path (input folder + /cleaned/ + filename):

```python
from app import shadow_lift, freq_separation, detect_floor
from rembg import remove, new_session
from PIL import Image
import numpy as np, cv2, os

fpath = '<INPUT_PATH>'
out_dir = os.path.join(os.path.dirname(fpath), 'cleaned')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, os.path.basename(fpath))

img = cv2.imread(fpath)
h, w = img.shape[:2]
pil_img = Image.open(fpath).convert('RGB')
session = new_session('birefnet-portrait')
sm = np.array(remove(pil_img, session=session, only_mask=True))
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, k, iterations=3)
sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
pk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
protected = cv2.dilate(sm, pk, iterations=2)
bg_mask = (protected < 128).astype(np.uint8)
fm, fs = detect_floor(img, bg_mask)
print(f'{w}x{h} | Floor: {"row " + str(fs) if np.any(fm > 0) else "none"}')
current = shadow_lift(img, bg_mask, fm, sm, strength=0.7)
current = freq_separation(current, bg_mask, fm, sm, strength=0.5)

orig_pil = Image.open(fpath)
icc = orig_pil.info.get('icc_profile')
exif = orig_pil.info.get('exif')
rgb = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
pil_out = Image.fromarray(rgb)
kw = {}
if icc: kw['icc_profile'] = icc
if exif: kw['exif'] = exif
ext = os.path.splitext(out)[1].lower()
if ext in ('.tif', '.tiff'):
    pil_out.save(out, **kw)
else:
    pil_out.save(out, quality=98, subsampling=0, **kw)
print(f'Saved: {out}')
```

3. Show the user the output image using the Read tool so they can see the result.
