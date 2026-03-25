# Clean Backdrop

Free, open-source tool to clean up studio photo backdrops. Uses shadow lifting and frequency separation to remove shadows, marks, and blemishes while preserving the subject perfectly. No AI inpainting artifacts - just clean math on your GPU.

An alternative to paid tools like Retouch4me Clean Backdrop.

### Before & After

| Before | After |
|--------|-------|
| ![Before](docs/before.jpg) | ![After](docs/after.jpg) |

## How It Works

Two complementary techniques:

1. **Shadow Lift** - Blends cast shadows toward the sampled clean wall color. Removes shadows while preserving the natural wall gradient. Adjustable strength 0-100%.

2. **Texture Smoothing** (Frequency Separation) - Separates the image into low-frequency (lighting gradient) and high-frequency (texture/marks/scuffs). Smooths the texture while keeping the gradient intact. No false positives, no smudging.

Both techniques use **birefnet-portrait** for subject segmentation (state-of-the-art edge quality) and run on GPU via CUDA.

### Smart Edge Handling

- **Smooth subject masking** - Distance-based feathering that scales with image size, eliminating hard "bar" artifacts at subject boundaries. No visible seams between processed background and subject.
- **Automatic floor detection** - Distinguishes real floors (wood, tile) from wall shadow/vignetting using color analysis. Real floors are preserved; shadow darkening on walls is cleaned.
- **Vertical floor transition** - Wall-to-floor boundary uses a row-based ramp so floor texture and contact shadows around feet are never disturbed.

## Web UI

Local web interface with sliders and instant preview.

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

- Drop images from Explorer or paste a file path
- Adjust shadow lift and texture smoothing independently
- View tabs: Original | Shadows | Texture | Preview
- Auto-processes on image load
- Save outputs next to the original with `_clean` suffix
- ICC color profiles and EXIF metadata preserved

## Batch Processing

```bash
# Shadow lift + texture smoothing
python batch.py "D:\Photos\Export\My Shoot" --lift 70 --texture 50

# Shadow lift only
python batch.py /path/to/folder --lift 80 --texture 0

# Custom output folder
python batch.py "D:\Photos\shoot" --output retouched --lift 100 --texture 60

# Disable floor detection (treat everything as wall)
python batch.py /path/to/folder --no-floor
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA
- cuDNN 9 (for GPU-accelerated segmentation via `onnxruntime-gpu`)
- ~928MB for birefnet-portrait model (downloaded on first run)

## Installation

```bash
git clone https://github.com/isaacrowntree/clean-backdrop.git
cd clean-backdrop
pip install -r requirements.txt
pip install onnxruntime-gpu  # for GPU segmentation
```

### cuDNN Setup

ONNX Runtime needs `libcudnn.so.9` on `LD_LIBRARY_PATH`. If you see a `libcudnn.so.9: cannot open shared object file` error, find and export the path:

```bash
# Find cuDNN on your system
find /usr -name "libcudnn.so.9" 2>/dev/null
find /usr/local -name "libcudnn.so.9" 2>/dev/null

# Add to your shell profile (adjust path to match your system)
echo 'export LD_LIBRARY_PATH="/path/to/cudnn/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

Without cuDNN, segmentation falls back to CPU (slower but still works).

## License

MIT License - see [LICENSE](LICENSE) for details.
