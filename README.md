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
```

## Requirements

- Python 3.10+
- NVIDIA GPU (CUDA) - used for segmentation and processing
- `onnxruntime-gpu` for GPU-accelerated subject segmentation
- ~928MB for birefnet-portrait model (downloaded on first run)

## Installation

```bash
git clone https://github.com/isaacrowntree/clean-backdrop.git
cd clean-backdrop
pip install -r requirements.txt
pip install onnxruntime-gpu  # for GPU segmentation
```

## License

MIT License - see [LICENSE](LICENSE) for details.
