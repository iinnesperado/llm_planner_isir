# Installation

## Requirements

- Python 3.8-3.11
- NVIDIA GPU with 4GB+ VRAM (recommended, CPU works but slower)
- 8GB RAM minimum

## Install

### 1. Setup environment

Using conda:
```bash
conda create -n object_reid python=3.10
conda activate object_reid
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

Or virtualenv:
```bash
python3 -m venv object_reid_env
source object_reid_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install library

```bash
cd /path/to/object_reid_pillar2/object_reid_pillar
pip install -e .
```

This installs: ultralytics, open_clip_torch, torch, opencv-python, numpy, tqdm

### 3. Test it

```bash
python -c "from object_reid_pillar.core.pipeline import ReidPipeline; print('OK')"
```

## Models

All models download automatically on first use:
- FastSAM (s/m/x) - Auto-downloads from GitHub
- DINOv2 - Auto-downloads via torch.hub
- CLIP - Auto-downloads via torch.hub

First run will take a few minutes to download ~250MB of models.

## Check GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Should print `CUDA: True` if GPU is set up.

## Troubleshooting

**CUDA out of memory**: Use smaller models (FastSAM-S) or lower resolution

**NumPy errors**: Run `pip install "numpy<2.0"`

**Video codec issues**: Install ffmpeg (`sudo apt-get install ffmpeg` on Linux)

**Slow first run**: First run downloads ~250MB of models, takes a few minutes

## System packages (Linux)

```bash
sudo apt-get update
sudo apt-get install python3-dev ffmpeg libsm6 libxext6
```

That's it. See [USAGE.md](USAGE.md) for how to use it.
