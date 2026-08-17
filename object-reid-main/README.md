# Object Re-Identification Library

Detects and identifies objects in videos using DINOv2 features and FastSAM segmentation.

## What it does

The library segments objects in each frame, extracts their features, and matches them against a database of known objects. Works well across different lighting conditions and viewpoints thanks to data augmentation.

## Quick Start

### Install

```bash
cd /path/to/object_reid_pillar2/object_reid_pillar
pip install -e .
```

See [INSTALL.md](INSTALL.md) for details.

### Build a database

Put your object images in folders (one folder per object):

```
my_objects/
├── keyboard/
│   ├── img001.jpg
│   └── img002.jpg
├── mug/
│   └── img001.jpg
└── mouse/
    └── img001.jpg
```

Then:

```bash
cd demo
python build_db.py --source my_objects --output my_db.pkl
```

This applies augmentation (brightness, rotation, etc.) and creates compact prototypes for each object.

### Run on a video

```bash
python run_demo.py --input video.mp4 --db my_db.pkl --output result.mp4
```

## How it works

```
Video → FastSAM detection → Crop objects → DINOv2 features → Match database → Labels
```

**Defaults:**
- DINOv2-Small encoder (fast, accurate enough)
- FastSAM-X detector (most accurate)
- Standard augmentation enabled
- Class prototypes (geodesic mean)
- Threshold 0.75 for matching

## Main scripts

- `build_db.py` - Build database from image folders
- `run_demo.py` - Run inference on videos

See [USAGE.md](USAGE.md) for all options.

## Project structure

```
object_reid_pillar/
├── object_reid_pillar/
│   └── core/
│       ├── encoder.py      # DINOv2, CLIP, Hybrid
│       ├── database.py     # Build & match
│       ├── segmenter.py    # FastSAM/YOLO
│       └── pipeline.py     # Main pipeline
├── demo/
│   ├── build_db.py
│   ├── run_demo.py
│   └── sam3_objects/       # Example images
└── setup.py
```

## Notes

- First run downloads models (~100MB for DINOv2, ~150MB for FastSAM)
- GPU recommended but works on CPU (slower)
- Database with 9 objects = ~15KB file size

Part of PILLAR WP2.
