# Usage Guide

## Building a database

### Basic usage

```bash
cd demo
python build_db.py --source my_objects --output my_db.pkl
```

This uses the defaults:
- DINOv2-Small encoder
- Standard augmentation (enabled)
- Class prototypes (geodesic mean)

### Options

```bash
# Disable augmentation
python build_db.py --source my_objects --output my_db.pkl --no-augment

# Use aggressive augmentation (wider ranges)
python build_db.py --source my_objects --output my_db.pkl --aggressive-aug

# Use larger DINOv2 model (slower, more accurate)
python build_db.py --source my_objects --output my_db.pkl --model dinov2_vitb14

# Store all vectors instead of prototypes (larger file, might help if few images per class)
python build_db.py --source my_objects --output my_db.pkl --no-prototypes
```

### Folder structure

Each subfolder = one object class. Put multiple images of each object:

```
my_objects/
├── keyboard_logitech/
│   ├── view1.jpg
│   ├── view2.jpg
│   ├── view3.jpg
│   └── ...
├── mug_blue/
│   ├── img1.jpg
│   └── img2.jpg
└── mouse/
    └── ...
```

More images per object = better prototypes. 5-20 images is usually enough.

## Running inference

### Basic usage

```bash
python run_demo.py --input video.mp4 --db my_db.pkl --output result.mp4
```

### Options

```bash
# Adjust similarity threshold (lower = more detections, more false positives)
python run_demo.py --input video.mp4 --db my_db.pkl --output result.mp4 --threshold 0.65

# Use different detector size
python run_demo.py --input video.mp4 --db my_db.pkl --output result.mp4 --detector-size x  # s/m/x

# Skip frames (process every N frames, faster)
python run_demo.py --input video.mp4 --db my_db.pkl --output result.mp4 --skip-frames 5

# Hide unknown objects (only show identified ones)
python run_demo.py --input video.mp4 --db my_db.pkl --output result.mp4 --hide-unknown

# Lower detector confidence (detect more objects)
python run_demo.py --input video.mp4 --db my_db.pkl --output result.mp4 --segment-conf 0.3
```

## Using in your code

```python
from object_reid_pillar.core.pipeline import ReidPipeline

# Create pipeline
pipeline = ReidPipeline(
    db_path="my_db.pkl",
    detector_size="m",      # s/m/x
    segment_conf=0.5,       # detection confidence
)

# Process single frame
import cv2
frame = cv2.imread("image.jpg")
annotated_frame, detections = pipeline.process_frame(
    frame,
    threshold=0.75,         # matching threshold
    hide_unknown=True       # don't draw unknown objects
)

# Save result
cv2.imwrite("output.jpg", annotated_frame)

# Check detections
for det in detections:
    print(f"{det['label']}: {det['score']:.3f} at {det['box']}")
```

### Building database in code

```python
from object_reid_pillar.core.pipeline import ReidPipeline

pipeline = ReidPipeline()
pipeline.create_database(
    source_folder="my_objects",
    save_path="my_db.pkl",
    use_class_prototypes=True,   # use prototypes (recommended)
    use_augmentation=True,        # enable augmentation (recommended)
    num_augmentations=6,          # how many per image
    aggressive_aug=False          # False = standard, True = aggressive
)
```

## Parameter guide

### Threshold

Controls when to accept a match:
- `0.75` (default): Conservative, fewer false positives
- `0.65`: More lenient, catches more but may have false positives
- `0.85`: Very strict, only very confident matches

### Detector size

- `s`: Fast, use for real-time or low-end GPU
- `m`: Medium, good balance of speed and accuracy
- `x`: Most accurate (default), best detection quality

### Augmentation

Standard (default):
- Brightness/contrast/saturation: ±20%
- Rotation: ±10°
- Scale: 90-100%

Aggressive (--aggressive-aug):
- Brightness/contrast: ±40%
- Saturation: ±50%
- Rotation: ±20°
- Scale: 80-100%

Use aggressive if your objects appear under very different conditions.

## Tips

**Not enough detections?**
- Lower `--threshold` (try 0.65)
- Lower `--segment-conf` (try 0.3)
- Add more training images with different viewpoints

**Too many false positives?**
- Raise `--threshold` (try 0.85)
- Use `--hide-unknown` to hide uncertain matches
- Add more training images of the correct objects

**Slow inference?**
- Use `--detector-size s`
- Use `--skip-frames 5` (process every 5th frame)
- Resize video to lower resolution before processing

**Database too large?**
- Make sure you're using `--use-prototypes` (default)
- Reduce number of training images per class (10-20 is enough)

**Poor accuracy?**
- Add more diverse training images (different angles, lighting)
- Enable augmentation if disabled
- Try `--model dinov2_vitb14` for better features (slower)

## Example workflow

1. Collect 10-20 images per object from different angles
2. Organize into folders
3. Build database: `python build_db.py --source objects --output db.pkl`
4. Test on video: `python run_demo.py --input test.mp4 --db db.pkl --output out.mp4`
5. If accuracy is bad, add more training images and rebuild
6. If too slow, use smaller detector or skip frames

That's it.
