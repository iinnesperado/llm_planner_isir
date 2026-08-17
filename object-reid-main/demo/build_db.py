import os
import sys
import argparse

# Add project root to path so we can import 'object_reid_pillar' without installing if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from object_reid_pillar.core.pipeline import ReidPipeline

def main():
    parser = argparse.ArgumentParser(description="Build object re-identification database from image folders")
    parser.add_argument("--source", type=str, default="sam3_objects",
                        help="Source folder containing object subfolders with images")
    parser.add_argument("--output", type=str, default="sam3_dinov2_db.pkl",
                        help="Output database pickle file name")
    parser.add_argument("--encoder", type=str, default="dinov2", choices=["dinov2", "clip", "hybrid"],
                        help="Encoder type: dinov2 (high precision), clip, or hybrid (CLIP+DINOv2)")
    parser.add_argument("--model", type=str, default=None,
                        help="Encoder model variant (e.g., dinov2_vitb14, dinov2_vitl14)")
    parser.add_argument("--use-prototypes", action="store_true", default=True,
                        help="Use class prototypes (geodesic mean) instead of all vectors - more compact, faster matching (default: True)")
    parser.add_argument("--no-prototypes", action="store_false", dest="use_prototypes",
                        help="Store all vectors instead of class prototypes")

    # Augmentation options
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Apply data augmentation (color + geometric transforms) to each image (default: True)")
    parser.add_argument("--no-augment", action="store_false", dest="augment",
                        help="Disable data augmentation")
    parser.add_argument("--num-augmentations", type=int, default=6,
                        help="Number of augmented versions per image (default: 6)")
    parser.add_argument("--aggressive-aug", action="store_true",
                        help="Use aggressive augmentation ranges (wider brightness/contrast/rotation)")

    # Hybrid encoder options
    parser.add_argument("--fusion-mode", type=str, default="concat", choices=["concat", "average", "weighted"],
                        help="Hybrid fusion mode: concat (1280-dim), average, or weighted")
    parser.add_argument("--clip-weight", type=float, default=0.5,
                        help="CLIP weight for hybrid weighted mode (0-1, default: 0.5)")

    args = parser.parse_args()

    # Resolve paths relative to this script
    source = os.path.join(os.path.dirname(__file__), args.source)
    db_path = os.path.join(os.path.dirname(__file__), args.output)

    # Check if we have folders in source
    if not os.path.exists(source):
        print(f"Error: Source folder '{source}' not found!")
        print(f"Please create it with subfolders containing object images.")
        print("Example structure:")
        print("  sam3_objects/")
        print("    keyboard_1/")
        print("      frame_00001.png")
        print("      frame_00002.png")
        print("    cup_blue/")
        print("      frame_00001.png")
        return

    subfolders = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
    if not subfolders:
        print(f"Error: No subfolders found in {source}")
        print("Each subfolder should represent one object class with multiple images.")
        return

    print(f"Found {len(subfolders)} object classes: {', '.join(subfolders)}")
    print(f"Encoder: {args.encoder.upper()}" + (f" ({args.model})" if args.model else ""))
    if args.encoder == "hybrid":
        print(f"  Fusion mode: {args.fusion_mode}")
        if args.fusion_mode == "weighted":
            print(f"  CLIP weight: {args.clip_weight}")
    print(f"Mode: {'Class Prototypes (Geodesic Mean)' if args.use_prototypes else 'All Vectors'}")
    if args.augment:
        print(f"Augmentation: Enabled ({args.num_augmentations} augmentations per image)")
    print(f"Building database from: {source}")
    print(f"Output will be saved to: {db_path}")
    print("-" * 60)

    # Create pipeline with specified encoder and build database
    pipeline = ReidPipeline(
        encoder_type=args.encoder,
        encoder_model=args.model,
        fusion_mode=args.fusion_mode,
        clip_weight=args.clip_weight
    )
    pipeline.create_database(source, save_path=db_path,
                           use_class_prototypes=args.use_prototypes,
                           use_augmentation=args.augment,
                           num_augmentations=args.num_augmentations,
                           aggressive_aug=args.aggressive_aug)

    print("-" * 60)
    print(f"Database successfully created with {args.encoder.upper()}!")
    print(f"To run the demo, use:")
    print(f"  python run_demo.py --input <video_file> --db {args.output} --encoder {args.encoder}")

if __name__ == "__main__":
    main()
