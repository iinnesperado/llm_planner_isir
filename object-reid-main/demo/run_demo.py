import cv2
import sys
import os
import time
import argparse
from tqdm import tqdm

# Ensure we can find the library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from object_reid_pillar.core.pipeline import ReidPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Object Re-ID on a video and save output.")
    parser.add_argument("--input", type=str, default="assets/test_sample.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default="output_reid_dinov2.mp4", help="Path to save output video")
    parser.add_argument("--db", type=str, default="sam3_dinov2_db.pkl", help="Path to database pickle file")

    # Encoder options
    parser.add_argument("--encoder", type=str, default="dinov2", choices=["dinov2", "clip", "hybrid"],
                        help="Encoder type: dinov2 (high precision), clip, or hybrid (CLIP+DINOv2)")
    parser.add_argument("--encoder-model", type=str, default=None,
                        help="Encoder model variant (e.g., dinov2_vitb14, dinov2_vitl14)")
    parser.add_argument("--fusion-mode", type=str, default="concat", choices=["concat", "average", "weighted"],
                        help="Hybrid fusion mode: concat (1280-dim), average, or weighted")
    parser.add_argument("--clip-weight", type=float, default=0.5,
                        help="CLIP weight for hybrid weighted mode (0-1, default: 0.5)")

    # Detector options
    parser.add_argument("--detector", type=str, default="fastsam", choices=["fastsam", "yolo"],
                        help="Detector type: fastsam (default) or yolo")
    parser.add_argument("--detector-size", type=str, default="x", choices=["s", "m", "l", "x"],
                        help="Detector model size: s=small/fast, x=large/accurate (default: x)")
    parser.add_argument("--detector-imgsz", type=int, default=640, help="Detector input image size")

    # Detection and matching parameters
    parser.add_argument("--conf", type=float, default=0.5, help="Detector confidence threshold")
    parser.add_argument("--thresh", type=float, default=0.75, help="Similarity threshold (higher = more precision)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K matches for voting")
    parser.add_argument("--margin", type=float, default=0.1, help="Margin between best/second-best for precision")
    parser.add_argument("--min-box-area", type=int, default=400, help="Minimum box area to process")

    # Processing options
    parser.add_argument("--show", action="store_true", help="Show the video window while processing")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every N frames (default: 1=all frames)")
    parser.add_argument("--hide-unknown", action="store_true", default=True, help="Hide unknown objects (default: True)")
    parser.add_argument("--show-unknown", action="store_false", dest="hide_unknown", help="Show unknown objects")
    args = parser.parse_args()

    # 1. Validate Paths
    # Resolve input video path (check relative to script first)
    input_video = args.input
    if not os.path.exists(input_video):
        input_video = os.path.join(os.path.dirname(__file__), args.input)
        if not os.path.exists(input_video):
            print(f"Error: Input video '{args.input}' not found.")
            print(f"Tried: {args.input} and {input_video}")
            return

    # Resolve database path
    db_path = os.path.join(os.path.dirname(__file__), args.db)
    if not os.path.exists(db_path):
        # Try looking in the current directory if not found in demo dir
        if os.path.exists(args.db):
            db_path = args.db
        else:
            print(f"Error: Database '{args.db}' not found.")
            print(f"Please run: python build_db.py --source sam3_objects --output {args.db}")
            return

    # 2. Initialize Pipeline
    print(f"Loading Pipeline...")
    print(f"  Encoder: {args.encoder.upper()}" + (f" ({args.encoder_model})" if args.encoder_model else ""))
    if args.encoder == "hybrid":
        print(f"    Fusion mode: {args.fusion_mode}")
        if args.fusion_mode == "weighted":
            print(f"    CLIP weight: {args.clip_weight}")
    print(f"  Detector: {args.detector.upper()}-{args.detector_size.upper()} (imgsz={args.detector_imgsz})")
    print(f"  Database: {db_path}")
    print(f"  Top-K: {args.top_k}, Margin: {args.margin}")
    pipeline = ReidPipeline(
        db_path=db_path,
        segment_conf=args.conf,
        encoder_type=args.encoder,
        encoder_model=args.encoder_model,
        top_k=args.top_k,
        margin=args.margin,
        detector_type=args.detector,
        detector_size=args.detector_size,
        detector_imgsz=args.detector_imgsz,
        fusion_mode=args.fusion_mode,
        clip_weight=args.clip_weight
    )

    # 3. Setup Video Input
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. Setup Video Output
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 'mp4v' for .mp4, 'XVID' for .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_to_process = total_frames // args.skip_frames
    print(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    print(f"Processing every {args.skip_frames} frame(s) ({frames_to_process} frames total)")
    print(f"Input: {input_video}")
    print(f"Output: {output_path}")
    print(f"Database: {db_path}")
    print(f"Segmentation confidence: {args.conf} | Similarity threshold: {args.thresh}")
    print("-" * 60)

    try:
        # Progress bar
        pbar = tqdm(total=total_frames, unit="frames", desc="Processing")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            #print(frame.type) np array of ints
            if not ret:
                break

            # Skip frames if requested
            if frame_idx % args.skip_frames == 0:
                # --- PROCESS FRAME ---
                annotated_frame, detections = pipeline.process_frame(
                    frame,
                    threshold=0.2,#args.thresh,
                    min_box_area=args.min_box_area,
                    hide_unknown=args.hide_unknown
                )
                
                print(args.thresh)
                
                print(detections)
                print([(d['label'],d['score']) for d in detections if d['label'] != 'Unknown'])

                # Add info overlay
                encoder_name = args.encoder.upper()
                detector_name = f"{args.detector.upper()}-{args.detector_size.upper()}"
                info_text = f"Frame: {frame_idx}/{total_frames} | {detector_name} + {encoder_name}"
                cv2.putText(annotated_frame, info_text, (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Count detected objects
                known_objects = [d for d in detections if d['label'] != 'Unknown']
                if known_objects:
                    obj_counts = {}
                    for d in known_objects:
                        obj_counts[d['label']] = obj_counts.get(d['label'], 0) + 1

                    y_offset = 60
                    for obj_name, count in obj_counts.items():
                        cv2.putText(annotated_frame, f"{obj_name}: {count}", (20, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        y_offset += 25

                # Write to file
                out.write(annotated_frame)

                # Show window if requested
                if args.show:
                    cv2.imshow("Re-ID Output", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing interrupted by user.")
                        break
            else:
                # For skipped frames, just write the original frame
                out.write(frame)
                
            time.sleep(100)

            frame_idx += 1
            pbar.update(1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\nDone! Video saved to {args.output}")

if __name__ == "__main__":
    main()
