import cv2
import sys
import os
import time
import argparse
from tqdm import tqdm

# Ensure we can find the library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from object_reid_pillar.core.pipeline import ReidPipeline

def get_pred(img_path):

    # Resolve database path
    db = 'my_db.pkl'
    db_path = os.path.join(os.path.dirname(__file__), db)
    if not os.path.exists(db_path):
        # Try looking in the current directory if not found in demo dir
        if os.path.exists(db):
            db_path = db
        else:
            print(f"Error")
            return
    pipeline = ReidPipeline(
        db_path=db_path,
        segment_conf=0.5,
        encoder_type='dinov2',
        encoder_model=None,
        top_k=5,
        margin=0.1,
        detector_type='fastsam',
        detector_size='x',
        detector_imgsz=640,
        fusion_mode='concat',
        clip_weight=0.5
    )


    try:
        # Progress bar

        frame_idx = 0
        while True:
            frame = cv2.imread(img_path)

            # Skip frames if requested
            if frame_idx == 0:
                # --- PROCESS FRAME ---
                annotated_frame, detections = pipeline.process_frame(
                    frame,
                    threshold=0.2,
                    min_box_area=400,
                    hide_unknown=True
                )
                
                #print(detections)
                return([(d['label'],d['score']) for d in detections if d['label'] != 'Unknown'])
    except Exception as e:
        print(e)
                
if __name__ == "__main__":
    print(get_pred('/home/user/ines_ros2_humble/object_pics/new_object_20260723_144134/rgb_000.png'))
