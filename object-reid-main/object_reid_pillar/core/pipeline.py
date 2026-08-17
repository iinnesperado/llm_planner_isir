import cv2
import numpy as np
from PIL import Image
from .segmenter import SceneSegmenter
from .encoder import CLIPEncoder, DINOv2Encoder, HybridEncoder
from .database import ObjectDatabase

class ReidPipeline:
    def __init__(self, db_path=None, segment_conf=0.5, encoder_type="dinov2",
                 encoder_model=None, top_k=5, margin=0.1, detector_type="fastsam",
                 detector_size="x", detector_imgsz=640, fusion_mode="concat",
                 clip_weight=0.5):
        """
        Initialize the Re-ID pipeline.

        Args:
            db_path: Path to database file
            segment_conf: Segmentation confidence threshold
            encoder_type: 'dinov2' (default, high precision), 'clip', or 'hybrid'
            encoder_model: Model variant (e.g., 'dinov2_vitb14', 'dinov2_vitl14')
            top_k: Number of top matches to consider for voting
            margin: Margin for precision filtering
            detector_type: 'fastsam' or 'yolo'
            detector_size: 's' (small/fast), 'm' (medium), or 'x' (large/accurate, default)
            detector_imgsz: Input image size for detector
            fusion_mode: For hybrid encoder: 'concat', 'average', or 'weighted'
            clip_weight: For hybrid encoder weighted mode: weight for CLIP features
        """
        # Initialize encoder based on type
        if encoder_type.lower() == "dinov2":
            model = encoder_model or "dinov2_vitb14"
            self.encoder = DINOv2Encoder(model_name=model)
        elif encoder_type.lower() == "clip":
            model = encoder_model or "ViT-B-32"
            self.encoder = CLIPEncoder(model_name=model)
        elif encoder_type.lower() == "hybrid":
            self.encoder = HybridEncoder(
                dinov2_model=encoder_model or "dinov2_vitb14",
                fusion_mode=fusion_mode,
                clip_weight=clip_weight
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Use 'dinov2', 'clip', or 'hybrid'.")

        self.encoder_type = encoder_type

        # Initialize segmenter with detector parameters
        self.segmenter = SceneSegmenter(
            detector_type=detector_type,
            model_size=detector_size,
            conf=segment_conf,
            imgsz=detector_imgsz
        )

        self.db = ObjectDatabase(self.encoder)
        self.top_k = top_k
        self.margin = margin

        if db_path:
            self.db.load(db_path)

    def create_database(self, source_folder, save_path="object_db.pkl", use_class_prototypes=True,
                        use_augmentation=True, num_augmentations=6, aggressive_aug=False):
        """
        Create database from source folder.

        Args:
            source_folder: Path to folder with class subfolders
            save_path: Path to save database pickle file
            use_class_prototypes: If True, store one prototype (geodesic mean) per class
                                 instead of all individual vectors
            use_augmentation: If True, augment images with color/geometric transforms
            num_augmentations: Number of augmented versions per image
            aggressive_aug: If True, use aggressive augmentation ranges
        """
        self.db.build_from_folder(source_folder,
                                  use_class_prototypes=use_class_prototypes,
                                  use_augmentation=use_augmentation,
                                  num_augmentations=num_augmentations,
                                  aggressive_aug=aggressive_aug)
        self.db.save(save_path)

    def process_frame(self, frame, threshold=0.7, min_box_area=400, hide_unknown=True):
        """
        Full pipeline: Segment -> Crop -> Encode -> Identify -> Annotate

        Args:
            frame: Input BGR image
            threshold: Similarity threshold for matching
            min_box_area: Minimum bounding box area to process (filters tiny detections)
            hide_unknown: If True, don't draw bounding boxes for unknown objects

        Returns:
            annotated_frame: Frame with bounding boxes and labels
            detections: List of detection dictionaries
        """
        results = self.segmenter.segment(frame)

        detections = []
        annotated_frame = frame.copy()

        if not results.boxes:
            return annotated_frame, detections

        h, w, _ = frame.shape

        # Extract Crops with size filtering
        crops = []
        boxes_coords = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Basic sanity check on dimensions
            if x1 >= x2 or y1 >= y2:
                continue

            # Filter tiny boxes that are likely noise
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < min_box_area:
                continue

            crop_bgr = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

            if crop_bgr.size == 0:
                continue

            # Convert to RGB
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)

            crops.append(pil_crop)
            boxes_coords.append((x1, y1, x2, y2))

        if not crops:
            return annotated_frame, []

        # Batch encode crops
        # embeddings shape: (N, D)
        embeddings = self.encoder.get_embedding(crops)

        # Match each crop with precision-focused strategy
        for i, embedding in enumerate(embeddings):
            # Pass (1, D) vector to matcher
            emb_vector = embedding[np.newaxis, :]

            label, score = self.db.find_best_match(
                emb_vector,
                threshold=threshold,
                top_k=self.top_k,
                margin=self.margin
            )

            x1, y1, x2, y2 = boxes_coords[i]

            # Skip drawing if unknown and hide_unknown is True
            if label == "Unknown" and hide_unknown:
                # Still add to detections list for statistics, but don't draw
                detections.append({
                    "label": label,
                    "score": float(score),
                    "box": [x1, y1, x2, y2]
                })
                continue

            # Visualization with confidence-based styling
            if label == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                thickness = 2
            else:
                # Green with varying intensity based on confidence
                intensity = int(255 * min(score, 1.0))
                color = (0, intensity, 0)  # Brighter green = higher confidence
                thickness = 3

            # Draw Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Draw Label Background
            label_text = f"{label} ({score:.3f})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + tw + 5, y1), color, -1)

            # Draw Text
            cv2.putText(annotated_frame, label_text, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            detections.append({
                "label": label,
                "score": float(score),
                "box": [x1, y1, x2, y2]
            })

        return annotated_frame, detections
