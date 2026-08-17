from ultralytics import FastSAM, YOLO
import torch


class SceneSegmenter:
    """
    Flexible scene segmenter supporting multiple detector backends.
    """
    def __init__(self, detector_type="fastsam", model_size="s", conf=0.5,
                 imgsz=640, device=None, custom_model_path=None):
        """
        Initialize scene segmenter.

        Args:
            detector_type: 'fastsam' (default) or 'yolo'
            model_size: Model size - 's' (small, fast), 'x' (large, accurate)
            conf: Confidence threshold for detections
            imgsz: Input image size
            device: Device to run on (cuda/cpu)
            custom_model_path: Optional custom model path (overrides detector_type/model_size)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.imgsz = imgsz
        self.detector_type = detector_type.lower()

        # Determine model path
        if custom_model_path:
            model_path = custom_model_path
            self.model_name = custom_model_path
        else:
            if self.detector_type == "fastsam":
                model_path = f"FastSAM-{model_size}.pt"
                self.model_name = f"FastSAM-{model_size}"
            elif self.detector_type == "yolo":
                model_path = f"yolov8{model_size}-seg.pt"
                self.model_name = f"YOLOv8{model_size}-seg"
            else:
                raise ValueError(f"Unknown detector_type: {detector_type}. Use 'fastsam' or 'yolo'.")

        print(f"Loading {self.model_name} on {self.device}...")

        # Load appropriate model (ultralytics auto-downloads if needed)
        if self.detector_type == "fastsam":
            self.model = FastSAM(model_path)
        elif self.detector_type == "yolo":
            self.model = YOLO(model_path)

        print(f"Detector loaded successfully: {self.model_name}")

    def segment(self, image_bgr):
        """
        Input: OpenCV BGR Image
        Returns: results object with .boxes attribute
        """
        # Using verbose=False to keep console clean
        results = self.model(
            image_bgr,
            device=self.device,
            retina_masks=True,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False
        )
        return results[0]
