import torch
import open_clip
from PIL import Image
import numpy as np
import torchvision.transforms as T


class DINOv2Encoder:
    """
    DINOv2 encoder for high-precision instance-level matching.
    DINOv2 provides better discrimination for object re-identification compared to CLIP.
    """
    def __init__(self, model_name="dinov2_vitb14", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DINOv2 ({model_name}) on {self.device}...")

        # Load DINOv2 model from torch hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # DINOv2 preprocessing - standard ImageNet normalization
        self.preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"DINOv2 model loaded successfully. Feature dim: 768")

    def get_embedding(self, image_input):
        """
        image_input: PIL Image or list of PIL Images
        Returns: L2-normalized numpy array
        """
        # Handle list of images or single image
        if isinstance(image_input, list):
            if not image_input:
                return None
            image_tensor = torch.stack([self.preprocess(img) for img in image_input]).to(self.device)
        else:
            image_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get CLS token features (global representation)
            features = self.model(image_tensor)

            # L2 normalize for cosine similarity
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        # Move to CPU and numpy
        return features.cpu().numpy()


class CLIPEncoder:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading OpenCLIP ({model_name}) on {self.device}...")

        # Using open_clip as per your reference code
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.model.eval()

    def get_embedding(self, image_input):
        """
        image_input: PIL Image or list of PIL Images
        Returns: Numpy array (Normalized)
        """
        # Handle list of images or single image
        if isinstance(image_input, list):
            if not image_input:
                return None
            image_tensor = torch.stack([self.preprocess(img) for img in image_input]).to(self.device)
        else:
            image_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image_tensor)

        # Move to CPU and numpy for the math calculations later
        return features.cpu().numpy()


class HybridEncoder:
    """
    Hybrid encoder combining CLIP and DINOv2 features.
    Provides flexibility to use both feature types for improved matching.
    """
    def __init__(self, dinov2_model="dinov2_vitb14", clip_model="ViT-B-32",
                 fusion_mode="concat", clip_weight=0.5, device=None):
        """
        Args:
            dinov2_model: DINOv2 model variant (e.g., dinov2_vitb14)
            clip_model: CLIP model variant (e.g., ViT-B-32)
            fusion_mode: How to combine features:
                - "concat": Concatenate both features (dim: 768 + 512 = 1280)
                - "average": Average both features (requires same dim, so we'll use weighted)
                - "weighted": Weighted combination: clip_weight * CLIP + (1-clip_weight) * DINOv2
            clip_weight: Weight for CLIP features (only for "weighted" mode)
            device: Device to run on
        """
        self.fusion_mode = fusion_mode
        self.clip_weight = clip_weight
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading Hybrid Encoder (CLIP + DINOv2) on {self.device}...")
        print(f"  Fusion mode: {fusion_mode}")
        if fusion_mode == "weighted":
            print(f"  CLIP weight: {clip_weight}, DINOv2 weight: {1-clip_weight}")

        # Initialize both encoders
        self.clip_encoder = CLIPEncoder(model_name=clip_model, device=self.device)
        self.dinov2_encoder = DINOv2Encoder(model_name=dinov2_model, device=self.device)

        # Determine output feature dimension
        if fusion_mode == "concat":
            self.feature_dim = 768 + 512  # DINOv2 (768) + CLIP (512)
        else:
            self.feature_dim = 768  # Use DINOv2 dimension as base

        print(f"Hybrid encoder loaded. Output feature dim: {self.feature_dim}")

    def get_embedding(self, image_input):
        """
        Extract hybrid features from image(s).

        Args:
            image_input: PIL Image or list of PIL Images

        Returns:
            L2-normalized numpy array with hybrid features
        """
        # Get features from both encoders
        clip_features = self.clip_encoder.get_embedding(image_input)
        dinov2_features = self.dinov2_encoder.get_embedding(image_input)

        # Combine features based on fusion mode
        if self.fusion_mode == "concat":
            # Concatenate: [CLIP (512) | DINOv2 (768)] = 1280-dim
            hybrid_features = np.concatenate([clip_features, dinov2_features], axis=1)

        elif self.fusion_mode == "weighted":
            # Weighted combination requires same dimensions
            # We'll project CLIP (512) to DINOv2 space (768) via padding/interpolation
            # Simple approach: pad CLIP features with zeros to match DINOv2 dim
            clip_padded = np.pad(clip_features, ((0, 0), (0, 768 - 512)), mode='constant')

            # Weighted combination
            hybrid_features = (self.clip_weight * clip_padded +
                             (1 - self.clip_weight) * dinov2_features)

        else:  # "average" mode
            # Similar to weighted but with equal weights (0.5 each)
            clip_padded = np.pad(clip_features, ((0, 0), (0, 768 - 512)), mode='constant')
            hybrid_features = 0.5 * clip_padded + 0.5 * dinov2_features

        # L2 normalize the combined features
        norms = np.linalg.norm(hybrid_features, axis=1, keepdims=True) + 1e-8
        hybrid_features = hybrid_features / norms

        return hybrid_features
