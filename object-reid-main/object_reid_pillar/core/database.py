import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import pickle
import random

class ObjectDatabase:
    def __init__(self, encoder):
        self.encoder = encoder
        # We separate features and classes for vectorized numpy calculations
        self.dictionary_features = [] # List of numpy arrays, will turn to stacked array
        self.dictionary_classes = []  # List of strings
        self.dictionary_filenames = [] # List of strings (optional, for debug)

    def augment_image(self, pil_img, aggressive=False):
        """
        Apply color and geometric augmentations that preserve object identity.

        Args:
            pil_img: PIL Image to augment
            aggressive: If True, use more aggressive augmentation ranges

        Returns list of augmented images including the original.
        """
        augmented = [pil_img]  # Always include original

        # Set ranges based on aggressive mode
        if aggressive:
            brightness_range = (0.6, 1.4)
            contrast_range = (0.6, 1.4)
            saturation_range = (0.5, 1.5)
            rotation_range = (-20, 20)
            scale_range = (0.8, 1.0)
        else:
            brightness_range = (0.8, 1.2)
            contrast_range = (0.8, 1.2)
            saturation_range = (0.8, 1.2)
            rotation_range = (-10, 10)
            scale_range = (0.9, 1.0)

        # Color augmentations (preserve object appearance)
        # 1. Brightness variation
        enhancer = ImageEnhance.Brightness(pil_img)
        augmented.append(enhancer.enhance(random.uniform(*brightness_range)))

        # 2. Contrast variation
        enhancer = ImageEnhance.Contrast(pil_img)
        augmented.append(enhancer.enhance(random.uniform(*contrast_range)))

        # 3. Color/Saturation variation
        enhancer = ImageEnhance.Color(pil_img)
        augmented.append(enhancer.enhance(random.uniform(*saturation_range)))

        # Geometric augmentations (preserve object identity)
        # 4. Horizontal flip
        augmented.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))

        # 5. Small rotation
        angle = random.uniform(*rotation_range)
        augmented.append(pil_img.rotate(angle, fillcolor=(255, 255, 255)))

        # 6. Scale variation (crop and resize)
        w, h = pil_img.size
        scale = random.uniform(*scale_range)
        new_w, new_h = int(w * scale), int(h * scale)
        left = random.randint(0, w - new_w) if new_w < w else 0
        top = random.randint(0, h - new_h) if new_h < h else 0
        cropped = pil_img.crop((left, top, left + new_w, top + new_h))
        augmented.append(cropped.resize((w, h), Image.BILINEAR))

        return augmented

    def build_from_folder(self, root_folder, use_class_prototypes=False, use_augmentation=False, num_augmentations=6, aggressive_aug=False):
        """
        Build database from folder structure.

        Args:
            root_folder: Path to folder with class subfolders
            use_class_prototypes: If True, compute and store one prototype per class
                                 (geodesic mean on unit sphere). Otherwise store all vectors.
            use_augmentation: If True, augment each image before extracting features
            num_augmentations: Number of augmented versions per image (default: 6)
            aggressive_aug: If True, use aggressive augmentation ranges
        """
        if not os.path.exists(root_folder):
            print(f"Folder {root_folder} not found.")
            return

        print(f"Building database from {root_folder}...")
        if use_class_prototypes:
            print(f"  Mode: Class prototypes (geodesic mean per class)")
        else:
            print(f"  Mode: All individual vectors")
        if use_augmentation:
            aug_type = "Aggressive" if aggressive_aug else "Standard"
            print(f"  Augmentation: {aug_type} ({num_augmentations} augmentations per image)")

        temp_features = []
        temp_classes = []
        temp_files = []

        # Collect all features per class
        class_features = {}  # {class_name: [features]}

        for class_name in os.listdir(root_folder):
            class_path = os.path.join(root_folder, class_name)
            if not os.path.isdir(class_path):
                continue

            class_features[class_name] = []

            for img_name in tqdm(os.listdir(class_path), desc=f"Loading {class_name}"):
                img_path = os.path.join(class_path, img_name)
                try:
                    # Open and Convert
                    pil_img = Image.open(img_path).convert("RGB")

                    # Apply augmentation if enabled
                    if use_augmentation:
                        augmented_images = self.augment_image(pil_img, aggressive=aggressive_aug)[:num_augmentations + 1]  # +1 for original
                    else:
                        augmented_images = [pil_img]

                    # Extract features from all versions (original + augmented)
                    for idx, aug_img in enumerate(augmented_images):
                        emb = self.encoder.get_embedding(aug_img)

                        if use_class_prototypes:
                            class_features[class_name].append(emb)
                        else:
                            aug_suffix = f"_aug{idx}" if idx > 0 else ""
                            temp_features.append(emb)
                            temp_classes.append(class_name)
                            temp_files.append(f"{img_name}{aug_suffix}")

                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

        if use_class_prototypes:
            # Compute prototype (geodesic mean) for each class
            print("Computing class prototypes...")
            for class_name, features in class_features.items():
                if not features:
                    continue

                # Stack all features for this class: (N, D)
                class_vecs = np.vstack(features)

                # Geodesic mean on unit sphere = arithmetic mean + renormalize
                prototype = np.mean(class_vecs, axis=0, keepdims=True)  # (1, D)
                prototype = prototype / (np.linalg.norm(prototype) + 1e-8)  # Renormalize

                temp_features.append(prototype)
                temp_classes.append(class_name)
                temp_files.append(f"prototype_{class_name}")

            print(f"Database built: {len(temp_features)} class prototypes")

        if temp_features:
            # Stack into (N, D) array
            self.dictionary_features = np.vstack(temp_features)
            self.dictionary_classes = np.array(temp_classes)
            self.dictionary_filenames = np.array(temp_files)
            if not use_class_prototypes:
                print(f"Database built: {self.dictionary_features.shape[0]} vectors.")
        else:
            print("No valid images found.")

    def save(self, path="object_db.pkl"):
        data = {
            'features': self.dictionary_features,
            'classes': self.dictionary_classes,
            'filenames': self.dictionary_filenames
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Database saved to {path}")

    def load(self, path="object_db.pkl"):
        if not os.path.exists(path):
            print(f"Database file {path} not found.")
            return

        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.dictionary_features = data['features']
            self.dictionary_classes = data['classes']
            self.dictionary_filenames = data.get('filenames', [])
        print(f"Database loaded: {len(self.dictionary_classes)} entries.")

    def find_best_match(self, query_embedding, threshold=0.6, top_k=5, margin=0.05):
        """
        High-precision matching with voting and margin-based filtering.

        Args:
            query_embedding: (1, D) query feature vector
            threshold: Minimum similarity score for a match
            top_k: Number of top matches to consider for voting
            margin: Margin between best and second-best class for high confidence

        Returns:
            (class_name, confidence_score)
        """
        if len(self.dictionary_features) == 0:
            return "Unknown", 0.0

        # query_embedding: (1, D)
        # self.dictionary_features: (N, D)

        # Cosine Similarity (features should already be L2 normalized)
        # For normalized vectors: cosine_sim = dot_product
        similarities = np.dot(self.dictionary_features, query_embedding.T).squeeze()

        # Get top-k matches
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        top_k_scores = similarities[top_k_indices]
        top_k_classes = self.dictionary_classes[top_k_indices]

        # Filter matches above threshold
        valid_mask = top_k_scores > threshold
        if not valid_mask.any():
            return "Unknown", top_k_scores[0] if len(top_k_scores) > 0 else 0.0

        valid_indices = top_k_indices[valid_mask]
        valid_scores = top_k_scores[valid_mask]
        valid_classes = top_k_classes[valid_mask]

        # Voting: Count occurrences of each class in top matches
        class_votes = {}
        class_scores = {}
        for cls, score in zip(valid_classes, valid_scores):
            class_votes[cls] = class_votes.get(cls, 0) + 1
            if cls not in class_scores:
                class_scores[cls] = []
            class_scores[cls].append(score)

        # Find class with most votes
        best_class = max(class_votes.items(), key=lambda x: (x[1], np.mean(class_scores[x[0]])))
        best_class_name = best_class[0]
        avg_score = np.mean(class_scores[best_class_name])

        # Precision check: verify margin between best and second best
        if len(class_scores) > 1:
            sorted_classes = sorted(class_scores.items(),
                                   key=lambda x: np.mean(x[1]),
                                   reverse=True)
            best_avg = np.mean(sorted_classes[0][1])
            second_best_avg = np.mean(sorted_classes[1][1])

            # If margin is too small, be conservative
            if (best_avg - second_best_avg) < margin:
                return "Unknown", best_avg

        return best_class_name, avg_score
