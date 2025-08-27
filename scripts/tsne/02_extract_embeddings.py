#!/usr/bin/env python3
"""
Script to extract image embeddings using OpenCLIP or DINOv2 for all sampled game frames.
For each image in ./sampled_frames, extracts embeddings and saves them as .npy files.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import torchvision.transforms as transforms

try:
    import open_clip
except ImportError:
    open_clip = None


class EmbeddingExtractor:
    """Extract image embeddings using OpenCLIP models."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None):
        """
        Initialize the embedding extractor.

        Args:
            model_name: OpenCLIP model architecture (e.g., 'ViT-B-32', 'ViT-L-14')
            pretrained: Pretrained weights to use (e.g., 'openai', 'laion2b_s34b_b79k')
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and preprocessing
        print(f"Loading OpenCLIP model: {model_name} with {pretrained} weights...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

        # Get embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_features = self.model.encode_image(dummy_input)
            self.embedding_dim = dummy_features.shape[-1]

        print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract embedding for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Normalized embedding as numpy array
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model.encode_image(image_tensor)
                # Normalize features (standard practice for CLIP embeddings)
                features = features / features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().squeeze()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def extract_batch_embeddings(self, image_paths: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for a batch of images efficiently.

        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process in each batch

        Returns:
            Dictionary mapping image paths to their embeddings
        """
        embeddings = {}

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            valid_paths = []

            # Load and preprocess batch
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    image_tensor = self.preprocess(image)
                    batch_images.append(image_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue

            if not batch_images:
                continue

            # Stack into batch tensor
            batch_tensor = torch.stack(batch_images).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                # Normalize features
                features = features / features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().numpy()

            # Store embeddings
            for path, embedding in zip(valid_paths, features_np):
                embeddings[path] = embedding

        return embeddings


class DINOv2EmbeddingExtractor:
    """Extract image embeddings using DINOv2 models."""

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = None):
        """
        Initialize the DINOv2 embedding extractor.

        Args:
            model_name: DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load DINOv2 model from torch hub
        print(f"Loading DINOv2 model: {model_name}...")
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")
            print("Make sure you have internet connection for the first download.")
            sys.exit(1)

        # Define preprocessing transforms for DINOv2
        # DINOv2 expects images to be normalized with ImageNet stats
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Get embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_features = self.model(dummy_input)
            self.embedding_dim = dummy_features.shape[-1]

        print(f"DINOv2 model loaded successfully. Embedding dimension: {self.embedding_dim}")

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract embedding for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Embedding as numpy array
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)

            return features.cpu().numpy().squeeze()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def extract_batch_embeddings(self, image_paths: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for a batch of images efficiently.

        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process in each batch

        Returns:
            Dictionary mapping image paths to their embeddings
        """
        embeddings = {}

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            valid_paths = []

            # Load and preprocess batch
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    image_tensor = self.preprocess(image)
                    batch_images.append(image_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue

            if not batch_images:
                continue

            # Stack into batch tensor
            batch_tensor = torch.stack(batch_images).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model(batch_tensor)
                features_np = features.cpu().numpy()

            # Store embeddings
            for path, embedding in zip(valid_paths, features_np):
                embeddings[path] = embedding

        return embeddings


def find_all_images(sampled_frames_dir: str) -> Dict[str, List[str]]:
    """
    Find all image files in the sampled_frames directory, organized by game.

    Args:
        sampled_frames_dir: Path to the sampled_frames directory

    Returns:
        Dictionary mapping game names to lists of image paths
    """
    games_images = {}
    sampled_frames_path = Path(sampled_frames_dir)

    if not sampled_frames_path.exists():
        raise FileNotFoundError(f"Directory {sampled_frames_dir} does not exist")

    # Iterate through game directories
    for game_dir in sampled_frames_path.iterdir():
        if game_dir.is_dir():
            game_name = game_dir.name
            image_files = []

            # Find all image files in the game directory
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(game_dir.glob(ext))

            if image_files:
                games_images[game_name] = [str(path) for path in sorted(image_files)]
                print(f"Found {len(image_files)} images for {game_name}")

    return games_images


def create_embeddings_directory(sampled_frames_dir: str, embeddings_dir: str = None) -> str:
    """
    Create embeddings directory structure mirroring the sampled_frames structure.

    Args:
        sampled_frames_dir: Path to the sampled_frames directory
        embeddings_dir: Path for embeddings directory (default: ./embeddings_clip)

    Returns:
        Path to the embeddings directory
    """
    if embeddings_dir is None:
        embeddings_dir = "./embeddings_clip"

    embeddings_path = Path(embeddings_dir)
    embeddings_path.mkdir(exist_ok=True)

    # Create subdirectories for each game
    sampled_frames_path = Path(sampled_frames_dir)
    for game_dir in sampled_frames_path.iterdir():
        if game_dir.is_dir():
            game_embeddings_dir = embeddings_path / game_dir.name
            game_embeddings_dir.mkdir(exist_ok=True)

    return str(embeddings_path)


def save_embedding(embedding: np.ndarray, image_path: str, embeddings_dir: str):
    """
    Save embedding to a .npy file with the same structure as the image path.

    Args:
        embedding: The embedding array to save
        image_path: Original image path
        embeddings_dir: Base directory for embeddings
    """
    # Convert image path to embedding path
    image_path_obj = Path(image_path)
    relative_path = image_path_obj.relative_to("sampled_frames")

    # Change extension to .npy
    embedding_path = Path(embeddings_dir) / relative_path.with_suffix(".npy")

    # Save embedding
    np.save(embedding_path, embedding)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract image embeddings using OpenCLIP or DINOv2",
        epilog="""
Examples:
  # Extract embeddings using OpenCLIP (default)
  python extract_embeddings.py

  # Extract embeddings using DINOv2 (saves to ./embeddings_dino/)
  python extract_embeddings.py --model-type dinov2

  # Use specific DINOv2 model variant
  python extract_embeddings.py --model-type dinov2 --dinov2-model dinov2_vitl14

  # Use specific OpenCLIP model
  python extract_embeddings.py --model-type openclip --openclip-model ViT-L-14 --openclip-pretrained laion2b_s34b_b79k
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-type",
        choices=["openclip", "dinov2"],
        default="openclip",
        help="Type of model to use for embedding extraction (default: openclip)",
    )

    # OpenCLIP specific arguments
    parser.add_argument("--openclip-model", default="ViT-B-32", help="OpenCLIP model architecture (default: ViT-B-32)")
    parser.add_argument(
        "--openclip-pretrained", default="openai", help="OpenCLIP pretrained weights (default: openai)"
    )

    # DINOv2 specific arguments
    parser.add_argument(
        "--dinov2-model",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        default="dinov2_vitb14",
        help="DINOv2 model variant (default: dinov2_vitb14)",
    )

    # Common arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")
    parser.add_argument(
        "--sampled-frames-dir",
        default="./sampled_frames",
        help="Directory containing sampled frames (default: ./sampled_frames)",
    )
    parser.add_argument(
        "--embeddings-dir",
        help="Directory to save embeddings (default: ./embeddings_clip for OpenCLIP, ./embeddings_dino for DINOv2)",
    )

    return parser.parse_args()


def main():
    """Main function to extract embeddings for all sampled frames."""
    args = parse_args()

    # Set default embeddings directory based on model type
    if args.embeddings_dir is None:
        if args.model_type == "dinov2":
            embeddings_dir = "./embeddings_dino"
        else:
            embeddings_dir = "./embeddings_clip"
    else:
        embeddings_dir = args.embeddings_dir

    # Print configuration
    if args.model_type == "openclip":
        if open_clip is None:
            print("Error: open_clip_torch is not installed. Please install it with:")
            print("pip install open_clip_torch")
            sys.exit(1)

        print("=== OpenCLIP Image Embedding Extraction ===")
        print(f"Model: {args.openclip_model}")
        print(f"Pretrained: {args.openclip_pretrained}")
    else:
        print("=== DINOv2 Image Embedding Extraction ===")
        print(f"Model: {args.dinov2_model}")

    print(f"Batch size: {args.batch_size}")
    print(f"Sampled frames directory: {args.sampled_frames_dir}")
    print(f"Embeddings directory: {embeddings_dir}")
    print()

    # Find all images
    print("Finding all image files...")
    games_images = find_all_images(args.sampled_frames_dir)

    total_images = sum(len(images) for images in games_images.values())
    print(f"Found {total_images} total images across {len(games_images)} games")
    print()

    # Create embeddings directory structure
    print("Creating embeddings directory structure...")
    embeddings_dir = create_embeddings_directory(args.sampled_frames_dir, embeddings_dir)
    print(f"Embeddings will be saved to: {embeddings_dir}")
    print()

    # Initialize extractor based on model type
    if args.model_type == "openclip":
        extractor = EmbeddingExtractor(model_name=args.openclip_model, pretrained=args.openclip_pretrained)
    else:
        extractor = DINOv2EmbeddingExtractor(model_name=args.dinov2_model)
    print()

    # Process each game
    for game_name, image_paths in games_images.items():
        print(f"=== Processing {game_name} ({len(image_paths)} images) ===")

        # Extract embeddings in batches
        embeddings = extractor.extract_batch_embeddings(image_paths, batch_size=args.batch_size)

        # Save embeddings
        print("Saving embeddings...")
        for image_path, embedding in tqdm(embeddings.items(), desc="Saving"):
            save_embedding(embedding, image_path, embeddings_dir)

        print(f"Completed {game_name}: {len(embeddings)}/{len(image_paths)} embeddings saved")
        print()

    print("=== Embedding extraction completed! ===")
    print(f"Embeddings saved to: {embeddings_dir}")
    print(f"Embedding dimension: {extractor.embedding_dim}")


if __name__ == "__main__":
    main()
