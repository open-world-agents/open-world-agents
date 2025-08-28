#!/usr/bin/env python3
"""
Script to visualize image embeddings using t-SNE.
Loads embeddings from ./embeddings directory and creates static visualizations.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union, Literal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except ImportError:
    print("Error: scikit-learn is not installed. Please install it with:")
    print("pip install scikit-learn")
    sys.exit(1)


class EmbeddingVisualizer:
    """Visualize image embeddings using t-SNE and other dimensionality reduction techniques."""

    def __init__(self, embeddings_dir: str = "./embeddings"):
        """
        Initialize the visualizer.

        Args:
            embeddings_dir: Path to the embeddings directory
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings = {}
        self.labels = {}
        self.game_names = []

        if not self.embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory {embeddings_dir} does not exist")

    def load_embeddings(
        self, max_per_game: Optional[int] = None, sample_randomly: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load embeddings from all games.

        Args:
            max_per_game: Maximum number of embeddings to load per game (None for all)
            sample_randomly: Whether to sample randomly or take first N embeddings

        Returns:
            Tuple of (embeddings_array, labels_array, file_paths)
        """
        print("Loading embeddings...")

        all_embeddings = []
        all_labels = []
        all_paths = []

        # Get all game directories
        game_dirs = [d for d in self.embeddings_dir.iterdir() if d.is_dir()]
        self.game_names = [d.name for d in game_dirs]

        print(f"Found {len(game_dirs)} games: {', '.join(self.game_names)}")

        for game_idx, game_dir in enumerate(tqdm(game_dirs, desc="Loading games")):
            game_name = game_dir.name

            # Find all .npy files in the game directory
            embedding_files = list(game_dir.glob("*.npy"))

            if max_per_game and len(embedding_files) > max_per_game:
                if sample_randomly:
                    np.random.seed(42)  # For reproducibility
                    embedding_files = np.random.choice(embedding_files, max_per_game, replace=False)
                else:
                    embedding_files = embedding_files[:max_per_game]

            print(f"Loading {len(embedding_files)} embeddings from {game_name}")

            # Load embeddings for this game
            game_embeddings = []
            game_paths = []

            for emb_file in tqdm(embedding_files, desc=f"Loading {game_name}", leave=False):
                try:
                    embedding = np.load(emb_file)
                    game_embeddings.append(embedding)
                    game_paths.append(str(emb_file))
                except Exception as e:
                    print(f"Error loading {emb_file}: {e}")
                    continue

            if game_embeddings:
                all_embeddings.extend(game_embeddings)
                all_labels.extend([game_idx] * len(game_embeddings))
                all_paths.extend(game_paths)

        embeddings_array = np.array(all_embeddings)
        labels_array = np.array(all_labels)

        print(f"Loaded {len(embeddings_array)} total embeddings")
        print(f"Embedding shape: {embeddings_array.shape}")

        return embeddings_array, labels_array, all_paths

    def apply_pca(self, embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
        """
        Apply PCA for initial dimensionality reduction before t-SNE.

        Args:
            embeddings: Input embeddings
            n_components: Number of PCA components

        Returns:
            PCA-reduced embeddings
        """
        print(f"Applying PCA to reduce to {n_components} dimensions...")
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings)

        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"PCA explained variance ratio: {explained_variance:.3f}")

        return embeddings_pca

    def apply_tsne(
        self,
        embeddings: np.ndarray,
        perplexity: int = 30,
        max_iter: int = 1000,
        learning_rate: Union[Literal["auto"], float] = "auto",
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Apply t-SNE dimensionality reduction.

        Args:
            embeddings: Input embeddings
            perplexity: t-SNE perplexity parameter
            max_iter: Maximum number of iterations
            learning_rate: Learning rate ('auto' or float)
            random_state: Random state for reproducibility

        Returns:
            2D t-SNE embeddings
        """
        print(f"Applying t-SNE with perplexity={perplexity}, max_iter={max_iter}...")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=1,
        )

        embeddings_2d = tsne.fit_transform(embeddings)
        return embeddings_2d

    def plot_matplotlib(
        self,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        save_path: str = "tsne_visualization.png",
        figsize: Tuple[int, int] = (12, 8),
    ):
        """
        Create a matplotlib visualization of the t-SNE results.

        Args:
            embeddings_2d: 2D t-SNE embeddings
            labels: Game labels
            save_path: Path to save the plot
            figsize: Figure size
        """
        print("Creating matplotlib visualization...")

        plt.figure(figsize=figsize)

        # Create a color palette
        colors = sns.color_palette("husl", len(self.game_names))

        # Plot each game with different colors
        for game_idx, game_name in enumerate(self.game_names):
            mask = labels == game_idx
            if np.any(mask):
                plt.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[colors[game_idx]],
                    label=game_name,
                    alpha=0.7,
                    s=20,
                )

        # plt.title("t-SNE Visualization of Game Frame Embeddings", fontsize=32)
        plt.legend(loc="lower left", fontsize=16)

        # Remove x and y axis tick labels (grid numbers)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Matplotlib plot saved to: {save_path}")
        plt.show()


def main():
    """Main function to run the t-SNE visualization."""
    parser = argparse.ArgumentParser(description="Visualize game frame embeddings using t-SNE")
    parser.add_argument("--embeddings-dir", default="./embeddings", help="Path to embeddings directory")
    parser.add_argument(
        "--max-per-game", type=int, default=None, help="Maximum number of embeddings per game (default: all)"
    )
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity parameter")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum number of t-SNE iterations")
    parser.add_argument("--pca-components", type=int, default=50, help="Number of PCA components for pre-processing")
    parser.add_argument("--output-dir", default="./visualizations", help="Directory to save visualizations")
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA preprocessing")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=== t-SNE Visualization of Game Frame Embeddings ===")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Max per game: {args.max_per_game}")
    print(f"Perplexity: {args.perplexity}")
    print(f"Max iterations: {args.max_iter}")
    print(f"PCA components: {args.pca_components}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Initialize visualizer
    visualizer = EmbeddingVisualizer(args.embeddings_dir)

    # Load embeddings
    embeddings, labels, _ = visualizer.load_embeddings(max_per_game=args.max_per_game, sample_randomly=True)

    # Apply PCA preprocessing if requested
    if not args.no_pca and embeddings.shape[1] > args.pca_components:
        embeddings = visualizer.apply_pca(embeddings, args.pca_components)

    # Apply t-SNE
    embeddings_2d = visualizer.apply_tsne(embeddings, perplexity=args.perplexity, max_iter=args.max_iter)

    # Create visualizations
    print("\n=== Creating Visualizations ===")

    # Matplotlib plot
    matplotlib_path = output_dir / "tsne_visualization.png"
    visualizer.plot_matplotlib(embeddings_2d, labels, str(matplotlib_path))

    print("\n=== Visualization Complete! ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
