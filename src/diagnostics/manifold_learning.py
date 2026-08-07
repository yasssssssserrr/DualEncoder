"""t-SNE and UMAP 2D manifold dimensionality reduction for encoder embeddings."""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from sklearn.manifold import TSNE


@dataclass
class ManifoldProjections:
    tsne_2d: np.ndarray          # Shape (N, 2)
    umap_2d: Optional[np.ndarray] # Shape (N, 2) if umap-learn installed, else PCA
    frame_indices: np.ndarray    # Shape (N,)
    sweep_labels: np.ndarray     # Shape (N,)


def compute_tsne_umap_projections(
    features: np.ndarray,
    frame_indices: np.ndarray,
    sweep_labels: Optional[np.ndarray] = None,
    perplexity: float = 15.0,
    random_state: int = 42,
) -> ManifoldProjections:
    """Compute 2D t-SNE and UMAP projections of high-dimensional encoder features."""
    N = len(features)
    
    # Adjust perplexity if N is small
    effective_perp = min(perplexity, max(2.0, (N - 1) / 3.0))
    
    # 1. t-SNE
    tsne = TSNE(n_components=2, perplexity=effective_perp, random_state=random_state, max_iter=1000)
    tsne_2d = tsne.fit_transform(features)

    # 2. UMAP (fallback to PCA or Multidimensional Scaling if umap not installed)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=min(15, N - 1))
        umap_2d = reducer.fit_transform(features)
    except (ImportError, Exception):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        umap_2d = pca.fit_transform(features)

    if sweep_labels is None:
        sweep_labels = np.zeros(N, dtype=int)

    return ManifoldProjections(
        tsne_2d=tsne_2d,
        umap_2d=umap_2d,
        frame_indices=frame_indices,
        sweep_labels=sweep_labels,
    )
