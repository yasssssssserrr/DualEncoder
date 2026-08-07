"""Domain gap and representation alignment analysis using Centered Kernel Alignment (CKA) and PCA."""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from sklearn.decomposition import PCA
import torch


def centering(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix K."""
    N = K.shape[0]
    H = np.eye(N) - (1.0 / N) * np.ones((N, N))
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear Centered Kernel Alignment (CKA) between two representation matrices.
    
    Args:
        X: Shape (N, D1) representation matrix 1.
        Y: Shape (N, D2) representation matrix 2.
        
    Returns:
        cka: Float value in [0, 1].
    """
    if X.shape[0] != Y.shape[0]:
        min_n = min(X.shape[0], Y.shape[0])
        X = X[:min_n]
        Y = Y[:min_n]

    # Center features
    X = X - np.mean(X, axis=0, keepdims=True)
    Y = Y - np.mean(Y, axis=0, keepdims=True)

    # Gram matrices
    K = X @ X.T
    L = Y @ Y.T

    # HSIC
    hsic_kl = np.trace(K @ L)
    hsic_kk = np.trace(K @ K)
    hsic_ll = np.trace(L @ L)

    if hsic_kk <= 0 or hsic_ll <= 0:
        return 0.0

    cka = hsic_kl / (np.sqrt(hsic_kk) * np.sqrt(hsic_ll) + 1e-12)
    return float(np.clip(cka, 0.0, 1.0))


def compute_pca_embeddings(
    features_list: list[np.ndarray],
    n_components: int = 3,
) -> Tuple[list[np.ndarray], PCA]:
    """Fit a joint PCA across multiple feature arrays and return projected coordinates.
    
    Args:
        features_list: List of (N_i, D) feature arrays.
        n_components: Number of PCA dimensions (default 3).
        
    Returns:
        projected_list: List of (N_i, n_components) projected coordinates.
        pca_model: Fitted sklearn PCA model.
    """
    flat_list = [f.reshape(f.shape[0], -1) for f in features_list]
    combined = np.vstack(flat_list)

    pca = PCA(n_components=min(n_components, combined.shape[1], combined.shape[0]))
    pca.fit(combined)

    projected_list = [pca.transform(f) for f in flat_list]
    return projected_list, pca
