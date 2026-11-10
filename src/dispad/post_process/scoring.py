import cv2
import numpy as np
import torch
import logging
from ..utils.common import topk_mean
from ..core.spatial_pca import spatial_difference_features


def aggregate_image_score(anomaly_map: np.ndarray, method: str) -> float:
    """
    Aggregates a pixel-level anomaly map into a single image-level score.
    """
    if method == "max":
        return float(np.max(anomaly_map))
    elif method == "p99":
        return float(np.percentile(anomaly_map, 99))
    elif method == "mtop5":
        return float(np.mean(np.sort(anomaly_map.flatten())[-5:]))
    elif method == "mtop1p":
        return topk_mean(anomaly_map, frac=0.01)
    elif method == "mean":
        return float(np.mean(anomaly_map))
    else:
        logging.warning(
            f"Unknown image score aggregation '{method}'. Defaulting to 'mean'."
        )
        return float(np.mean(anomaly_map))


def _kernel_self_dot(X: np.ndarray, kpca) -> np.ndarray:
    """Compute k(x,x) for several sklearn KPCA kernels."""
    if kpca.kernel in ("rbf", "cosine"):
        return np.ones(X.shape[0])
    elif kpca.kernel == "linear":
        return np.sum(X**2, axis=1) + kpca.coef0
    elif kpca.kernel == "poly":
        gamma = kpca.gamma if kpca.gamma is not None else 1.0 / X.shape[1]
        return (gamma * np.sum(X**2, axis=1) + kpca.coef0) ** kpca.degree
    elif kpca.kernel == "sigmoid":
        gamma = kpca.gamma if kpca.gamma is not None else 1.0 / X.shape[1]
        return np.tanh(gamma * np.sum(X**2, axis=1) + kpca.coef0)
    else:
        logging.warning(
            f"Cannot compute k(x,x) for kernel '{kpca.kernel}'. Reconstruction error will be approximate."
        )
        return np.zeros(X.shape[0], dtype=X.dtype)


def _row_l2(X: np.ndarray, eps: float) -> np.ndarray:
    """Normalize rows to unit length."""
    n = (torch.linalg.vector_norm(X, dim=1, keepdim=True) if torch.is_tensor(X)
         else np.linalg.norm(X, axis=1, keepdims=True))
    return X / (n + eps)


def pca_reconstruct(X: np.ndarray, pca: dict, drop_k: int = 0) -> np.ndarray:
    """Reconstruct X in original space using unscaled components."""
    if torch.is_tensor(X):
        mu = torch.as_tensor(pca["mu"], dtype=X.dtype, device=X.device)
        C = torch.as_tensor(pca["components"][:, : pca["k"]], dtype=X.dtype, device=X.device)
    else:
        mu = np.asarray(pca["mu"], dtype=X.dtype)
        C = np.asarray(pca["components"][:, : pca["k"]], dtype=X.dtype)
    X0 = X - mu
    Z = X0 @ C
    if drop_k > 0:
        if drop_k >= Z.shape[1]:
            Z[:] = 0.0
        else:
            Z[:, :drop_k] = 0.0
    X_recon = (Z @ C.T) + mu
    return X_recon


def _calculate_kpca_scores(X: np.ndarray, pca: dict, drop_k: int = 0):
    """Calculates anomaly scores for Kernel PCA."""
    scaler = pca["scaler"]
    kpca = pca["kpca"]
    X_scaled = scaler.transform(X)

    X_proj = kpca.transform(X_scaled)
    k_x_x = _kernel_self_dot(X_scaled, kpca)

    if drop_k > 0:
        if drop_k >= X_proj.shape[1]:
            X_proj = np.zeros_like(X_proj)
        else:
            X_proj = X_proj[:, drop_k:]

    proj_norm_sq = np.sum(X_proj**2, axis=1)
    score = k_x_x - proj_norm_sq
    return np.maximum(0.0, score)


def _calculate_pca_scores_torch(X, pca, method, drop_k):
    """Same scoring formulas as NumPy, without leaving the tensor's device."""
    if method in ("reconstruction", "cosine"):
        reconstructed = pca_reconstruct(X, pca, drop_k)
        if method == "reconstruction":
            return (X - reconstructed).square().sum(dim=1)
        similarity = (_row_l2(X, pca["eps"]) * _row_l2(reconstructed, pca["eps"])).sum(dim=1)
        return 1.0 - similarity.clamp(-1.0, 1.0)
    if method in ("mahalanobis", "euclidean"):
        mu = torch.as_tensor(pca["mu"], dtype=X.dtype, device=X.device)
        components = torch.as_tensor(pca["components"][:, drop_k:pca["k"]],
                                     dtype=X.dtype, device=X.device)
        projected_sq = ((X - mu) @ components).square()
        if method == "mahalanobis":
            eigvals = torch.as_tensor(pca["eigvals"][drop_k:], dtype=X.dtype, device=X.device)
            projected_sq = projected_sq / (eigvals + pca["eps"])
        return projected_sq.sum(dim=1)
    raise ValueError(f"Unknown scoring method '{method}'.")


def _calculate_pca_scores(X: np.ndarray, pca: dict, method: str, drop_k: int = 0):
    """Calculates anomaly scores for standard PCA."""
    if drop_k < 0:
        raise ValueError("drop_k must be non-negative.")
    if drop_k > 0 and drop_k >= pca["k"]:
        logging.warning(f"drop_k ({drop_k}) is >= num components ({pca['k']}).")
        if method in ("mahalanobis", "euclidean"):
            return X.new_zeros(len(X)) if torch.is_tensor(X) else np.zeros(X.shape[0], dtype=X.dtype)

    if torch.is_tensor(X):
        return _calculate_pca_scores_torch(X, pca, method, drop_k)

    if method == "reconstruction":
        X_recon = pca_reconstruct(X, pca, drop_k=drop_k)
        return np.sum((X - X_recon) ** 2, axis=1)

    elif method == "mahalanobis":
        mu = np.asarray(pca["mu"], dtype=X.dtype)
        C = np.asarray(pca["components"][:, : pca["k"]], dtype=X.dtype)
        Z = (X - mu) @ C  # [N, k]

        if drop_k >= pca["k"]:
            return np.zeros(X.shape[0], dtype=X.dtype)

        Z_abnormal = Z[:, drop_k:]
        eigvals_abnormal = np.asarray(pca["eigvals"][drop_k:], dtype=X.dtype)
        cov_inv = np.diag(1.0 / (eigvals_abnormal + pca["eps"]))
        return np.einsum("ij,jk,ik->i", Z_abnormal, cov_inv, Z_abnormal)

    elif method == "euclidean":
        mu = np.asarray(pca["mu"], dtype=X.dtype)
        C = np.asarray(pca["components"][:, : pca["k"]], dtype=X.dtype)
        Z = (X - mu) @ C  # [N, k]

        if drop_k >= pca["k"]:
            return np.zeros(X.shape[0], dtype=X.dtype)

        Z_abnormal = Z[:, drop_k:]
        return np.sum(Z_abnormal**2, axis=1)

    elif method == "cosine":
        X_recon = pca_reconstruct(X, pca, drop_k=drop_k)
        X_norm = _row_l2(X, pca["eps"])
        X_recon_norm = _row_l2(X_recon, pca["eps"])
        sim = np.einsum("ij,ij->i", X_norm, X_recon_norm)
        sim = np.clip(sim, -1.0, 1.0)
        return 1.0 - sim

    else:
        raise ValueError(f"Unknown scoring method '{method}'.")


@torch.no_grad()
def _calculate_nearest_neighbor_scores(X, references, batch_size=256, bank_batch_size=4096):
    """Exact cosine 1-NN scores (1 - similarity) / 2 in [0, 1], in bounded blocks.

    Zero vectors have cosine similarity 0. Normalize reference blocks on demand
    to avoid retaining a second full-sized feature bank.
    """
    score_device = (references.device if torch.is_tensor(references) else
                    X.device if torch.is_tensor(X) else torch.device("cpu"))
    queries = torch.as_tensor(X, device=score_device)
    if queries.dtype not in (torch.float32, torch.float64):
        queries = queries.float()
    references = torch.as_tensor(references, dtype=queries.dtype, device=score_device)
    if references.ndim != 2 or not all(references.shape):
        raise ValueError("Nearest-neighbor scoring requires a nonempty 2D reference bank.")
    if queries.ndim != 2 or queries.shape[1] != references.shape[1]:
        raise ValueError("Query and reference feature dimensions must match.")
    queries = torch.nn.functional.normalize(queries, dim=1, eps=1e-12)
    reference_norms = torch.linalg.vector_norm(references, dim=1, keepdim=True).clamp_min(1e-12)
    scores = queries.new_empty(len(queries))
    for start in range(0, len(queries), batch_size):
        query = queries[start:start + batch_size]
        nearest = query.new_full((len(query),), float("inf"))
        for bank_start in range(0, len(references), bank_batch_size):
            block = (references[bank_start:bank_start + bank_batch_size] /
                     reference_norms[bank_start:bank_start + bank_batch_size])
            similarity = query @ block.T
            distances = ((1.0 - similarity) / 2.0).clamp_(0.0, 1.0)
            torch.minimum(nearest, distances.min(dim=1).values, out=nearest)
        scores[start:start + len(query)] = nearest
    return scores


@torch.no_grad()
def _calculate_base_anomaly_scores(X: np.ndarray, pca: dict, method: str, drop_k: int = 0,
                                   apply_diffusion=True):
    """
    Calculates anomaly scores using PCA, KernelPCA, or direct nearest-neighbor search.
    Acts as a router to the appropriate scoring function.
    CUDA PCA parameters keep all score calculations on GPU; only final scores
    return to NumPy for the existing image post-processing and evaluation code.
    """
    if apply_diffusion and "feature_adapter" in pca:
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()  # The optional graph adapter has a NumPy interface.
        X = pca["feature_adapter"].transform(X)
    if method == "nearest_neighbor":
        if "reference_features" not in pca:
            raise ValueError("Collect normal reference features before nearest-neighbor scoring.")
        return _calculate_nearest_neighbor_scores(X, pca["reference_features"]).cpu().numpy()
    if "kpca" in pca:
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        if method != "reconstruction":
            logging.warning(
                "Kernel PCA only supports 'reconstruction' scoring. Using 'reconstruction'."
            )
        return _calculate_kpca_scores(X, pca, drop_k)
    else:
        if torch.is_tensor(pca["mu"]) or torch.is_tensor(X):
            score_device = pca["mu"].device if torch.is_tensor(pca["mu"]) else X.device
            X = torch.as_tensor(X, device=score_device)
        scores = _calculate_pca_scores(X, pca, method, drop_k)
        return scores.cpu().numpy() if torch.is_tensor(scores) else scores


@torch.no_grad()
def calculate_anomaly_scores(X: np.ndarray, pca: dict, method: str, drop_k: int = 0):
    """Fuse original scores with spatial-difference reconstruction scores.

    Spatial scoring needs [B,H,W,D] maps; legacy flat inputs remain supported
    when disabled. Both branches share one optional graph diffusion step.
    """
    maps = X if X.ndim == 4 else None
    features = X.reshape(-1, X.shape[-1]) if maps is not None else X
    weight = pca.get("spatial_pca_weight", 0.0)
    if weight == 0:
        return _calculate_base_anomaly_scores(features, pca, method, drop_k)
    if maps is None:
        raise ValueError("Spatial PCA scoring requires [B,H,W,D] maps before flattening.")
    if "feature_adapter" in pca:
        if torch.is_tensor(features):
            features = features.detach().cpu().numpy()
        features = pca["feature_adapter"].transform(features)
    spatial_pca = pca["spatial_pca"]
    differences = spatial_difference_features(
        features.reshape(maps.shape), spatial_pca.get("neighborhood", "spatial"),
        spatial_pca.get("neighborhood_seed", 42),
    ).reshape(-1, maps.shape[-1])
    spatial_scores = _calculate_base_anomaly_scores(differences, spatial_pca, "reconstruction")
    if weight == 1:
        return spatial_scores
    original_scores = _calculate_base_anomaly_scores(features, pca, method, drop_k,
                                                     apply_diffusion=False)
    return (1.0 - weight) * original_scores + weight * spatial_scores


def post_process_map(
    anomaly_map: np.ndarray,
    res,
    blur: bool = True,
    close_holes: bool = False,
    close_k_size: int = 5,
):
    """Resize, blur, and optionally close holes in the anomaly map."""
    if anomaly_map.dtype != np.float32:
        anomaly_map = anomaly_map.astype(np.float32)

    dsize = (res, res) if isinstance(res, int) else (res[1], res[0])
    map_resized = cv2.resize(anomaly_map, dsize, interpolation=cv2.INTER_LINEAR)

    if close_holes:
        if close_k_size % 2 == 0:
            close_k_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k_size, close_k_size))
        map_resized = cv2.morphologyEx(map_resized, cv2.MORPH_CLOSE, kernel)
    if blur:
        sigma = 4.0
        k_size = 3
        return cv2.GaussianBlur(map_resized, (k_size, k_size), sigma)
    else:
        return map_resized
