"""Global PCA of center features minus their spatial-neighbor mean."""

import logging
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .pca import PCAModel


@lru_cache(maxsize=16)
def _random_neighbor_indices(height, width, seed):
    """Sample distinct indices outside each center's clipped 3x3 window."""
    rng = np.random.default_rng(seed)
    indices = np.full((8, height * width), -1, dtype=np.int64)
    for center in range(height * width):
        row, col = divmod(center, width)
        window = [r * width + c
                  for r in range(max(0, row - 1), min(height, row + 2))
                  for c in range(max(0, col - 1), min(width, col + 2))]
        count = len(window) - 1
        available = height * width - len(window)
        if available < count:
            raise ValueError("Random-neighborhood PCA needs enough patches outside each 3x3 window "
                             f"to match its neighbor count without replacement (got {height}x{width}).")
        sampled = rng.choice(available, size=count, replace=False)
        # Convert ranks in the complement back to flat patch indices.
        for excluded in window:
            sampled += sampled >= excluded
        indices[:count, center] = sampled
    return indices


@torch.no_grad()
def spatial_difference_features(feature_maps, neighborhood="spatial", seed=42):
    """Compute 3x3 center-minus-neighbor-mean features, excluding the center.

    Each [H,W] map is independent. Borders use only existing neighbors (three
    at corners, five at edges); padding never contributes artificial neighbors.
    Preserve NumPy/tensor input type and the spatial layout.
    """
    features = torch.as_tensor(feature_maps)
    if features.ndim != 4 or not all(features.shape) or features.shape[1] * features.shape[2] < 2:
        raise ValueError("Spatial PCA requires [B,H,W,D] maps with at least two spatial patches.")
    if features.dtype not in (torch.float32, torch.float64):
        features = features.float()
    if neighborhood == "random":
        batch, height, width, channels = features.shape
        indices = torch.as_tensor(_random_neighbor_indices(height, width, seed), device=features.device)
        flat = features.reshape(batch, height * width, channels)
        neighbor_sum = torch.zeros_like(flat)
        for index in indices:
            values = flat.index_select(1, index.clamp_min(0))
            neighbor_sum += values.masked_fill_(index[None, :, None] < 0, 0)
        neighbor_count = (indices >= 0).sum(dim=0).reshape(1, -1, 1)
        differences = (flat - neighbor_sum / neighbor_count).reshape(features.shape)
        return differences if torch.is_tensor(feature_maps) else differences.numpy()
    if neighborhood != "spatial":
        raise ValueError(f"Unknown PCA neighborhood: {neighborhood}")
    channels_first = features.permute(0, 3, 1, 2)
    neighbor_sum = F.avg_pool2d(channels_first, 3, stride=1, padding=1,
                                divisor_override=1) - channels_first
    ones = features.new_ones((1, 1, features.shape[1], features.shape[2]))
    neighbor_count = F.avg_pool2d(ones, 3, stride=1, padding=1, divisor_override=1) - 1
    differences = (channels_first - neighbor_sum / neighbor_count).permute(0, 2, 3, 1).contiguous()
    return differences if torch.is_tensor(feature_maps) else differences.numpy()


class SpatialDifferencePCA:
    """Collect sufficient statistics once and reuse the existing global PCA.

    Batch-merged centered covariance needs O(D^2) storage, without retaining a
    reference bank. Select centers after constructing their complete context.
    """

    def __init__(self, ev=0.99, device=None, neighborhood="spatial", seed=42):
        if not 0 < ev <= 1:
            raise ValueError("Spatial PCA explained variance must be in (0, 1].")
        self.model = PCAModel(ev=ev)
        if device is not None:
            self.model.device = torch.device(device)
        self.neighborhood = neighborhood
        self.seed = seed
        self.count = 0
        self.mean = None
        self.scatter = None

    def add_diffused(self, feature_maps, adapted_centers, adapter, selection=None, source_ids=None):
        """Restore reference layouts from the existing diffusion output.

        Selected centers reuse fit_transform results exactly. Masked neighbors
        also diffuse against the fitted normal bank, with same-image exclusion
        when source_ids is given. They supply context but never enter PCA stats.
        """
        if selection is None:
            maps = adapted_centers.reshape(feature_maps.shape)
        else:
            maps = (feature_maps.detach().cpu().numpy() if torch.is_tensor(feature_maps)
                    else np.asarray(feature_maps)).copy()
            flat = maps.reshape(-1, maps.shape[-1])
            selection = np.asarray(selection, dtype=bool).reshape(-1)
            flat[selection] = adapted_centers
            missing = ~selection
            if missing.any():
                ids = (None if source_ids is None else
                       np.broadcast_to(source_ids, (len(flat),))[missing])
                flat[missing], _ = adapter._adapt(flat[missing], ids)
        self.add(maps, selection)

    @torch.no_grad()
    def add(self, feature_maps, selection=None):
        differences = spatial_difference_features(feature_maps, self.neighborhood, self.seed)
        batch = torch.as_tensor(differences, device=self.model.device, dtype=self.model.dtype)
        batch = batch.reshape(-1, batch.shape[-1])
        if selection is not None:
            batch = batch[torch.as_tensor(selection, dtype=torch.bool, device=batch.device)]
        if not len(batch):
            return
        batch_mean = batch.mean(dim=0)
        centered = batch - batch_mean
        batch_scatter = centered.T @ centered
        if self.count == 0:
            self.mean, self.scatter = batch_mean, batch_scatter
        else:
            delta = batch_mean - self.mean
            total = self.count + len(batch)
            self.scatter += batch_scatter + torch.outer(delta, delta) * (self.count * len(batch) / total)
            self.mean += delta * (len(batch) / total)
        self.count += len(batch)

    @torch.no_grad()
    def fit(self):
        if self.count < 2:
            raise ValueError("Spatial PCA requires at least two selected normal reference patches.")
        model = self.model
        model.mu_ = self.mean
        evecs = model._compute_eigendecomposition(self.scatter / (self.count - 1))
        model.explained_variance_.clamp_min_(0)
        model.k = None
        if not (model.explained_variance_.sum() > 0):
            model.k = 0  # Constant normal differences: score departures from the mean.
        model._select_k_components(evecs)
        logging.info("Spatial difference PCA: references=%d, feature_dim=%d, components=%d, ev=%g, neighborhood=%s.",
                     self.count, len(self.mean), model.k, model.ev_ratio, self.neighborhood)
        params = model._build_pca_params()
        params.update(neighborhood=self.neighborhood, neighborhood_seed=self.seed)
        return params
