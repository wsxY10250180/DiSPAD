"""Training-free, normal-reference-only patch feature diffusion."""

import logging
import time

import numpy as np
import torch


class NormalGraphAdapter:
    """One gated diffusion step on raw features; normalized copies define edges.

    With one reference image, neighbors exclude only the patch itself. With
    multiple reference images, they exclude every patch from the same source.
    Queries only read the original normal bank, so predictions do not depend
    on other queries or their order.
    """

    def __init__(self, k=5, alpha=0.2, seed=42, device=None, bank_size=8192,
                 batch_size=256, bank_batch_size=2048, calibration_size=4096,
                 quantile=0.95, eps=1e-8):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("Diffusion k must be a positive integer.")
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Diffusion alpha must be in [0, 1].")
        if not 0 < quantile <= 1 or not np.isfinite(eps) or eps <= 0:
            raise ValueError("Invalid diffusion calibration settings.")
        for value in (bank_size, batch_size, bank_batch_size, calibration_size):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("Diffusion memory limits must be positive integers.")
        self.k = k
        self.alpha = alpha
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.bank_size = bank_size
        self.batch_size = batch_size
        self.bank_batch_size = bank_batch_size
        self.calibration_size = calibration_size
        self.quantile = quantile
        self.eps = eps
        self.threshold_ = None
        self._fitted = False
        self.reset_query_stats()

    @staticmethod
    def _features(features, allow_empty=False):
        X = np.asarray(features)
        if (X.ndim != 2 or not X.shape[1] or
                (not allow_empty and not len(X)) or not np.isfinite(X).all()):
            raise ValueError("Diffusion requires a finite 2D feature matrix.")
        return np.asarray(X, dtype=np.float64 if X.dtype == np.float64 else np.float32)

    @staticmethod
    def _balanced_indices(source_ids, budget, rng):
        """Seeded round-robin sampling keeps every original image represented."""
        groups = np.unique(source_ids)
        if len(groups) > budget:
            raise ValueError("Diffusion bank size must cover every reference image.")
        if len(source_ids) <= budget:
            return np.arange(len(source_ids))
        pools = [rng.permutation(np.flatnonzero(source_ids == group)) for group in groups]
        positions = np.zeros(len(pools), dtype=int)
        selected = []
        while len(selected) < budget:
            for i, pool in enumerate(pools):
                if positions[i] < len(pool):
                    selected.append(pool[positions[i]])
                    positions[i] += 1
                    if len(selected) == budget:
                        break
        return np.sort(selected)

    def _tensor(self, X):
        if any(stride < 0 for stride in X.strides):
            X = X.copy()
        return torch.as_tensor(np.ascontiguousarray(X), dtype=self._bank.dtype, device=self.device)

    def _normalize(self, X):
        norms = torch.linalg.vector_norm(X, dim=1, keepdim=True)
        return X / norms.clamp_min(self.eps), norms[:, 0] > self.eps

    @torch.no_grad()
    def _neighbors(self, X, query_ids=None):
        """Exact top-k over the sampled bank, without a full pairwise matrix."""
        k = min(self.k, len(self._bank))
        distances = torch.full((len(X), k), float("inf"),
                               dtype=self._bank.dtype, device=self.device)
        indices = torch.full((len(X), k), -1, dtype=torch.long, device=self.device)
        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            normalized, valid = self._normalize(self._tensor(X[start:end]))
            best_distances = distances[start:end]
            best_indices = indices[start:end]
            for bank_start in range(0, len(self._bank), self.bank_batch_size):
                bank_end = min(bank_start + self.bank_batch_size, len(self._bank))
                block = (1.0 - normalized @ self._bank_normalized[bank_start:bank_end].T).clamp(0, 2)
                block[:, ~self._bank_valid[bank_start:bank_end]] = float("inf")
                block[~valid] = float("inf")
                if query_ids is not None:
                    excluded = torch.as_tensor(np.array(query_ids[start:end], copy=True),
                                               device=self.device)
                    block.masked_fill_(excluded[:, None] == self._bank_exclusion_ids[None, bank_start:bank_end],
                                       float("inf"))
                values, positions = torch.topk(block, min(k, block.shape[1]),
                                              dim=1, largest=False, sorted=True)
                candidate_distances = torch.cat((best_distances, values), dim=1)
                candidate_indices = torch.cat((best_indices, positions + bank_start), dim=1)
                best_distances, selection = torch.topk(candidate_distances, k, dim=1,
                                                      largest=False, sorted=True)
                best_indices = candidate_indices.gather(1, selection)
            distances[start:end] = best_distances
            indices[start:end] = best_indices
        return distances, indices

    def _calibrate(self, X, source_ids, reference_ids, rng):
        if len(X) < 2:
            self.calibration_mode_ = "insufficient normal patches"
            return
        selected = self._balanced_indices(source_ids,
                                         max(2, self.calibration_size, len(np.unique(source_ids))), rng)
        distances, _ = self._neighbors(X[selected], reference_ids[selected])
        self.calibration_mode_ = ("same-image (self excluded)" if len(np.unique(source_ids)) == 1
                                  else "cross-image")
        values = distances.cpu().numpy()
        counts = np.isfinite(values).sum(axis=1)
        usable = counts > 0
        # Use the kth neighbor, or the last valid neighbor when fewer exist.
        scales = values[usable, counts[usable] - 1]
        if len(scales):
            self.threshold_ = max(float(np.quantile(scales, self.quantile)), self.eps)

    @torch.no_grad()
    def _adapt(self, X, reference_ids=None):
        output = X.copy()
        stats = {"patches": len(X), "adapted": 0, "beta_sum": 0.0, "change_sum": 0.0}
        if self.threshold_ is None or self.alpha == 0:
            return output, stats
        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            query_ids = None if reference_ids is None else reference_ids[start:end]
            distances, indices = self._neighbors(X[start:end], query_ids)
            weights = (1.0 - distances / (self.threshold_ + self.eps)).clamp_min(0)
            active = weights.sum(dim=1) > 0
            if not active.any():
                continue
            weights = weights[active]
            beta = self.alpha * weights.max(dim=1).values
            weights = weights / weights.sum(dim=1, keepdim=True)
            neighbors = self._bank[indices[active].clamp_min(0)]
            average = (weights[:, :, None] * neighbors).sum(dim=1)
            original = self._tensor(X[start:end])[active]
            adapted = (1.0 - beta[:, None]) * original + beta[:, None] * average
            active_rows = np.flatnonzero(active.cpu().numpy()) + start
            output[active_rows] = adapted.cpu().numpy()
            stats["adapted"] += len(active_rows)
            stats["beta_sum"] += float(beta.sum())
            stats["change_sum"] += float(torch.linalg.vector_norm(adapted - original, dim=1).sum())
        return output, stats

    @torch.no_grad()
    def fit_transform(self, features, source_ids, *, adapt_references=True):
        """Calibrate on normal references and optionally diffuse them."""
        started = time.perf_counter()
        X = self._features(features)
        source_ids = np.asarray(source_ids)
        if source_ids.shape != (len(X),) or not np.issubdtype(source_ids.dtype, np.integer):
            raise ValueError("Provide one integer original-image ID per reference patch.")
        self.threshold_ = None
        self._fitted = False
        self.reset_query_stats()
        rng = np.random.default_rng(self.seed)
        self.bank_indices_ = self._balanced_indices(source_ids, self.bank_size, rng)
        bank = X[self.bank_indices_].copy()
        self._bank = torch.as_tensor(bank, device=self.device)
        self._bank_normalized, self._bank_valid = self._normalize(self._bank)
        # Augmented views/crops share source_ids and do not count as extra shots.
        reference_ids = np.arange(len(X)) if len(np.unique(source_ids)) == 1 else source_ids
        self._bank_exclusion_ids = torch.as_tensor(reference_ids[self.bank_indices_], device=self.device)
        self._calibrate(X, source_ids, reference_ids, rng)
        if adapt_references:
            adapted, stats = self._adapt(X, reference_ids)
        else:
            adapted = X
            stats = {"patches": len(X), "adapted": 0, "beta_sum": 0.0, "change_sum": 0.0}
            logging.info("Graph diffusion: test features only.")
        self.reference_stats_ = stats
        self._fitted = True
        logging.info(
            "Graph diffusion: device=%s, bank=%d/%d, k=%d, alpha=%.3f, "
            "threshold=%s, calibration=%s, reference adapted=%.2f%%, time=%.3fs",
            self.device, len(bank), len(X), self.k, self.alpha, self.threshold_,
            self.calibration_mode_, 100 * stats["adapted"] / len(X), time.perf_counter() - started,
        )
        return adapted

    def transform(self, features):
        if not self._fitted:
            raise ValueError("Fit the normal graph adapter before transforming queries.")
        X = self._features(features, allow_empty=True)
        if X.shape[1] != self._bank.shape[1]:
            raise ValueError("Query and reference feature dimensions must match.")
        adapted, stats = self._adapt(X)
        for key, value in stats.items():
            self.query_stats_[key] += value
        return adapted

    def reset_query_stats(self):
        self.query_stats_ = {"patches": 0, "adapted": 0, "beta_sum": 0.0, "change_sum": 0.0}

    def log_query_stats(self):
        stats = self.query_stats_
        count = max(1, stats["patches"])
        logging.info("Graph diffusion queries: patches=%d, adapted=%.2f%%, "
                     "mean strength=%.6f, mean L2 change=%.6f",
                     stats["patches"], 100 * stats["adapted"] / count,
                     stats["beta_sum"] / count, stats["change_sum"] / count)
