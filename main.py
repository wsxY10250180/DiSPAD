import os
import hashlib
import json
import math
import logging
from pathlib import Path, PureWindowsPath

import numpy as np
import random
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import label as cc_label, generate_binary_structure
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)

from src.dispad.config import get_args, parse_layer_indices, parse_grouped_layers
from src.dispad.utils.common import (
    setup_logging,
    save_config,
    min_max_norm,
)
from src.dispad.utils.performance import synchronized_time, write_performance_stats
from src.dispad.data.datasets import get_dataset_handler
from src.dispad.core.extractor import FeatureExtractor
from src.dispad.core.pca import PCAModel, KernelPCAModel
from src.dispad.core.graph_adapter import NormalGraphAdapter
from src.dispad.core.spatial_pca import SpatialDifferencePCA
from src.dispad.post_process.scoring import (
    calculate_anomaly_scores,
    post_process_map,
)
from src.dispad.utils.viz import save_visualization, save_overlay_for_intro
from src.dispad.post_process.specular import (
    specular_mask_torch,
    filter_specular_anomalies,
)
from src.dispad.core.patching import process_image_patched, get_patch_coords
from src.dispad.data.transforms import get_augmentation_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MVTEC_PCA_MASK_CATEGORIES = {"capsule", "hazelnut", "pill", "screw", "toothbrush"}

print(f"Using device: {DEVICE}")


def _best_f1_threshold_from_scores(y_true, y_score):
    """Return threshold maximizing F1 on validation scores."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or y_score.size == 0 or (y_true.max() == y_true.min()):
        return None, 0.0
    p, r, t = precision_recall_curve(y_true, y_score)
    if t.size == 0:
        return None, 0.0
    f1 = (2 * p[:-1] * r[:-1]) / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i])


def _quantile_threshold_from_negatives(y_true, y_score, target_fpr=0.01):
    """
    Fallback: pick threshold so that ~target_fpr of NEGATIVES exceed it.
    y_true in {0,1}, negatives are 0. Returns None if no negatives.
    """
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    neg = y_score[y_true == 0]
    if neg.size == 0:
        return None
    q = np.clip(1.0 - float(target_fpr), 0.0, 1.0)
    return float(np.quantile(neg, q, interpolation="linear"))


def _pick_threshold_with_fallback(y_true, y_score, target_fpr):
    """
    Try PR-optimal F1; if degenerate (single-class), fall back to negative-quantile.
    Returns (thr, how), where how ∈ {"pr", "quantile", "none"}.
    """
    thr_pr, _ = _best_f1_threshold_from_scores(y_true, y_score)
    if thr_pr is not None:
        return thr_pr, "pr"
    thr_q = _quantile_threshold_from_negatives(y_true, y_score, target_fpr)
    if thr_q is not None:
        return thr_q, "quantile"
    return None, "none"


def topk_mean(arr, frac=0.01):
    flat = arr.ravel()
    k = max(1, int(len(flat) * frac))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def compute_aupro(
    anomaly_maps,
    gt_masks,
    fpr_limit: float = 0.3,
    num_thresholds: int = 300,
    connectivity: int = 8,
):
    """
    MVTec-AD AUPRO (Bergmann et al.).

    Args:
        anomaly_maps: list/array of [H, W] float prediction maps. Higher = more anomalous.
        gt_masks:     list/array of [H, W] binary masks (uint8/bool). 1 = anomaly.
        fpr_limit:    integration upper bound on Set FPR. MVTec convention: 0.3.
        num_thresholds: number of FPR-linspaced thresholds inside [0, fpr_limit].
        connectivity: 4 or 8 for connected components.

    Returns:
        AUPRO in [0, 1] (perfect detector = 1.0). NaN if undefined.
    """
    preds = [np.asarray(p, dtype=np.float32) for p in anomaly_maps]
    gts = [np.asarray(g, dtype=np.uint8) for g in gt_masks]
    if len(preds) != len(gts) or any(pred.shape != gt.shape for pred, gt in zip(preds, gts)):
        raise ValueError("Anomaly maps and ground-truth masks must have matching shapes.")

    if any(not np.isfinite(pred).all() for pred in preds):
        return float("nan")

    # 1. Collect sorted scores inside each connected region.
    structure = generate_binary_structure(2, 2 if connectivity == 8 else 1)
    region_sorted_scores = []  # list of 1D arrays, one per region
    for pred, gt in zip(preds, gts):
        if not gt.any():
            continue
        labeled, n = cc_label(gt, structure=structure)
        for r in range(1, n + 1):
            region_scores = pred[labeled == r]
            region_scores.sort()
            region_sorted_scores.append(region_scores)

    if len(region_sorted_scores) == 0:
        return float("nan")

    # 2. Sort background scores in one compact buffer.
    n_neg = sum(gt.size - np.count_nonzero(gt) for gt in gts)
    if n_neg == 0:
        return float("nan")
    neg_sorted = np.empty(n_neg, dtype=np.float32)
    offset = 0
    for pred, gt in zip(preds, gts):
        background_scores = pred[gt == 0]
        end = offset + background_scores.size
        neg_sorted[offset:end] = background_scores
        offset = end
    neg_sorted.sort()

    # 3. Pick thresholds that are linear in FPR over [0, fpr_limit].
    target_fprs = np.linspace(0.0, fpr_limit, num_thresholds + 1)[1:]  # exclude 0
    # quantile(1 - f): use sorted neg array directly for stability
    q_idx = np.clip(
        np.floor((1.0 - target_fprs) * (n_neg - 1)).astype(np.int64), 0, n_neg - 1
    )
    thresholds = neg_sorted[q_idx]  # shape: [num_thresholds]

    # 4. For each threshold, compute realized FPR and mean PRO.
    fp_counts = n_neg - np.searchsorted(neg_sorted, thresholds, side="left")
    fprs = fp_counts.astype(np.float64) / n_neg

    # Mean PRO across regions, vectorized via searchsorted on each region's sorted scores.
    pros_accum = np.zeros(num_thresholds, dtype=np.float64)
    for region_scores in region_sorted_scores:
        rs = region_scores
        area = rs.size
        # for each t: overlap = (rs >= t).sum() / area
        ge_counts = rs.size - np.searchsorted(rs, thresholds, side="left")
        pros_accum += ge_counts / area
    pros = pros_accum / len(region_sorted_scores)

    # 5. Sort by FPR (should already be ascending up to ties), prepend (0, 0) anchor.
    order = np.argsort(fprs, kind="stable")
    fprs_s = np.concatenate([[0.0], fprs[order]])
    pros_s = np.concatenate([[0.0], pros[order]])

    # 6. Clip strictly to [0, fpr_limit] with linear interpolation at the boundary.
    if fprs_s[-1] > fpr_limit:
        cut = np.searchsorted(fprs_s, fpr_limit, side="right")
        # linear interp between fprs_s[cut-1] and fprs_s[cut] at x=fpr_limit
        f0, f1 = fprs_s[cut - 1], fprs_s[cut]
        p0, p1 = pros_s[cut - 1], pros_s[cut]
        p_at = p0 + (p1 - p0) * (fpr_limit - f0) / (f1 - f0) if f1 > f0 else p0
        fprs_s = np.concatenate([fprs_s[:cut], [fpr_limit]])
        pros_s = np.concatenate([pros_s[:cut], [p_at]])
    elif fprs_s[-1] < fpr_limit:
        # didn't reach fpr_limit (rare): extrapolate flat from last point
        fprs_s = np.concatenate([fprs_s, [fpr_limit]])
        pros_s = np.concatenate([pros_s, [pros_s[-1]]])

    aupro = np.trapz(pros_s, fprs_s) / fpr_limit
    return float(aupro)


def _image_score(anomaly_map_final, method):
    if method == "max":
        img_score = np.max(anomaly_map_final)
    elif method == "p99":
        img_score = np.percentile(anomaly_map_final, 99)
    elif method == "mtop5":
        img_score = np.mean(np.sort(anomaly_map_final.flatten())[-5:])
    elif method == "mtop1p":
        img_score = topk_mean(anomaly_map_final, frac=0.01)
    else:
        img_score = np.mean(anomaly_map_final)
    return img_score


def main():
    args = get_args()
    if args.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    diffuse_references = args.use_graph_diffusion and args.k_shot != 1
    few_shot_records = None
    if args.k_shot is not None and not args.batched_zero_shot:
        dataset_prefix = {"mvtec_ad": "mvtec", "visa": "visa"}.get(args.dataset_name, args.dataset_name)
        record_path = (
            Path(__file__).resolve().parent / "few_shot_records"
            / f"{dataset_prefix}_{args.k_shot}shot_normal_samples.json"
        )
        with record_path.open(encoding="utf-8-sig") as record_file:
            few_shot_records = json.load(record_file)
    run_name = f"{args.dataset_name}_{args.agg_method}_layers{''.join(args.layers.split(','))}_res{args.image_res}_docrop{int(args.docrop)}"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        print("No seed specified; aborting for reproducibility.")
        return
    if args.patch_size:
        run_name += f"_patch{args.patch_size}"
    if args.use_kernel_pca:
        run_name += f"_kpca-{args.kernel_pca_kernel}"
    if args.use_graph_diffusion:
        graph_tag = f"gdiff-k{args.diffusion_k}-a{args.diffusion_alpha}"
        if not diffuse_references:
            graph_tag += "-testonly"
        run_name += f"_{graph_tag}"
    if args.spatial_pca_weight > 0:
        run_name += f"_spca-w{args.spatial_pca_weight}-ev{args.spatial_pca_ev}"
        if args.spatial_pca_neighborhood == "random":
            run_name += "-random"
        if args.use_graph_diffusion:
            run_name += "-postdiff"
    if args.use_specular_filter:
        run_name += "_spec-filt"
    if args.bg_mask_method == "pca_normality":
        run_name += "_mask-pca_normality-last"
    elif args.bg_mask_method:
        run_name += f"_mask-{args.bg_mask_method}_thr-{args.mask_threshold_method}"
        if args.mask_threshold_method == "percentile":
            run_name += f"{args.percentile_threshold}"
        if args.bg_mask_method == "dino_saliency":
            run_name += f"_L{args.dino_saliency_layer}"
    run_name += f"_score-{args.score_method}"
    run_name += f"_clahe{int(args.use_clahe)}"
    run_name += f"_dropk{args.drop_k}"
    run_name += f"_model-{args.model_ckpt.split('/')[-1]}"
    run_name += (
        f"pca_ev{args.pca_ev}" if args.pca_ev is not None else f"_pca_dim{args.pca_dim}"
    )
    run_name += f"_i-score{args.img_score_agg}"

    # Add k-shot and augmentation info to run name
    if args.k_shot is not None:
        run_name += f"_k{args.k_shot}"
        if args.aug_count > 0 and args.aug_list:
            # Create a short string for augs, e.g., "hrc"
            aug_str = "".join(sorted([a[0] for a in args.aug_list]))
            run_name += f"_aug{args.aug_count}x{aug_str}"

    run_name += f"_seed{args.seed}"

    if ((args.use_graph_diffusion or args.spatial_pca_weight > 0) and os.name == "nt" and
            len(os.path.abspath(os.path.join(args.outdir, run_name))) > 180):
        digest = hashlib.sha256(run_name.encode()).hexdigest()[:12]
        method_tag = (graph_tag if args.use_graph_diffusion
                      else f"spca-w{args.spatial_pca_weight}-ev{args.spatial_pca_ev}" )
        run_name = (f"{args.dataset_name}_{method_tag}"
                    f"_res{args.image_res}_k{args.k_shot}_seed{args.seed}_{digest}")

    args.outdir = os.path.join(args.outdir, run_name)
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(args.outdir, not args.no_log_file)
    save_config(args)
    if args.save_anomaly_maps:
        dataset_root = Path(os.path.abspath(args.dataset_path))
        anomaly_map_dir = Path(args.outdir) / "anomaly_maps"

    # Augmentations
    aug_transform = None

    # Parse layer args
    layers = parse_layer_indices(args.layers)
    grouped_layers = (
        parse_grouped_layers(args.grouped_layers) if args.agg_method == "group" else []
    )

    # Init model
    model_start = synchronized_time(DEVICE)
    extractor = FeatureExtractor(
        args.model_ckpt, return_tensors=torch.cuda.is_available() and not args.use_kernel_pca,
    )

    model_load_time = synchronized_time(DEVICE) - model_start

    # Get dataset categories
    if args.categories:
        categories = args.categories
    else:
        categories = sorted(
            [
                f.name
                for f in Path(args.dataset_path).iterdir()
                if f.is_dir() and f.name != "split_csv"
            ]
        )

    # Main loop
    all_results = []
    performance_rows = []
    for category in categories:
        logging.info(f"--- Processing Category: {category} ---")
        use_pca_mask = args.bg_mask_method == "pca_normality" and (
            args.dataset_name == "visa"
            or (args.dataset_name == "mvtec_ad" and category in MVTEC_PCA_MASK_CATEGORIES)
        )
        if args.bg_mask_method == "pca_normality":
            logging.info("PCA foreground masking for %s: %s", category, "on" if use_pca_mask else "off")

        if args.k_shot is not None and args.aug_count > 0 and args.aug_list:
            aug_list = args.aug_list
            if not (
                args.dataset_name == "mvtec_ad"
                # and category in {"hazelnut", "metal_nut", "screw", "wood", "zipper"}
                and category in {"screw"}
            ):
                aug_list = [aug for aug in aug_list if aug != "rotate"]
            aug_transform = get_augmentation_transform(aug_list, args.image_res)
            if not aug_transform.transforms:
                aug_transform = None

        else:
            aug_transform = None
        if category in args.no_aug_categories:
            logging.warning(f"Disabling augmentation for {category} category")
            aug_transform = None
        # Rotation counts include the original image at 0 degrees.
        num_aug_multiplier = 1
        if aug_transform:
            num_aug_multiplier = (
                args.aug_count if "rotate" in aug_list else 1 + args.aug_count
            )
        handler = get_dataset_handler(args.dataset_name, args.dataset_path, category)
        train_paths = handler.get_train_paths()
        val_paths = handler.get_validation_paths()
        test_paths = handler.get_test_paths()

        if args.debug_limit is not None:
            logging.warning(
                f"--- DEBUG MODE: Limiting validation and test sets to {args.debug_limit} images ---"
            )
            if val_paths:
                val_paths = val_paths[: args.debug_limit]
            if test_paths:
                test_paths = test_paths[: args.debug_limit]

        if not train_paths and few_shot_records is None:
            logging.warning(f"No training images found for {category}. Skipping.")
            continue

        if args.batched_zero_shot:
            # Batched 0-shot train=test
            logging.info(
                f"--- Batched 0-Shot Mode: Fitting PCA on {len(test_paths)} test images ---"
            )
            train_paths = test_paths.copy()
            val_paths = None

        if few_shot_records is not None:
            recorded_paths = few_shot_records.get(category, [])
            names = [PureWindowsPath(path).name for path in recorded_paths]
            if len(names) != args.k_shot or len(set(names)) != args.k_shot:
                raise ValueError(f"{record_path}: expected {args.k_shot} distinct samples for {category}")
            train_by_name = {Path(path).name: path for path in train_paths}
            missing = [name for name in names if name not in train_by_name]
            if missing:
                raise FileNotFoundError(f"{record_path}: samples missing from {category} training set: {missing}")
            train_paths = [train_by_name[name] for name in names]
            for i, path in enumerate(train_paths, 1):
                logging.info(f"  K-Shot image {i}/{args.k_shot}: {Path(path).name}")

        reference_start = synchronized_time(DEVICE)
        spatial_model = (SpatialDifferencePCA(ev=args.spatial_pca_ev, device=DEVICE,
                                              neighborhood=args.spatial_pca_neighborhood, seed=args.seed)
                         if args.spatial_pca_weight > 0 else None)
        collect_spatial = spatial_model is not None
        spatial_batches = []

        def pack_reference_batch(features, source_ids, with_source_ids, spatial_tokens, selection=None):
            if collect_spatial:
                if diffuse_references:
                    maps = (spatial_tokens.detach().cpu().numpy() if torch.is_tensor(spatial_tokens)
                            else spatial_tokens)
                    spatial_batches.append((maps, selection, source_ids))
                else:
                    spatial_model.add(spatial_tokens, selection)
            if args.use_graph_diffusion and torch.is_tensor(features):
                features = features.detach().cpu().numpy()
            if not with_source_ids:
                return features
            if np.isscalar(source_ids):
                source_ids = np.full(len(features), source_ids, dtype=np.int64)
            elif selection is not None:
                source_ids = source_ids[selection]
            return features, source_ids

        def select_reference_tokens(features, selection):
            if torch.is_tensor(features):
                selection = torch.as_tensor(selection, device=features.device)
            return features[selection]

        # 1. Fit PCA Model
        if args.patch_size:
            if args.bg_mask_method == "pca_normality":
                logging.error(
                    "PCA Normality mask is not compatible with --patch_size. "
                    "Use 'dino_saliency' or no mask."
                )
                raise ValueError("Cannot use pca_normality mask with patch_size.")

            temp_img = Image.open(train_paths[0]).convert("RGB")
            temp_patch = temp_img.crop((0, 0, args.patch_size, args.patch_size))
            temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
                [temp_patch],
                args.image_res,
                layers,
                args.agg_method,
                grouped_layers,
                args.docrop,
                use_clahe=args.use_clahe,
                dino_saliency_layer=args.dino_saliency_layer,
            )
            feature_dim = temp_tokens.shape[-1]
            tokens_per_patch = h_p * w_p

            # Calculate total number of patches and tokens (with augmentations)
            total_patches = 0
            num_batches = 0

            for path in train_paths:
                img = Image.open(path).convert("RGB")
                patch_coords = get_patch_coords(
                    img.height, img.width, args.patch_size, args.patch_overlap
                )
                total_patches += len(patch_coords) * num_aug_multiplier
                num_batches += (
                    math.ceil(len(patch_coords) / args.batch_size) * num_aug_multiplier
                )
            total_tokens = total_patches * tokens_per_patch

            logging.info(
                f"Feature dim: {feature_dim}, Tokens per patch: {tokens_per_patch}, "
                f"Base train patches: {total_patches // num_aug_multiplier}, "
                f"Total train patches (w/ aug): {total_patches}, Total train tokens: {total_tokens}"
            )

            def feature_generator_patched(with_source_ids=False):
                for source_id, path in enumerate(train_paths):
                    pil_img = Image.open(path).convert("RGB")

                    # Create a list of images to process: original + augmentations
                    images_to_process = [pil_img]
                    if aug_transform:
                        for aug_index in range(1, num_aug_multiplier):
                            images_to_process.append(aug_transform(
                                pil_img, rotation_angle=360 * aug_index / args.aug_count
                            ))

                    # Process each image (original + augmented)
                    for img in images_to_process:
                        patch_coords = get_patch_coords(
                            img.height,
                            img.width,
                            args.patch_size,
                            args.patch_overlap,
                        )
                        for i in range(0, len(patch_coords), args.batch_size):
                            coord_batch = patch_coords[i : i + args.batch_size]
                            patch_batch = [img.crop(c) for c in coord_batch]
                            (
                                tokens_batch,
                                _,
                                saliency_masks_batch,
                            ) = extractor.extract_tokens(
                                patch_batch,
                                args.image_res,
                                layers,
                                args.agg_method,
                                grouped_layers,
                                args.docrop,
                                use_clahe=args.use_clahe,
                                dino_saliency_layer=args.dino_saliency_layer,
                            )
                            tokens_flat = tokens_batch.reshape(-1, feature_dim)

                            if args.bg_mask_method == "dino_saliency":
                                masks_flat = saliency_masks_batch.reshape(-1)
                                try:
                                    if args.mask_threshold_method == "percentile":
                                        threshold = np.percentile(
                                            masks_flat, args.percentile_threshold * 100
                                        )
                                        foreground_selection = masks_flat >= threshold
                                    else:
                                        norm_mask = cv2.normalize(
                                            masks_flat,
                                            None,
                                            0,
                                            255,
                                            cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U,
                                        )
                                        _, binary_mask = cv2.threshold(
                                            norm_mask,
                                            0,
                                            255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                        )
                                        foreground_selection = binary_mask.flatten() > 0

                                    foreground_tokens = select_reference_tokens(tokens_flat, foreground_selection)

                                    if foreground_tokens.shape[0] > 0:
                                        yield pack_reference_batch(foreground_tokens, source_id, with_source_ids,
                                                                   tokens_batch, foreground_selection)
                                    else:
                                        logging.warning(
                                            "No foreground patch tokens found. Yielding all tokens."
                                        )
                                        yield pack_reference_batch(tokens_flat, source_id, with_source_ids, tokens_batch)
                                except Exception as e:
                                    logging.warning(
                                        f"Masking failed: {e}. Yielding all tokens."
                                    )
                                    yield pack_reference_batch(tokens_flat, source_id, with_source_ids, tokens_batch)
                            else:
                                yield pack_reference_batch(tokens_flat, source_id, with_source_ids, tokens_batch)

            feature_generator = feature_generator_patched

        else:
            # PCA without patching
            temp_img = Image.open(train_paths[0]).convert("RGB")
            temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
                [temp_img],
                args.image_res,
                layers,
                args.agg_method,
                grouped_layers,
                args.docrop,
                use_clahe=args.use_clahe,
                dino_saliency_layer=args.dino_saliency_layer,
            )
            feature_dim = temp_tokens.shape[-1]
            total_train_images = len(train_paths) * num_aug_multiplier
            total_tokens = total_train_images * h_p * w_p

            logging.info(
                f"Feature dim: {feature_dim}, Tokens per image: {h_p * w_p}, "
                f"Base train images: {len(train_paths)}, "
                f"Total train images (w/ aug): {total_train_images}, Total train tokens: {total_tokens}"
            )

            def feature_generator_full(with_source_ids=False):
                all_imgs_to_process = []
                image_source_ids = []
                for source_id, path in enumerate(train_paths):
                    pil_img = Image.open(path).convert("RGB")
                    all_imgs_to_process.append(pil_img)
                    image_source_ids.append(source_id)
                    if aug_transform:
                        for aug_index in range(1, num_aug_multiplier):
                            all_imgs_to_process.append(aug_transform(
                                pil_img, rotation_angle=360 * aug_index / args.aug_count
                            ))
                            image_source_ids.append(source_id)

                # Now process all_imgs_to_process in batches
                for i in range(0, len(all_imgs_to_process), args.batch_size):
                    img_batch = all_imgs_to_process[i : i + args.batch_size]
                    (
                        tokens_batch,
                        _,
                        saliency_masks_batch,
                    ) = extractor.extract_tokens(
                        img_batch,
                        args.image_res,
                        layers,
                        args.agg_method,
                        grouped_layers,
                        args.docrop,
                        use_clahe=args.use_clahe,
                        dino_saliency_layer=args.dino_saliency_layer,
                    )
                    tokens_flat = tokens_batch.reshape(-1, feature_dim)
                    source_ids = (np.repeat(image_source_ids[i : i + len(img_batch)], h_p * w_p)
                                  if with_source_ids else None)

                    # Train masking logic
                    if args.bg_mask_method == "dino_saliency":
                        masks_flat = saliency_masks_batch.reshape(-1)
                        try:
                            if args.mask_threshold_method == "percentile":
                                threshold = np.percentile(
                                    masks_flat, args.percentile_threshold * 100
                                )
                                foreground_selection = masks_flat >= threshold
                            else:
                                norm_mask = cv2.normalize(
                                    masks_flat,
                                    None,
                                    0,
                                    255,
                                    cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U,
                                )
                                _, binary_mask = cv2.threshold(
                                    norm_mask,
                                    0,
                                    255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                )
                                foreground_selection = binary_mask.flatten() > 0

                            foreground_tokens = select_reference_tokens(tokens_flat, foreground_selection)

                            if foreground_tokens.shape[0] > 0:
                                yield pack_reference_batch(foreground_tokens, source_ids, with_source_ids,
                                                           tokens_batch, foreground_selection)
                            else:
                                logging.warning(
                                    "No foreground tokens found. Yielding all tokens."
                                )
                                yield pack_reference_batch(tokens_flat, source_ids, with_source_ids, tokens_batch)
                        except Exception as e:
                            logging.warning(
                                f"Masking failed: {e}. Yielding all tokens."
                            )
                            yield pack_reference_batch(tokens_flat, source_ids, with_source_ids, tokens_batch)
                    else:
                        yield pack_reference_batch(tokens_flat, source_ids, with_source_ids, tokens_batch)

            num_batches = math.ceil(total_train_images / args.batch_size)
            feature_generator = feature_generator_full

        if spatial_model is not None:
            base_feature_generator = feature_generator

            def feature_generator_with_spatial(with_source_ids=False):
                nonlocal collect_spatial
                yield from base_feature_generator(with_source_ids=with_source_ids)
                collect_spatial = False  # Collect only the first pass, including when fitting PCA twice.

            feature_generator = feature_generator_with_spatial

        graph_adapter = None
        if args.use_graph_diffusion:
            batches = list(tqdm(feature_generator(with_source_ids=True),
                                desc="Normal Reference Collection", total=num_batches))
            all_train_tokens = np.concatenate([features for features, _ in batches])
            source_ids = np.concatenate([ids for _, ids in batches])
            del batches
            graph_adapter = NormalGraphAdapter(k=args.diffusion_k, alpha=args.diffusion_alpha,
                                              seed=args.seed, device=DEVICE)
            all_train_tokens = graph_adapter.fit_transform(
                all_train_tokens, source_ids, adapt_references=diffuse_references,
            )
            if spatial_model is not None and diffuse_references:
                cross_image = len(np.unique(source_ids)) > 1
                offset = 0
                for maps, selection, ids in spatial_batches:
                    count = maps.shape[0] * maps.shape[1] * maps.shape[2] if selection is None else int(np.sum(selection))
                    spatial_model.add_diffused(
                        maps, all_train_tokens[offset:offset + count], graph_adapter,
                        selection, ids if cross_image else None,
                    )
                    offset += count
                spatial_batches.clear()
                logging.info("Spatial difference PCA uses diffused reference features.")
            del source_ids
            total_tokens = len(all_train_tokens)
            fit_batch_size = max(1, args.batch_size * h_p * w_p)
            num_batches = math.ceil(total_tokens / fit_batch_size)

            def feature_generator_adapted():
                for start in range(0, total_tokens, fit_batch_size):
                    yield all_train_tokens[start : start + fit_batch_size]

            feature_generator = feature_generator_adapted

        if args.score_method == "nearest_neighbor":
            reference_batches = [
                torch.as_tensor(batch, device=DEVICE).detach()
                for batch in tqdm(feature_generator(), desc="Nearest Neighbor Reference Collection", total=num_batches)
            ]
            pca_params = {"reference_features": torch.cat(reference_batches)}
            del reference_batches
            logging.info("Nearest-neighbor scoring: references=%d, feature_dim=%d, metric=(1-cosine)/2 in [0,1]",
                         *pca_params["reference_features"].shape)
        elif args.use_kernel_pca:
            if args.bg_mask_method == "pca_normality":
                logging.error(
                    "PCA Normality mask is not compatible with Kernel PCA. "
                    "Use 'dino_saliency' or no mask."
                )
                raise ValueError("Cannot use pca_normality mask with use_kernel_pca.")

            if graph_adapter is None:
                logging.info("Collecting all features for Kernel PCA...")
                all_train_tokens = np.concatenate(
                    list(
                        tqdm(
                            feature_generator(),
                            desc="Feature Collection",
                            total=num_batches,
                        )
                    )
                )
            pca_model = KernelPCAModel(
                k=args.pca_dim,
                kernel=args.kernel_pca_kernel,
                gamma=args.kernel_pca_gamma,
            )
            pca_params = pca_model.fit(all_train_tokens)
        else:
            pca_model = PCAModel(
                k=args.pca_dim, ev=args.pca_ev, whiten=args.whiten,
            )
            pca_params = pca_model.fit(
                feature_generator,
                feature_dim,
                total_tokens,
                num_batches,
            )
        if graph_adapter is not None:
            pca_params["feature_adapter"] = graph_adapter
        if spatial_model is not None:
            pca_params["spatial_pca"] = spatial_model.fit()
            pca_params["spatial_pca_weight"] = args.spatial_pca_weight

        reference_init_time = synchronized_time(DEVICE) - reference_start

        # 2. Determine PR-optimal F1 thresholds (if validation set exists)
        if val_paths:
            logging.info(
                f"Collecting validation stats on {len(val_paths)} images for PR-optimal F1 thresholds..."
            )
            val_img_scores, val_img_labels = [], []
            val_px_scores_normalized, val_px_gts = [], []
            val_iter = tqdm(val_paths, desc="Validating")
            for i in range(0, len(val_paths), args.batch_size):
                path_batch = val_paths[i : i + args.batch_size]
                pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
                is_anomaly_batch = [
                    "good" not in str(p) and "Normal" not in str(p) for p in path_batch
                ]

                if args.patch_size:
                    anomaly_maps_batch, _ = process_image_patched(
                        pil_imgs,
                        extractor,
                        pca_params,
                        args,
                        h_p,
                        w_p,
                        feature_dim,
                    )
                    for j, anomaly_map_final in enumerate(anomaly_maps_batch):
                        if args.img_score_agg == "max":
                            img_score = float(np.max(anomaly_map_final))
                        elif args.img_score_agg == "p99":
                            img_score = float(np.percentile(anomaly_map_final, 99))
                        elif args.img_score_agg == "mtop5":
                            img_score = float(
                                np.mean(np.sort(anomaly_map_final.flatten())[-5:])
                            )
                        elif args.img_score_agg == "mtop1p":
                            img_score = topk_mean(anomaly_map_final, frac=0.01)
                        else:
                            img_score = float(np.mean(anomaly_map_final))
                        val_img_scores.append(img_score)
                        val_img_labels.append(1 if is_anomaly_batch[j] else 0)

                        # --- PIXEL METRICS (AUPRO, P-F1) ---
                        anomaly_map_normalized = min_max_norm(anomaly_map_final)
                        H, W = anomaly_map_normalized.shape
                        gt_mask = handler.get_ground_truth_mask(
                            path_batch[j], pil_imgs[j].size
                        )
                        gt_mask = (
                            np.array(
                                Image.fromarray(
                                    (gt_mask.astype(np.uint8) * 255)
                                ).resize((W, H), resample=Image.NEAREST)
                            )
                            > 127
                        )
                        val_px_gts.append(np.asarray(gt_mask, dtype=np.uint8).ravel())
                        val_px_scores_normalized.append(
                            np.asarray(anomaly_map_normalized, dtype=np.float32).ravel()
                        )

                else:
                    (
                        tokens,
                        (h_p, w_p),
                        saliency_masks_batch,
                    ) = extractor.extract_tokens(
                        pil_imgs,
                        args.image_res,
                        layers,
                        args.agg_method,
                        grouped_layers,
                        args.docrop,
                        use_clahe=args.use_clahe,
                        dino_saliency_layer=args.dino_saliency_layer,
                        use_pca_mask=use_pca_mask,
                    )
                    b = tokens.shape[0]

                    scores = calculate_anomaly_scores(
                        tokens,
                        pca_params,
                        args.score_method,
                        args.drop_k,
                    )
                    anomaly_maps = scores.reshape(b, h_p, w_p)

                    # Apply masking to validation
                    if args.bg_mask_method == "dino_saliency":
                        background_mask = np.zeros_like(anomaly_maps, dtype=bool)
                        for j in range(b):
                            saliency_map = saliency_masks_batch[j]
                            try:
                                if args.mask_threshold_method == "percentile":
                                    threshold = np.percentile(
                                        saliency_map, args.percentile_threshold * 100
                                    )
                                    background_mask[j] = saliency_map < threshold
                                else:  # otsu
                                    norm_mask = cv2.normalize(
                                        saliency_map,
                                        None,
                                        0,
                                        255,
                                        cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8U,
                                    )
                                    _, binary_mask = cv2.threshold(
                                        norm_mask,
                                        0,
                                        255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                    )
                                    background_mask[j] = binary_mask == 0
                            except Exception as e:
                                logging.warning(
                                    f"Saliency mask failed for val image {j}: {e}. Skipping mask."
                                )
                        anomaly_maps[background_mask] = 0.0

                    elif use_pca_mask:
                        anomaly_maps[~saliency_masks_batch] = 0.0
                    for j in range(anomaly_maps.shape[0]):
                        anomaly_map_final = post_process_map(
                            anomaly_maps[j], args.image_res
                        )

                        if args.use_specular_filter:
                            img_tensor = (
                                TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                            )
                            _, _, conf = specular_mask_torch(
                                img_tensor, tau=args.specular_tau
                            )
                            conf = torch.nn.functional.interpolate(
                                conf,
                                size=anomaly_map_final.shape,
                                mode="bilinear",
                                align_corners=False,
                            )
                            conf_map = conf.squeeze().cpu().numpy()
                            anomaly_map_final = (
                                filter_specular_anomalies(anomaly_map_final, conf_map)
                                .cpu()
                                .numpy()
                            )
                        if args.img_score_agg == "max":
                            img_score = float(np.max(anomaly_map_final))
                        elif args.img_score_agg == "p99":
                            img_score = float(np.percentile(anomaly_map_final, 99))
                        elif args.img_score_agg == "mtop5":
                            img_score = float(
                                np.mean(np.sort(anomaly_map_final.flatten())[-5:])
                            )
                        elif args.img_score_agg == "mtop1p":
                            img_score = topk_mean(anomaly_map_final, frac=0.01)
                        else:
                            img_score = float(np.mean(anomaly_map_final))
                        val_img_scores.append(img_score)
                        val_img_labels.append(1 if is_anomaly_batch[j] else 0)
                        anomaly_map_normalized = min_max_norm(anomaly_map_final)
                        H, W = anomaly_map_normalized.shape
                        gt_path_str = handler.get_ground_truth_path(path_batch[j])

                        if not gt_path_str or not os.path.exists(gt_path_str):
                            gt_mask = np.zeros((H, W), dtype=np.uint8)
                        else:
                            gt_mask_pil = Image.open(gt_path_str).convert("L")

                            if args.docrop:
                                resize_res = int(args.image_res / 0.875)
                                gt_mask_pil = TF.resize(
                                    gt_mask_pil,
                                    (resize_res, resize_res),
                                    interpolation=TF.InterpolationMode.NEAREST,
                                )
                                gt_mask_pil = TF.center_crop(
                                    gt_mask_pil, (args.image_res, args.image_res)
                                )

                            gt_mask_pil = TF.resize(
                                gt_mask_pil,
                                (H, W),
                                interpolation=TF.InterpolationMode.NEAREST,
                            )
                            gt_mask = (np.array(gt_mask_pil) > 0).astype(np.uint8)

                        val_px_gts.append(np.asarray(gt_mask, dtype=np.uint8).ravel())
                        val_px_scores_normalized.append(
                            np.asarray(anomaly_map_normalized, dtype=np.float32).ravel()
                        )
                val_iter.update(len(path_batch))

            target_img_fpr = getattr(args, "target_img_fpr", 0.05)
            target_px_fpr = getattr(args, "target_px_fpr", 0.05)

            # Threshold for I-F1 (using raw image scores)
            thr_img, how_img = _pick_threshold_with_fallback(
                val_img_labels, val_img_scores, target_img_fpr
            )
            # Threshold for P-F1 (using per-image normalized pixel scores)
            val_px_scores_mm = np.concatenate(val_px_scores_normalized)
            val_px_labels = np.concatenate(val_px_gts)
            val_px_scores_normalized.clear()
            val_px_gts.clear()
            thr_px, how_px = _pick_threshold_with_fallback(
                val_px_labels, val_px_scores_mm, target_px_fpr
            )
            del val_px_labels, val_px_scores_mm

            if how_img == "none":
                logging.warning(
                    "Validation image threshold degenerate and no negatives: image F1 will be NaN."
                )
            if how_px == "none":
                logging.warning(
                    "Validation pixel threshold degenerate and no negatives: pixel F1 will be NaN."
                )

            logging.info(
                f"Chosen thresholds — Image: {thr_img if thr_img is not None else float('nan'):.6g} "
                f"({how_img}), Pixel: {thr_px if thr_px is not None else float('nan'):.6g} ({how_px})"
            )

        else:
            logging.warning("No validation set found. F1 scores will be N/A.")
            thr_img, thr_px = None, None

        # 3. Evaluate on Test Set
        if graph_adapter is not None:
            graph_adapter.reset_query_stats()
        logging.info(f"Evaluating on {len(test_paths)} test images...")
        img_true, img_pred_f1 = [], []
        img_pred_auroc = []
        test_gt_masks = []
        test_anomaly_maps = []
        px_true_positives = 0
        px_predicted_positives = 0
        vis_saved_count = 0
        all_inference_times = []

        logging.info("Number of test images: {}".format(len(test_paths)))
        test_iter = tqdm(test_paths, desc=f"Testing {category}")
        # Replay the first complete batch for warmup; discard its outputs and time.
        warmup_starts = [0] * args.warmup_iters if test_paths else []
        batch_starts = warmup_starts + list(range(0, len(test_paths), args.batch_size))
        for batch_index, i in enumerate(batch_starts):
            path_batch = test_paths[i : i + args.batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
            is_anomaly_batch = [
                "good" not in str(p) and "Normal" not in str(p) for p in path_batch
            ]
            start_time = synchronized_time(DEVICE)

            final_anomaly_maps_for_batch = []
            saliency_maps_for_viz_batch = []

            if args.patch_size:
                (
                    anomaly_maps_batch,
                    saliency_maps_batch,
                ) = process_image_patched(
                    pil_imgs, extractor, pca_params, args, h_p, w_p, feature_dim
                )

                saliency_maps_for_viz_batch = saliency_maps_batch

                for j, anomaly_map_pre_specular in enumerate(anomaly_maps_batch):
                    anomaly_map_final = anomaly_map_pre_specular
                    if args.use_specular_filter:
                        img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                        _, _, conf = specular_mask_torch(
                            img_tensor, tau=args.specular_tau
                        )
                        conf = torch.nn.functional.interpolate(
                            conf,
                            size=anomaly_map_pre_specular.shape,
                            mode="bilinear",
                            align_corners=False,
                        )
                        conf_map = conf.squeeze().cpu().numpy()
                        anomaly_map_final = (
                            filter_specular_anomalies(
                                anomaly_map_pre_specular, conf_map
                            )
                            .cpu()
                            .numpy()
                        )
                    final_anomaly_maps_for_batch.append(anomaly_map_final)

            else:
                # Step 1: Feature Extraction
                (
                    tokens,
                    (h_p, w_p),
                    saliency_masks_batch,
                ) = extractor.extract_tokens(
                    pil_imgs,
                    args.image_res,
                    layers,
                    args.agg_method,
                    grouped_layers,
                    args.docrop,
                    use_clahe=args.use_clahe,
                    dino_saliency_layer=args.dino_saliency_layer,
                    use_pca_mask=use_pca_mask,
                )
                b = tokens.shape[0]

                # Step 2: Anomaly Scoring
                scores = calculate_anomaly_scores(
                    tokens,
                    pca_params,
                    args.score_method,
                    args.drop_k,
                )
                anomaly_maps = scores.reshape(b, h_p, w_p)

                # Step 3: Masking Strategy
                mask_for_viz = None
                background_mask = np.zeros_like(anomaly_maps, dtype=bool)

                if args.bg_mask_method == "dino_saliency":
                    mask_for_viz = saliency_masks_batch
                    for j in range(b):
                        saliency_map = saliency_masks_batch[j]
                        try:
                            if args.mask_threshold_method == "percentile":
                                threshold = np.percentile(
                                    saliency_map, args.percentile_threshold * 100
                                )
                                background_mask[j] = saliency_map < threshold
                            else:  # otsu
                                norm_mask = cv2.normalize(
                                    saliency_map,
                                    None,
                                    0,
                                    255,
                                    cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U,
                                )
                                _, binary_mask = cv2.threshold(
                                    norm_mask,
                                    0,
                                    255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                )
                                background_mask[j] = binary_mask == 0
                        except Exception as e:
                            logging.warning(
                                f"Saliency mask failed for test image {j}: {e}. Skipping mask."
                            )

                elif use_pca_mask:
                    background_mask = ~saliency_masks_batch
                    mask_for_viz = saliency_masks_batch.astype(np.float32)

                anomaly_maps[background_mask] = 0.0

                saliency_maps_for_viz_batch = mask_for_viz

                for j in range(anomaly_maps.shape[0]):
                    pil_img = pil_imgs[j]
                    anomaly_map_pre_specular = post_process_map(
                        anomaly_maps[j], args.image_res
                    )
                    anomaly_map_final = anomaly_map_pre_specular
                    if args.use_specular_filter:
                        img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                        _, _, conf = specular_mask_torch(
                            img_tensor, tau=args.specular_tau
                        )
                        conf = torch.nn.functional.interpolate(
                            conf,
                            size=anomaly_map_final.shape,
                            mode="bilinear",
                            align_corners=False,
                        )
                        conf_map = conf.squeeze().cpu().numpy()
                        anomaly_map_final = (
                            filter_specular_anomalies(anomaly_map_final, conf_map)
                            .cpu()
                            .numpy()
                        )
                    final_anomaly_maps_for_batch.append(anomaly_map_final)

            image_scores_for_batch = [
                _image_score(anomaly_map, args.img_score_agg)
                for anomaly_map in final_anomaly_maps_for_batch
            ]
            end_time = synchronized_time(DEVICE)
            if batch_index < args.warmup_iters:
                if graph_adapter is not None:
                    graph_adapter.reset_query_stats()
                continue
            all_inference_times.append(end_time - start_time)

            for j, anomaly_map_final in enumerate(final_anomaly_maps_for_batch):
                is_anomaly = is_anomaly_batch[j]
                path = path_batch[j]
                pil_img = pil_imgs[j]
                if args.save_anomaly_maps:
                    relative_path = Path(os.path.abspath(path)).relative_to(dataset_root)
                    map_path = anomaly_map_dir / relative_path.with_suffix(".tiff")
                    map_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.asarray(anomaly_map_final, dtype=np.float32)).save(map_path)
                img_score = image_scores_for_batch[j]

                img_true.append(1 if is_anomaly else 0)
                img_pred_auroc.append(float(img_score))
                if thr_img is not None:
                    img_pred_f1.append(1 if img_score >= thr_img else 0)

                anomaly_map_normalized = min_max_norm(anomaly_map_final)
                H, W = anomaly_map_normalized.shape
                gt_path_str = handler.get_ground_truth_path(path)
                if not gt_path_str or not os.path.exists(gt_path_str):
                    gt_mask = np.zeros((H, W), dtype=np.uint8)
                else:
                    gt_mask_pil = Image.open(gt_path_str).convert("L")
                    if args.docrop:
                        resize_res = int(args.image_res / 0.875)
                        gt_mask_pil = TF.resize(
                            gt_mask_pil,
                            (resize_res, resize_res),
                            interpolation=TF.InterpolationMode.NEAREST,
                        )
                        gt_mask_pil = TF.center_crop(
                            gt_mask_pil, (args.image_res, args.image_res)
                        )
                    gt_mask_pil = TF.resize(
                        gt_mask_pil,
                        (H, W),
                        interpolation=TF.InterpolationMode.NEAREST,
                    )
                    gt_mask = (np.array(gt_mask_pil) > 0).astype(np.uint8)

                test_gt_masks.append(gt_mask)
                test_anomaly_maps.append(np.asarray(anomaly_map_final, dtype=np.float32))
                if thr_px is not None:
                    predicted_positive = np.asarray(anomaly_map_normalized, dtype=np.float32) >= thr_px
                    px_predicted_positives += np.count_nonzero(predicted_positive)
                    px_true_positives += np.count_nonzero(predicted_positive & (gt_mask != 0))

                if is_anomaly:
                    if args.save_intro_overlays:
                        vis_img = pil_img
                        save_overlay_for_intro(
                            path,
                            vis_img,
                            anomaly_map_normalized,
                            args.outdir,
                            category,
                        )
                    if vis_saved_count < args.vis_count:
                        vis_img = pil_img
                        if args.docrop and not args.patch_size:
                            resize_res = int(args.image_res / 0.875)
                            vis_img = TF.resize(
                                vis_img,
                                (resize_res, resize_res),
                                interpolation=TF.InterpolationMode.BICUBIC,
                            )
                            vis_img = TF.center_crop(
                                vis_img, (args.image_res, args.image_res)
                            )
                        saliency_map_for_viz = None
                        raw_mask_map = None
                        if saliency_maps_for_viz_batch is not None:
                            raw_mask_map = saliency_maps_for_viz_batch[j]

                        if raw_mask_map is not None:
                            try:
                                if use_pca_mask:
                                    binary_mask = raw_mask_map

                                elif args.bg_mask_method == "dino_saliency":
                                    if args.mask_threshold_method == "percentile":
                                        threshold_val = np.percentile(
                                            raw_mask_map,
                                            args.percentile_threshold * 100,
                                        )
                                        binary_mask = (
                                            raw_mask_map >= threshold_val
                                        ).astype(np.float32)
                                    else:  # otsu
                                        norm_mask = cv2.normalize(
                                            raw_mask_map,
                                            None,
                                            0,
                                            255,
                                            cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U,
                                        )
                                        _, binary_mask_u8 = cv2.threshold(
                                            norm_mask,
                                            0,
                                            255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                        )
                                        binary_mask = (binary_mask_u8 > 0).astype(
                                            np.float32
                                        )
                                saliency_map_for_viz = post_process_map(
                                    binary_mask,
                                    anomaly_map_normalized.shape,
                                    blur=False,
                                )
                            except Exception as e:
                                logging.warning(
                                    f"Saliency mask processing failed for visualization: {e}."
                                )

                        save_visualization(
                            path,
                            vis_img,
                            gt_mask,
                            anomaly_map_normalized,
                            args.outdir,
                            category,
                            vis_saved_count,
                            saliency_mask=saliency_map_for_viz,
                        )
                        vis_saved_count += 1

            test_iter.update(len(path_batch))
        if graph_adapter is not None:
            graph_adapter.log_query_stats()
        performance_rows.append({
            "category": category,
            "reference_init_time_s": reference_init_time,
            "num_images": len(img_pred_auroc),
            "inference_time_s": sum(all_inference_times),
        })
        if all_inference_times:
            times_arr = np.array(all_inference_times)
            total_images_processed = len(test_paths)
            total_time = np.sum(times_arr)

            avg_time_per_image = total_time / total_images_processed
            images_per_second = 1.0 / avg_time_per_image

            logging.info(f"--- Timing Results for {category} ---")
            logging.info(f"Total test images: {total_images_processed}")
            logging.info(
                f"Batch size: {args.batch_size} (Processed {len(all_inference_times)} batches)"
            )
            logging.info(f"Total inference time: {total_time:.4f} s")
            logging.info(f"Avg. time per image: {avg_time_per_image:.6f} s")
            logging.info(f"Images per second (FPS): {images_per_second:.2f}")

            # Report batch stats
            if len(times_arr) > 1:
                times_arr_stats = times_arr[1:]
                logging.info(
                    f"Avg. time per batch (excl. 1st): {np.mean(times_arr_stats):.6f} s"
                )
                logging.info(
                    f"Median time per batch (excl. 1st): {np.median(times_arr_stats):.6f} s"
                )
            else:
                logging.info(f"Avg. time per batch: {np.mean(times_arr):.6f} s")
        img_auroc = (
            roc_auc_score(img_true, img_pred_auroc)
            if len(np.unique(img_true)) > 1
            else np.nan
        )

        img_aupr = (
            average_precision_score(img_true, img_pred_auroc)
            if len(np.unique(img_true)) > 1
            else np.nan
        )

        px_positive_count = sum(np.count_nonzero(mask) for mask in test_gt_masks)
        px_count = sum(mask.size for mask in test_gt_masks)
        px_auroc = np.nan
        if 0 < px_positive_count < px_count:
            px_true_arr = np.concatenate([mask.ravel() for mask in test_gt_masks])
            px_pred_arr = np.concatenate([scores.ravel() for scores in test_anomaly_maps])
            px_auroc = roc_auc_score(px_true_arr, px_pred_arr)
            del px_true_arr, px_pred_arr
        img_f1 = f1_score(img_true, img_pred_f1) if (thr_img is not None) else np.nan
        if thr_px is not None and px_positive_count:
            px_f1 = 2 * px_true_positives / (px_positive_count + px_predicted_positives)
        else:
            px_f1 = np.nan
        if px_positive_count:
            fpr_cap = getattr(args, "pro_integration_limit", 0.3)
            au_pro = compute_aupro(
                test_anomaly_maps,
                test_gt_masks,
                fpr_limit=fpr_cap,
                num_thresholds=300,
                connectivity=8,
            )
        else:
            logging.warning(
                f"No anomalous ground-truth regions found in test set for {category}. "
                "AUPRO is not computable."
            )
            au_pro = np.nan
        test_anomaly_maps.clear()
        test_gt_masks.clear()
        logging.info(
            f"{category} Results | I-AUROC: {img_auroc:.4f} | I-AUPR: {img_aupr:.4f} | "
            f"P-AUROC: {px_auroc:.4f} | AU-PRO: {au_pro:.4f} | "
            f"I-F1: {img_f1:.4f} | P-F1: {px_f1:.4f}"
        )
        all_results.append(
            [category, args.seed, img_auroc, img_aupr, px_auroc, au_pro, img_f1, px_f1]
        )

    df = pd.DataFrame(
        all_results,
        columns=[
            "Category",
            "Seed",
            "Image AUROC",
            "Image AUPR",
            "Pixel AUROC",
            "AU-PRO",
            "Image F1",
            "Pixel F1",
        ],
    )
    if not df.empty and len(df) > 1:
        mean_values = df.mean(numeric_only=True)
        mean_row = pd.DataFrame(
            [["Average"] + mean_values.tolist()], columns=df.columns
        )
        df = pd.concat([df, mean_row], ignore_index=True)

    logging.info("\n--- Benchmark Final Results ---")
    logging.info("\n" + df.to_string(index=False, float_format="%.4f", na_rep="N/A"))

    results_path = os.path.join(args.outdir, "benchmark_results.csv")
    df.to_csv(results_path, index=False, float_format="%.4f")
    logging.info(f"\nResults saved to {results_path}")
    performance_path = os.path.join(args.outdir, "performance_stats.json")
    stats = write_performance_stats(
        performance_path, model_load_time, performance_rows,
        {**vars(args), "device": DEVICE},
    )
    logging.info("Performance stats saved to %s (initialization: %.4f s, FPS: %s)",
                 performance_path, stats["initialization_time_s"], stats["fps"])


if __name__ == "__main__":
    main()
