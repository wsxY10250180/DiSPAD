import argparse


def parse_layer_indices(arg_str: str):
    """Parses a comma-separated string of integers."""
    return [int(x.strip()) for x in arg_str.split(",")]


def parse_grouped_layers(arg_str: str):
    """Parses grouped layer indices from format like '-1,-2:-3,-4'."""
    if not arg_str:
        return []
    return [parse_layer_indices(group) for group in arg_str.split(":")]


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Anomaly Detection Benchmark Framework"
    )
    data_group = parser.add_argument_group("Dataset Arguments")
    model_group = parser.add_argument_group("Model & Feature Extraction Arguments")
    aug_group = parser.add_argument_group("Augmentation Arguments (for k-shot)")
    pca_group = parser.add_argument_group("Anomaly Detection (PCA) Arguments")
    score_group = parser.add_argument_group("Scoring & Evaluation Arguments")
    mask_group = parser.add_argument_group("Background Removal (Saliency) Arguments")
    specular_group = parser.add_argument_group("Specular Reflection Filter Arguments")
    log_group = parser.add_argument_group("Logistics")

    data_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    data_group.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["mvtec_ad", "mvtec_ad2", "visa"],
        help="Name of the dataset to use.",
    )
    data_group.add_argument(
        "--dataset_path", type=str, required=True, help="Root path to the dataset."
    )
    data_group.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specify categories to run, e.g., 'bottle screw'. If None, runs all.",
    )
    model_group.add_argument(
        "--model_ckpt",
        type=str,
        default="facebook/dinov2-with-registers-large",
        help="HuggingFace model checkpoint for feature extraction.",
    )
    model_group.add_argument(
        "--image_res", type=int, default=256, help="Image resolution for the model."
    )
    model_group.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Size of the square patches. If None, process in full resolution.",
    )
    model_group.add_argument(
        "--patch_overlap",
        type=float,
        default=0.0,
        help="Overlap ratio between patches.",
    )
    model_group.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for feature extraction."
    )
    model_group.add_argument(
        "--warmup_iters", type=int, default=1,
        help="Complete untimed test batches per category before measuring FPS (0 disables).",
    )
    model_group.add_argument(
        "--k_shot",
        type=int,
        default=None,
        help="Number of 'good' training images to use (k-shot). If None, all are used.",
    )
    model_group.add_argument(
        "--agg_method",
        type=str,
        default="mean",
        choices=["concat", "mean", "group"],
        help="Feature aggregation method across layers.",
    )
    model_group.add_argument(
        "--layers",
        type=str,
        default="-12,-13,-14,-15,-16,-17,-18",
        help="Comma-separated layer indices for 'concat' or 'mean' aggregation.",
    )
    model_group.add_argument(
        "--grouped_layers",
        type=str,
        default=None,
        help="Layer groups for 'group' agg. Format: '-1,-2:-3,-4'.",
    )
    model_group.add_argument(
        "--docrop",
        action="store_true",
        help="Apply center cropping during preprocessing.",
    )
    model_group.add_argument(
        "--use_clahe",
        action="store_true",
        help="Apply CLAHE to the images.",
    )

    aug_group.add_argument(
        "--aug_count",
        type=int,
        default=0,
        help="With rotate: total samples per k-shot image including the original, spaced by 360/N degrees. Otherwise: number of extra augmented samples. 0 disables augmentation. Only active if --k_shot is set.",
    )
    aug_group.add_argument(
        "--aug_list",
        type=str,
        nargs="+",
        default=["rotate"],
        help="List of augmentations to apply. Choices: hflip, vflip, rotate, color_jitter, affine.",
    )
    aug_group.add_argument(
        "--no_aug_categories",
        type=str,
        nargs="+",
        default=["transistor"],
        help="List of categories for which augmentations should be disabled.",
    )

    pca_group.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="Number of principal components to keep. Overrides --pca_ev.",
    )
    pca_group.add_argument(
        "--pca_ev",
        type=float,
        default=0.99,
        help="Explained variance to retain for global PCA when its dimension is unset.",
    )
    pca_group.add_argument(
        "--whiten", action="store_true", help="Apply whitening in PCA."
    )
    pca_group.add_argument(
        "--use_kernel_pca",
        action="store_true",
        help="Use Kernel PCA instead of standard PCA.",
    )
    pca_group.add_argument(
        "--use_graph_diffusion", action="store_true",
        help="Apply one confidence-gated normal-reference diffusion step before PCA/scoring.",
    )
    pca_group.add_argument(
        "--diffusion_k", type=int, default=5,
        help="Maximum normal neighbors per patch for graph diffusion.",
    )
    pca_group.add_argument(
        "--diffusion_alpha", type=float, default=0.2,
        help="Maximum diffusion mixing strength in [0, 1].",
    )
    pca_group.add_argument(
        "--spatial_pca_weight", type=float, default=0.0,
        help="Spatial-difference PCA weight w in [0, 1]; score=(1-w)*original+w*spatial. 0 disables it. "
             "Uses diffused features when graph diffusion is enabled.",
    )
    pca_group.add_argument(
        "--spatial_pca_ev", type=float, default=0.99,
        help="Explained variance retained independently by the spatial-difference PCA branch.",
    )
    pca_group.add_argument(
        "--spatial_pca_neighborhood", choices=["spatial", "random"], default="spatial",
        help="Use the true 3x3 neighbors or equally many random patches outside the window.",
    )
    pca_group.add_argument(
        "--kernel_pca_kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly", "sigmoid", "cosine"],
        help="Kernel to use for Kernel PCA.",
    )
    pca_group.add_argument(
        "--kernel_pca_gamma",
        type=float,
        default=None,
        help="Gamma for rbf, poly and sigmoid kernels. If None, it's set to 1/n_features.",
    )
    score_group.add_argument(
        "--score_method",
        type=str,
        default="reconstruction",
        choices=["reconstruction", "mahalanobis", "cosine", "euclidean", "nearest_neighbor"],
        help="Anomaly scoring method. nearest_neighbor uses minimum (1 - cosine similarity) / 2 in [0, 1].",
    )
    score_group.add_argument(
        "--drop_k",
        type=int,
        default=0,
        help="Number of initial principal components to drop during reconstruction scoring.",
    )
    score_group.add_argument(
        "--img_score_agg",
        type=str,
        default="mtop1p",
        choices=["max", "mean", "p99", "mtop5", "mtop1p"],
        help="Aggregation for image-level scores from pixel maps.",
    )
    score_group.add_argument(
        "--pro_integration_limit",
        type=float,
        default=0.3,
        help="Integration limit for AU-PRO calculation.",
    )
    mask_group.add_argument(
        "--bg_mask_method",
        type=str,
        default=None,
        choices=[None, "dino_saliency", "pca_normality"],
        help="Background masking method. pca_normality follows AnomalyDINO's MVTec AD and VisA category settings.",
    )
    mask_group.add_argument(
        "--mask_threshold_method",
        type=str,
        default="percentile",
        choices=["percentile", "otsu"],
        help="How to binarize the DINO attention saliency map.",
    )
    mask_group.add_argument(
        "--percentile_threshold",
        type=float,
        default=0.15,
        help="Percentile threshold (0.0-1.0) for 'percentile' method.",
    )
    mask_group.add_argument(
        "--dino_saliency_layer",
        type=int,
        default=6,
        help="Which transformer layer's attention to use for 'dino_saliency' mask (0-indexed).",
    )
    specular_group.add_argument(
        "--use_specular_filter",
        action="store_true",
        help="Enable the specular reflection filter as a post-processing step.",
    )
    specular_group.add_argument(
        "--specular_tau",
        type=float,
        default=0.6,
        help="Binarization threshold for the specular mask.",
    )
    specular_group.add_argument(
        "--specular_size_threshold_factor",
        type=float,
        default=1.5,
        help="Size threshold factor for filtering specular anomalies.",
    )
    log_group.add_argument(
        "--outdir",
        type=str,
        default="./results_full_shot",
        help="Directory to save results, logs, and visualizations.",
    )
    log_group.add_argument(
        "--vis_count",
        type=int,
        default=0,
        help="Number of anomalous examples to visualize per category.",
    )
    log_group.add_argument(
        "--save_anomaly_maps",
        action="store_true",
        help="Save unnormalized test anomaly score maps as float32 TIFF files.",
    )
    log_group.add_argument(
        "--save_intro_overlays",
        action="store_true",
        help="Save clean overlay images for the introductory figure.",
    )
    log_group.add_argument(
        "--no_log_file",
        action="store_true",
        help="Do not save a log file to the output directory.",
    )
    log_group.add_argument(
        "--debug_limit",
        type=int,
        default=None,
        help="Run in debug mode on a subset of N images.",
    )
    log_group.add_argument(
        "--batched_zero_shot",
        action="store_true",
        help="Run in batched zero-shot mode, fitting PCA on the test set.",
    )

    args = parser.parse_args()
    if not 0 <= args.spatial_pca_weight <= 1:
        parser.error("spatial_pca_weight must be in [0, 1].")
    if not 0 < args.spatial_pca_ev <= 1:
        parser.error("spatial_pca_ev must be in (0, 1].")
    if args.spatial_pca_weight > 0 and args.batched_zero_shot:
        parser.error("Spatial PCA requires normal references; batched zero-shot is unsupported.")
    if args.score_method == "nearest_neighbor" and (
            args.use_kernel_pca or args.batched_zero_shot):
        parser.error("Nearest-neighbor scoring requires normal references; disable kernel PCA and batched zero-shot.")
    if args.use_graph_diffusion:
        if args.diffusion_k < 1 or not 0 <= args.diffusion_alpha <= 1:
            parser.error("Graph diffusion requires k >= 1 and alpha in [0, 1].")
        if args.batched_zero_shot:
            parser.error("Graph diffusion requires known normal references; batched zero-shot is unsupported.")
    return args
