PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

# Set the absolute path to your MVTec AD dataset directory
#MVTEC_PATH="/root/wsx/mvtec"
MVTEC_PATH="D:\develop\dataset\mvtec"

# Set the absolute path to your VisA dataset directory
# (e.g., /path/to/your/datasets/VisA_pytorch/1cls)
#VISA_PATH="/root/wsx/convert_visa/1cls"
VISA_PATH="D:\develop\dataset\visa"

# Model identifiers from Hugging Face
BACKBONES=(
    "facebook/dinov2-small"
)

# Short names for output directories
NAMES=(
    "dinov2S"
)

# Proportional "equivalent middle layers" based on the
# DINOv2-G selection (layers 22-28, or 55%-70% deep)
LAYERS_STRINGS=(
    "-4,-5"          # S/B (12 layers): 55%-70% deep is ~layers 7-8
)
for i in ${!BACKBONES[@]}; do
    MODEL_CKPT=${BACKBONES[$i]}
    MODEL_NAME=${NAMES[$i]}
    LAYERS=${LAYERS_STRINGS[$i]}

    echo "========================================================"
    echo "=== STARTING EXPERIMENTS FOR BACKBONE: $MODEL_NAME ($MODEL_CKPT) ==="
    echo "=== Using Layers: $LAYERS ==="
    echo "========================================================"

    for k in 1 2 4
    do
        echo "--- Running MVTec AD k=$k for $MODEL_NAME ---"
        python -u main.py \
            --dataset_name mvtec_ad \
            --dataset_path "$MVTEC_PATH" \
            --image_res 448 \
            --k_shot $k \
            --layers="$LAYERS" \
            --model_ckpt "$MODEL_CKPT" \
            --aug_count 12 \
            --pca_ev 0.99 \
            --agg_method "mean" \
            --outdir "results_ablations/" \
            --use_graph_diffusion --diffusion_k 25 --diffusion_alpha 0.7 \
            --bg_mask_method pca_normality
    done

    for k in 1 2 4
    do
        echo "--- Running VisA k=$k for $MODEL_NAME ---"
        python -u main.py \
            --dataset_name visa \
            --dataset_path "$VISA_PATH" \
            --image_res 448 \
            --k_shot $k \
            --layers="$LAYERS" \
            --model_ckpt "$MODEL_CKPT" \
            --pca_ev 0.99 \
            --agg_method "mean" \
            --outdir "results_ablations/" \
            --use_graph_diffusion --diffusion_k 25 --diffusion_alpha 0.5 \
            --spatial_pca_weight 0.7 \
            --bg_mask_method pca_normality
    done

done

#            --save_anomaly_maps \
#            --spatial_pca_neighborhood random
