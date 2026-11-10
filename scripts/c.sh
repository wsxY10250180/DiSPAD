DATASET="mvtec"
OURS="DiSPAD"

python exp/mine_qualitative_cases.py \
  --heatmap-root D:\\develop\\dataset\\heatmap\\2shot \
  --data-root D:\\develop\\dataset\\${DATASET} \
  --ours ${OURS} \
  --methods ${OURS} SubspaceAD AnomalyDINO \
  --norm per_heatmap \
  --selection-strategy pixel --pixel-metrics auroc ap aupro \
  --out-dir D:\\develop\\dataset\\exp_res\\${OURS}\\${DATASET}