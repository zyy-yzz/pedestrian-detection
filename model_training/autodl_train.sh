#!/bin/bash
# ============================================================
# Enhanced YOLOv8 AutoDL Training Script
# One-click training on AutoDL instances
# ============================================================
# 本地打包:
#   bash model_training/package_for_autodl.sh
# 上传到 AutoDL:
#   将 autodl_upload/ 下两个 zip 上传到 /root/autodl-tmp/
# 训练:
#   bash /root/autodl-tmp/autodl_train.sh           # 从头训练
#   bash /root/autodl-tmp/autodl_train.sh --resume   # 断点续训
# ============================================================
set -e

TMP_DIR="/root/autodl-tmp"
WORK_DIR="/root/pedestrian-detection"
CONFIG="model_training/configs/autodl_train_config.yaml"
RESUME="${1:-}"

echo "============================================"
echo " Enhanced YOLOv8 AutoDL Training Pipeline"
echo "============================================"

# ---- 1. System dependencies ----
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 > /dev/null 2>&1

# ---- 2. Python packages ----
echo "[2/6] Installing Python packages..."
pip install -q ultralytics>=8.2.0 torch>=2.2.0 opencv-python>=4.9.0 pyyaml

# ---- 3. Extract files ----
echo "[3/6] Extracting project files..."

if [ -f "$TMP_DIR/autodl_project.zip" ]; then
    unzip -o "$TMP_DIR/autodl_project.zip" -d "$WORK_DIR/" > /dev/null
    echo "  Project files extracted"
else
    echo "ERROR: $TMP_DIR/autodl_project.zip not found!"
    exit 1
fi

if [ -f "$TMP_DIR/autodl_llvip.zip" ]; then
    echo "  Extracting LLVIP dataset (this may take a minute)..."
    unzip -o "$TMP_DIR/autodl_llvip.zip" -d "$WORK_DIR/" > /dev/null
    echo "  LLVIP dataset extracted"
else
    echo "ERROR: $TMP_DIR/autodl_llvip.zip not found!"
    exit 1
fi

# ---- 4. Verify dataset ----
echo "[4/6] Verifying dataset..."
XML_COUNT=$(find "$WORK_DIR/datasets/LLVIP/Annotations" -name "*.xml" 2>/dev/null | wc -l)
IR_TRAIN=$(find "$WORK_DIR/datasets/LLVIP/infrared/train" -name "*.jpg" 2>/dev/null | wc -l)
echo "  XML annotations: $XML_COUNT"
echo "  IR train images: $IR_TRAIN"
if [ "$XML_COUNT" -eq 0 ]; then
    echo "ERROR: No annotations found! Check LLVIP zip structure."
    exit 1
fi

# ---- 5. GPU check ----
echo "[5/6] GPU info..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: nvidia-smi not available"

# ---- 6. Train ----
echo "[6/6] Starting training..."
cd "$WORK_DIR"

if [ "$RESUME" = "--resume" ]; then
    echo "  Mode: RESUME from last checkpoint"
    python model_training/train.py --config "$CONFIG" --resume
else
    echo "  Mode: training from scratch"
    python model_training/train.py --config "$CONFIG"
fi

# ---- Copy best model ----
echo ""
echo "============================================"
echo " Copying best model to output directory"
echo "============================================"
BEST=$(find runs/night_ped_detection -name "best.pt" 2>/dev/null | sort | tail -1)
if [ -n "$BEST" ]; then
    cp "$BEST" "$TMP_DIR/enhanced_yolov8n.pt"
    echo "  Model saved: $TMP_DIR/enhanced_yolov8n.pt"
else
    echo "  WARNING: best.pt not found"
fi

# ---- Evaluate ----
echo ""
echo "============================================"
echo " Evaluation"
echo "============================================"
python -c "
import sys; sys.path.insert(0, '$WORK_DIR')
from model_training.evaluate import evaluate
m = evaluate('$TMP_DIR/enhanced_yolov8n.pt', '$CONFIG', '0')
print(f\"mAP@0.5:      {m['mAP_0.5']:.4f}  (target >= 0.78)\")
print(f\"mAP@0.5:0.95: {m['mAP_0.5_0.95']:.4f}  (target >= 0.52)\")
print(f\"FPS:          {m['fps']:.1f}\")
"

echo ""
echo "============================================"
echo " Done!"
echo " Download: $TMP_DIR/enhanced_yolov8n.pt"
echo "============================================"
