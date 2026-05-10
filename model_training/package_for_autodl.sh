#!/bin/bash
# ============================================================
# 本地打包脚本 — 打包项目文件和数据集用于 AutoDL 上传
# ============================================================
# 用法:
#   bash model_training/package_for_autodl.sh
# 输出:
#   autodl_upload/
#   ├── autodl_train.sh        # 训练脚本 (直接上传到 /root/autodl-tmp/)
#   ├── autodl_project.zip     # 项目代码 (model_training/)
#   └── autodl_llvip.zip       # LLVIP 数据集 (datasets/LLVIP/)
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/autodl_upload"

echo "============================================"
echo " Packaging for AutoDL Upload"
echo "============================================"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

# ---- 1. 打包项目代码 ----
echo "[1/3] Packaging project code..."
zip -r "$OUTPUT_DIR/autodl_project.zip" model_training/ \
    -x "model_training/notebooks/*" \
    -x "model_training/*.sh" \
    -x "model_training/__pycache__/*" \
    -x "*.ipynb_checkpoints/*"
echo "  -> autodl_project.zip"

# ---- 2. 拷贝训练脚本 ----
echo "[2/3] Copying training script..."
cp model_training/autodl_train.sh "$OUTPUT_DIR/"
echo "  -> autodl_train.sh"

# ---- 3. 打包数据集 ----
echo "[3/3] Packaging LLVIP dataset..."
if [ -d "datasets/LLVIP" ]; then
    zip -r "$OUTPUT_DIR/autodl_llvip.zip" datasets/LLVIP/ \
        -x "datasets/LLVIP/labels/*" \
        -x "datasets/LLVIP/data.yaml"
    echo "  -> autodl_llvip.zip"
else
    echo "  WARNING: datasets/LLVIP not found, skipping"
fi

echo ""
echo "============================================"
echo " Done! Upload these files to AutoDL:"
echo ""
echo "   $OUTPUT_DIR/autodl_train.sh"
echo "   $OUTPUT_DIR/autodl_project.zip"
echo "   $OUTPUT_DIR/autodl_llvip.zip"
echo ""
echo " Target: /root/autodl-tmp/"
echo ""
echo " Run on AutoDL:"
echo "   bash /root/autodl-tmp/autodl_train.sh           # 从头训练"
echo "   bash /root/autodl-tmp/autodl_train.sh --resume  # 断点续训"
echo "============================================"
