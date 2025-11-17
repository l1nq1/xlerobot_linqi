#!/bin/bash
# 使用多数据集配置运行训练
# 确保已经安装所有依赖：cd src && pip install -e .

echo "=========================================="
echo "开始多数据集训练"
echo "=========================================="
echo ""
echo "数据集列表:"
echo "  1. pickup_circle_50episodes_single_arm_1107"
echo "  2. pickup_circle_20episodes_single_arm_1107_2"
echo ""
echo "数据集根目录: /home1/linqi/test_dataset"
echo ""
echo "=========================================="
echo ""

python -m lerobot.scripts.train --config multi_dataset_config.yaml

