#!/bin/bash
# 多数据集训练启动脚本
# 自动配置 HuggingFace 环境变量和 token，无需手动登录

# 设置 HuggingFace 镜像端点（使用国内镜像加速）
export HF_ENDPOINT=https://hf-mirror.com

# 设置 HuggingFace Token（自动认证，无需手动登录）
# 请将您的 token 设置为环境变量，或取消下面的注释并填入您的 token
# export HF_TOKEN=your_huggingface_token_here
if [ -z "$HF_TOKEN" ]; then
    echo "警告: HF_TOKEN 未设置，请运行: export HF_TOKEN=your_token"
    echo "或者取消注释 train_multi_dataset.sh 中的 HF_TOKEN 行并填入您的 token"
fi

# 可选：设置 HuggingFace Hub 缓存目录（如果需要）
# export HF_HOME=/data/home/lixinyi/.cache/huggingface

echo "=========================================="
echo "HuggingFace 配置已自动设置："
echo "  HF_ENDPOINT: $HF_ENDPOINT"
if [ -n "$HF_TOKEN" ]; then
    echo "  HF_TOKEN: ${HF_TOKEN:0:10}...（已设置）"
else
    echo "  HF_TOKEN: 未设置"
fi
echo "=========================================="
echo ""

# 切换到项目目录
cd /data/home/lixinyi/linqi/xlerobot_linqi

# 运行训练命令，传递所有参数
python -m lerobot.scripts.train --config multi_dataset_config.yaml "$@"
