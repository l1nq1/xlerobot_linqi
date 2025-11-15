# HuggingFace 自动配置说明

## 已完成的配置

我已经为您创建了两个方案来自动配置 HuggingFace，无需每次手动登录：

### 方案 1：训练启动脚本（推荐）

使用 `train_multi_dataset.sh` 脚本启动训练：

```bash
cd /data/home/lixinyi/linqi/xlerobot_linqi
./train_multi_dataset.sh
```

或者传递额外参数：

```bash
./train_multi_dataset.sh --resume
```

**优点：**
- 只在训练时设置环境变量
- 不影响其他 conda 环境
- 可以轻松修改配置

### 方案 2：Conda 环境激活脚本（永久配置）

已在 `lerobot` conda 环境中创建了自动配置脚本：
- 位置：`/home/lixinyi/.conda/envs/lerobot/etc/conda/activate.d/hf_config.sh`
- 每次激活 `lerobot` 环境时自动设置环境变量

**使用方法：**
```bash
conda activate lerobot
# 环境变量已自动设置，无需手动配置
python -m lerobot.scripts.train --config multi_dataset_config.yaml
```

**优点：**
- 一次配置，永久生效
- 激活环境后自动设置
- 无需记住额外的启动脚本

## 配置内容

两个方案都设置了以下环境变量：

- `HF_ENDPOINT=https://hf-mirror.com` - 使用国内镜像加速
- `HF_TOKEN` - 自动认证 token（需要在脚本中设置或通过环境变量提供）

**重要：** 请在脚本中设置您的 HuggingFace token，或通过环境变量 `export HF_TOKEN=your_token` 提供。

## 验证配置

运行以下命令验证配置是否生效：

```bash
conda activate lerobot
echo "HF_ENDPOINT: $HF_ENDPOINT"
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN: ${HF_TOKEN:0:10}...（已设置）"
else
    echo "HF_TOKEN: 未设置"
fi
```

## 注意事项

1. **Token 安全**：Token 已硬编码在脚本中，请确保脚本文件权限安全（已设置为仅所有者可读）
2. **Token 更新**：如果 token 过期，需要更新脚本中的 token 值
3. **网络问题**：如果仍然遇到网络问题，可能需要配置代理或使用其他镜像源

## 修改配置

如果需要修改配置，可以编辑：

- **训练脚本**：`/data/home/lixinyi/linqi/xlerobot_linqi/train_multi_dataset.sh`
- **Conda 激活脚本**：`/home/lixinyi/.conda/envs/lerobot/etc/conda/activate.d/hf_config.sh`

修改后重新激活 conda 环境或重新运行脚本即可生效。

