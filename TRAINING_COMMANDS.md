# 多数据集训练命令参考

## 方法 1: 使用 YAML 配置文件（推荐）

这是最简单和最可靠的方法，特别适合本地数据集。

### 步骤 1: 创建配置文件

创建文件 `multi_dataset_config.yaml`：

```yaml
dataset:
  repo_id:
    - pickup_circle_50episodes_single_arm_1107
    - pickup_circle_20episodes_single_arm_1107_2
  root: /home1/linqi/test_dataset
  image_transforms:
    enable: false

policy:
  type: smolvla
  device: cuda
  push_to_hub: false
  chunk_size: 50

job_name: my_smolvla_training_multi_dataset
output_dir: tests/outputs/smolvla_multi_dataset
batch_size: 2
steps: 100000
save_freq: 5000
log_freq: 200
wandb:
  enable: false
```

### 步骤 2: 运行训练

```bash
python -m lerobot.scripts.train --config multi_dataset_config.yaml
```

## 方法 2: 使用 Makefile（修改后的版本）

更新您的 Makefile：

```makefile
test-smolvla-multi-train:
	python -m lerobot.scripts.train \
		--config multi_dataset_config.yaml
```

或者内联所有参数（但需要正确的列表语法）：

```makefile
test-smolvla-multi-train:
	python -m lerobot.scripts.train \
		--policy.device=cuda \
		--policy.type=smolvla \
		--policy.push_to_hub=false \
		--dataset.root=/home1/linqi/test_dataset \
		--dataset.repo_id.0=pickup_circle_50episodes_single_arm_1107 \
		--dataset.repo_id.1=pickup_circle_20episodes_single_arm_1107_2 \
		--job_name=my_smolvla_training_multi_dataset \
		--dataset.image_transforms.enable=false \
		--batch_size=2 \
		--steps=100000 \
		--save_freq=5000 \
		--save_checkpoint=true \
		--log_freq=200 \
		--wandb.enable=false \
		--output_dir=tests/outputs/smolvla_multi_dataset
```

注意：使用 `.0`, `.1` 等来指定列表元素。

## 方法 3: 直接命令行

```bash
python -m lerobot.scripts.train \
    --policy.device=cuda \
    --policy.type=smolvla \
    --dataset.root=/home1/linqi/test_dataset \
    --dataset.repo_id.0=pickup_circle_50episodes_single_arm_1107 \
    --dataset.repo_id.1=pickup_circle_20episodes_single_arm_1107_2 \
    --job_name=my_training \
    --dataset.image_transforms.enable=false \
    --batch_size=2 \
    --steps=100000 \
    --output_dir=outputs/my_training \
    --wandb.enable=false
```

## 数据集目录结构

确保您的本地数据集结构如下：

```
/home1/linqi/test_dataset/
├── pickup_circle_50episodes_single_arm_1107/
│   ├── meta/
│   │   ├── info.json
│   │   ├── episodes.jsonl
│   │   └── stats.json
│   ├── data/
│   │   └── chunk-000/
│   │       ├── episode_000000.parquet
│   │       └── ...
│   └── videos/  (可选)
│       └── chunk-000/
│           └── ...
└── pickup_circle_20episodes_single_arm_1107_2/
    ├── meta/
    ├── data/
    └── videos/
```

## 验证配置

在运行训练前，可以先验证配置是否正确加载：

```bash
python -c "
from lerobot.configs.train import TrainPipelineConfig
import draccus

cfg = draccus.parse(TrainPipelineConfig, config_path='multi_dataset_config.yaml')
print('Dataset repo_id:', cfg.dataset.repo_id)
print('Type:', type(cfg.dataset.repo_id))
print('Is list:', isinstance(cfg.dataset.repo_id, list))
"
```

## 常见问题

### 问题 1: repo_id 被解析为字符串

如果您看到类似这样的错误：
```
'repo_id': "['dataset1', 'dataset2']"
```

**解决方案**: 使用 YAML 配置文件或使用 `.0`, `.1` 索引语法。

### 问题 2: FileNotFoundError: meta/info.json

如果错误显示找不到 `/home1/linqi/test_dataset/meta/info.json`：

**解决方案**: 确保每个数据集都在自己的子目录中：
- ✓ 正确: `/home1/linqi/test_dataset/dataset1/meta/info.json`
- ✗ 错误: `/home1/linqi/test_dataset/meta/info.json`

### 问题 3: HFValidationError

如果看到 HuggingFace 验证错误：

**解决方案**: 这通常意味着 repo_id 被错误解析。使用 YAML 配置文件可以避免这个问题。

## 完整的工作示例

已为您创建了配置文件 `multi_dataset_config.yaml`。运行：

```bash
cd /home1/linqi/bi_arm_lebro/lerobot
python -m lerobot.scripts.train --config multi_dataset_config.yaml
```

