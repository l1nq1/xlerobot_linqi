# 多数据集加载功能使用指南

本文档说明如何使用 LeRobot 的多数据集加载功能。

## 概述

多数据集加载功能允许您同时加载和训练多个数据集。数据集会被自动连接，只保留所有数据集中共有的特征键。

## 功能实现

已完成以下修改以支持多数据集加载：

1. **factory.py**: 更新 `make_dataset()` 函数以支持 `repo_id` 列表
2. **lerobot_dataset.py**: 为 `MultiLeRobotDataset` 添加 `meta` 属性以保持 API 一致性
3. **train.py**: 移除阻止多数据集使用的 `NotImplementedError` 检查

## 使用方法

### 方法 1: 使用配置文件

在您的训练配置 YAML 文件中:

```yaml
dataset:
  repo_id:
    - "your-org/dataset-1"
    - "your-org/dataset-2"
    - "your-org/dataset-3"
  root: null  # 可选，数据集存储路径
  episodes: null  # 可选，或者使用字典格式指定每个数据集的 episodes
  # episodes:
  #   "your-org/dataset-1": [0, 1, 2]
  #   "your-org/dataset-2": [0, 1]
```

### 方法 2: 使用代码

```python
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

# 创建数据集配置
dataset_config = DatasetConfig(
    repo_id=[
        "your-org/dataset-1",
        "your-org/dataset-2",
        "your-org/dataset-3"
    ],
    root=None,
)

# 创建训练配置
policy_config = DiffusionConfig(
    # 您的策略配置...
)

train_config = TrainPipelineConfig(
    dataset=dataset_config,
    policy=policy_config,
    # 其他训练参数...
)

# 创建多数据集
dataset = make_dataset(train_config)

# 现在您可以使用 dataset 进行训练
print(f"总帧数: {dataset.num_frames}")
print(f"总 episodes: {dataset.num_episodes}")
print(f"数据集映射: {dataset.repo_id_to_index}")
```

### 方法 3: 直接使用 MultiLeRobotDataset

```python
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

# 直接创建多数据集实例
dataset = MultiLeRobotDataset(
    repo_ids=[
        "your-org/dataset-1",
        "your-org/dataset-2",
        "your-org/dataset-3"
    ],
    root=None,  # 可选
    image_transforms=None,  # 可选
    delta_timestamps=None,  # 可选
    video_backend="pyav",  # 可选
)

# 使用数据集
for i, batch in enumerate(dataset):
    # batch 包含一个额外的键 "dataset_index"，指示该样本来自哪个数据集
    dataset_idx = batch["dataset_index"]
    print(f"样本 {i} 来自数据集: {dataset.repo_index_to_id[dataset_idx.item()]}")
    
    # 处理其他数据...
    if i >= 10:  # 只显示前 10 个样本
        break
```

## 重要说明

1. **特征键兼容性**: 多数据集功能只保留所有数据集中共有的特征键。如果某个键在某些数据集中不存在，它将被禁用并记录警告。

2. **FPS 兼容性**: 目前假设所有数据集具有相同的 FPS（帧率）。使用第一个数据集的元数据来解析 `delta_timestamps`。

3. **Episodes 配置**: 
   - 对于单个数据集: `episodes` 是整数列表
   - 对于多个数据集: `episodes` 是字典，映射 repo_id 到 episodes 列表

4. **数据集索引**: 每个批次数据中会自动添加 `"dataset_index"` 键，指示该样本来自哪个数据集（按 repo_ids 列表中的顺序索引）。

5. **统计聚合**: 多个数据集的统计信息会被自动聚合，用于数据归一化。

## 训练示例

使用命令行训练：

```bash
python -m lerobot.scripts.train \
    --dataset.repo_id="[your-org/dataset-1,your-org/dataset-2,your-org/dataset-3]" \
    --policy.type=diffusion \
    --output_dir=outputs/multi_dataset_training \
    # 其他参数...
```

或使用配置文件：

```bash
python -m lerobot.scripts.train \
    --config path/to/your/config.yaml
```

## 测试

现有的测试文件中已有 `test_multidataset_frames()` 测试函数（位于 `tests/datasets/test_datasets.py`），可以取消跳过该测试来验证功能：

```python
# 在 tests/datasets/test_datasets.py 中
# 移除 @pytest.mark.skip("TODO after fix multidataset") 装饰器
def test_multidataset_frames():
    # 测试代码...
```

## API 参考

### MultiLeRobotDataset

主要属性：
- `repo_ids`: 数据集仓库 ID 列表
- `num_frames`: 总帧数
- `num_episodes`: 总 episode 数
- `fps`: 帧率（来自第一个数据集）
- `camera_keys`: 相机键列表
- `stats`: 聚合的统计信息
- `meta`: 元数据访问器（与 `LeRobotDataset` API 兼容）
- `repo_id_to_index`: 映射 repo_id 到索引的字典
- `repo_index_to_id`: 映射索引到 repo_id 的字典

主要方法：
- `__len__()`: 返回总帧数
- `__getitem__(idx)`: 返回指定索引的数据批次，包含 `dataset_index` 键

## 故障排除

如果遇到问题：

1. **特征不匹配**: 确保所有数据集具有兼容的特征。检查日志中关于禁用特征的警告。

2. **FPS 不匹配**: 如果数据集的 FPS 不同，可能需要在使用前进行重采样。

3. **内存问题**: 加载多个大型数据集可能消耗大量内存。考虑使用 `episodes` 参数只加载部分 episodes。

## 贡献

如果您发现问题或有改进建议，请提交 issue 或 pull request。

