#!/usr/bin/env python
"""
示例：如何加载多个本地数据集

这个示例展示了如何使用 MultiLeRobotDataset 加载多个本地存储的数据集。
"""

from pathlib import Path
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset, LeRobotDataset
from lerobot.datasets.factory import make_dataset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig

# ============================================================================
# 方法 1: 直接使用 MultiLeRobotDataset（推荐用于本地数据集）
# ============================================================================

def example_1_direct_multi_dataset():
    """
    直接使用 MultiLeRobotDataset 加载多个本地数据集
    
    假设您的本地数据集结构如下：
    /path/to/datasets/
    ├── dataset1/
    │   ├── meta/
    │   ├── data/
    │   └── videos/
    ├── dataset2/
    │   ├── meta/
    │   ├── data/
    │   └── videos/
    └── dataset3/
        ├── meta/
        ├── data/
        └── videos/
    """
    
    # 设置本地数据集的根目录
    local_root = Path("/path/to/datasets")
    
    # 数据集名称（对应子目录名）
    dataset_names = ["dataset1", "dataset2", "dataset3"]
    
    # 创建 MultiLeRobotDataset
    # 注意：root 是数据集的父目录，repo_ids 是每个数据集的子目录名
    dataset = MultiLeRobotDataset(
        repo_ids=dataset_names,
        root=local_root,
        image_transforms=None,
        delta_timestamps=None,
        download_videos=False,  # 本地数据集不需要下载
        video_backend="pyav",
    )
    
    print(f"成功加载 {len(dataset_names)} 个数据集")
    print(f"总帧数: {dataset.num_frames}")
    print(f"总 episodes: {dataset.num_episodes}")
    print(f"相机键: {dataset.camera_keys}")
    print(f"数据集索引映射: {dataset.repo_id_to_index}")
    
    # 获取一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n样本键: {sample.keys()}")
        print(f"该样本来自数据集: {dataset.repo_index_to_id[sample['dataset_index'].item()]}")
    
    return dataset


# ============================================================================
# 方法 2: 使用 factory.make_dataset（适合训练流程）
# ============================================================================

def example_2_factory_with_config():
    """
    使用 factory.make_dataset 和配置对象加载多个本地数据集
    这种方法适合集成到训练流程中
    """
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    
    # 创建数据集配置
    dataset_config = DatasetConfig(
        repo_id=["dataset1", "dataset2", "dataset3"],
        root="/path/to/datasets",  # 本地数据集的根目录
        episodes=None,  # 加载所有 episodes
        # 或者为每个数据集指定特定的 episodes:
        # episodes={
        #     "dataset1": [0, 1, 2],
        #     "dataset2": [0, 1],
        #     "dataset3": None,  # 加载所有
        # }
    )
    
    # 创建策略配置（这里以 Diffusion 为例）
    policy_config = DiffusionConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
    )
    
    # 创建训练配置
    train_config = TrainPipelineConfig(
        dataset=dataset_config,
        policy=policy_config,
        output_dir=Path("outputs/multi_dataset_training"),
        batch_size=8,
        num_workers=4,
    )
    
    # 使用 factory 创建数据集
    dataset = make_dataset(train_config)
    
    print(f"通过 factory 成功加载数据集")
    print(f"数据集类型: {type(dataset).__name__}")
    print(f"总帧数: {dataset.num_frames}")
    
    return dataset


# ============================================================================
# 方法 3: 处理不同路径的本地数据集
# ============================================================================

def example_3_different_paths():
    """
    如果您的数据集在完全不同的路径下，可以使用这种方法
    
    例如：
    - /data/robot1/dataset_a/
    - /data/robot2/dataset_b/
    - /home/user/dataset_c/
    """
    
    # 方案 A: 如果数据集在不同的绝对路径下，可以先单独加载每个数据集
    # 然后使用 torch.utils.data.ConcatDataset
    from torch.utils.data import ConcatDataset
    
    dataset1 = LeRobotDataset(
        repo_id="dataset_a",
        root="/data/robot1/dataset_a"
    )
    
    dataset2 = LeRobotDataset(
        repo_id="dataset_b",
        root="/data/robot2/dataset_b"
    )
    
    dataset3 = LeRobotDataset(
        repo_id="dataset_c",
        root="/home/user/dataset_c"
    )
    
    # 使用 ConcatDataset 连接它们
    combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])
    
    print(f"连接了 3 个数据集")
    print(f"总长度: {len(combined_dataset)}")
    
    # 注意：ConcatDataset 不会自动添加 dataset_index
    # 如果需要这个功能，建议使用 MultiLeRobotDataset
    
    return combined_dataset


# ============================================================================
# 方法 4: 从命令行训练脚本使用
# ============================================================================

def example_4_command_line_usage():
    """
    从命令行使用训练脚本
    """
    print("""
    从命令行使用多个本地数据集：
    
    方法 A - 直接指定多个数据集：
    ```bash
    python -m lerobot.scripts.train \\
        --dataset.repo_id="[dataset1,dataset2,dataset3]" \\
        --dataset.root="/path/to/datasets" \\
        --policy.type=diffusion \\
        --output_dir=outputs/multi_training
    ```
    
    方法 B - 使用配置文件：
    1. 创建配置文件 config.yaml:
    ```yaml
    dataset:
      repo_id:
        - dataset1
        - dataset2
        - dataset3
      root: /path/to/datasets
      episodes: null
    
    policy:
      type: diffusion
      n_obs_steps: 2
      horizon: 16
    
    output_dir: outputs/multi_training
    batch_size: 8
    num_workers: 4
    ```
    
    2. 运行训练：
    ```bash
    python -m lerobot.scripts.train --config config.yaml
    ```
    """)


# ============================================================================
# 主函数 - 运行示例
# ============================================================================

def main():
    print("=" * 70)
    print("多个本地数据集加载示例")
    print("=" * 70)
    
    print("\n注意：以下示例使用了占位符路径")
    print("请将 '/path/to/datasets' 替换为您实际的数据集路径\n")
    
    print("=" * 70)
    print("方法 1: 直接使用 MultiLeRobotDataset")
    print("=" * 70)
    print("""
# 示例代码：
from pathlib import Path
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

dataset = MultiLeRobotDataset(
    repo_ids=["dataset1", "dataset2", "dataset3"],
    root=Path("/path/to/datasets"),  # 数据集的父目录
    download_videos=False,  # 本地数据集不需要下载
)

print(f"总帧数: {dataset.num_frames}")
print(f"总 episodes: {dataset.num_episodes}")

# 迭代数据
for sample in dataset:
    dataset_idx = sample["dataset_index"]
    # 处理数据...
    break
    """)
    
    print("\n" + "=" * 70)
    print("方法 2: 使用 factory.make_dataset")
    print("=" * 70)
    print("""
# 示例代码：
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

dataset_config = DatasetConfig(
    repo_id=["dataset1", "dataset2"],
    root="/path/to/datasets",
)

policy_config = DiffusionConfig(n_obs_steps=2)

train_config = TrainPipelineConfig(
    dataset=dataset_config,
    policy=policy_config,
)

dataset = make_dataset(train_config)
    """)
    
    print("\n" + "=" * 70)
    print("方法 3: 处理不同路径的数据集")
    print("=" * 70)
    print("""
# 如果数据集在完全不同的路径下：
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import ConcatDataset

dataset1 = LeRobotDataset(repo_id="ds1", root="/path1/ds1")
dataset2 = LeRobotDataset(repo_id="ds2", root="/path2/ds2")
dataset3 = LeRobotDataset(repo_id="ds3", root="/path3/ds3")

combined = ConcatDataset([dataset1, dataset2, dataset3])
    """)
    
    print("\n" + "=" * 70)
    print("方法 4: 命令行使用")
    print("=" * 70)
    example_4_command_line_usage()
    
    print("\n" + "=" * 70)
    print("关键要点")
    print("=" * 70)
    print("""
1. 本地数据集目录结构：
   /your/datasets/root/
   ├── dataset1/
   │   ├── meta/
   │   ├── data/
   │   └── videos/
   └── dataset2/
       ├── meta/
       ├── data/
       └── videos/

2. 使用 MultiLeRobotDataset 时：
   - root: 指向数据集的父目录
   - repo_ids: 是每个数据集的子目录名称列表
   - 设置 download_videos=False（本地数据集不需要下载）

3. 每个样本会自动包含 "dataset_index" 键，
   表示该样本来自哪个数据集

4. 只有所有数据集共有的特征键会被保留

5. 查看完整文档：MULTI_DATASET_USAGE.md
    """)
    
    print("=" * 70)


if __name__ == "__main__":
    main()

