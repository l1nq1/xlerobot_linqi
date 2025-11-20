# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import builtins
import datetime as dt
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.hub import HubMixin

TRAIN_CONFIG_NAME = "train_config.json"


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    steps: int = 100_000
    eval_freq: int = 20000
    log_freq: int = 100
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 2000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def __post_init__(self):
        self.checkpoint_path = None

    def validate(self):
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        # 强制刷新输出，确保日志立即显示
        sys.stdout.flush()
        sys.stderr.flush()
        
        policy_path = parser.get_path_arg("policy")


        if policy_path:
            # Only load the policy config
            policy_path_resolved = Path(policy_path).resolve()  
            # 获取命令行覆盖参数
            cli_overrides = parser.get_cli_overrides("policy")
            
            if self.policy is not None:
                # 将 YAML 中的 policy 配置转换为 CLI 参数格式
                yaml_policy_overrides = []
                try:
                    # 使用 draccus.encode 来正确处理嵌套的 dataclass
                    policy_dict = draccus.encode(self.policy)
                except Exception:
                    # 如果 draccus.encode 失败，回退到 asdict
                    policy_dict = asdict(self.policy)
                
                # 排除 path 和 type 字段，因为这些是特殊字段
                excluded_fields = {"path", "type", "pretrained_path"}
                for key, value in policy_dict.items():
                    if key not in excluded_fields and value is not None:
                        # 将值转换为字符串格式，处理列表、字典等复杂类型
                        if isinstance(value, (list, dict)):
                            value_str = json.dumps(value)
                        elif isinstance(value, bool):
                            value_str = str(value).lower()
                        else:
                            value_str = str(value)
                        yaml_policy_overrides.append(f"--{key}={value_str}")
                
                # 合并 YAML 配置和命令行参数，命令行参数优先级更高
                # 先添加 YAML 配置，再添加 CLI 参数，这样 CLI 参数会覆盖 YAML 配置
                all_overrides = yaml_policy_overrides + (cli_overrides or [])
                logging.info(f"从 YAML 配置文件中读取的 policy 参数: {len(yaml_policy_overrides)} 个")
                if yaml_policy_overrides:
                    logging.info(f"YAML policy 参数示例: {yaml_policy_overrides[:3]}...")
                logging.info(f"从命令行读取的 policy 覆盖参数: {len(cli_overrides or [])} 个")
                logging.info(f"合并后的覆盖参数总数: {len(all_overrides)} 个")
                cli_overrides = all_overrides
            
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

            if hasattr(self.policy, 'freeze_vision_encoder'):
                logging.info(f"✓ freeze_vision_encoder = {self.policy.freeze_vision_encoder}")
            if hasattr(self.policy, 'train_expert_only'):
                logging.info(f"✓ train_expert_only = {self.policy.train_expert_only}")
            logging.info("=" * 80)
        elif self.resume:
            logging.info("检测到 resume=True，将从检查点恢复训练...")
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )
            policy_path = Path(config_path).parent
            self.policy.pretrained_path = policy_path
            self.checkpoint_path = policy_path.parent
        else:
            logging.info("未指定预训练模型路径，将从头开始训练")
            logging.info("=" * 80)

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

        if self.policy.push_to_hub and not self.policy.repo_id:
            raise ValueError(
                "'policy.repo_id' argument missing. Please specify it to push the model to the hub."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict:
        return draccus.encode(self)

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    dataset: DatasetConfig | None = None  # NOTE: In RL, we don't need an offline dataset
