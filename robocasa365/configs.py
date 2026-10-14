"""Concrete omnisf post-training configs for the RoboCasa365 target splits.

Each config post-trains pi0.5 (with omnisf spatial-forcing alignment) on one
RoboCasa365 target split, starting from the omnisf pretrained checkpoint.

Defaults follow, unless noted otherwise:

* the omnisf post-training recipe used for LIBERO / RoboTwin
  (``omnisf_11.1_jax`` / ``omnisf_12.1_jax``): 8-way FSDP, EMA 0.999,
  state dropout 0.25, quantile normalization, spatial-forcing alignment on VLA
  layer 12 and VGGT layer -1 with coefficient 0.5;
* the official RoboCasa365 pi0.5 optimization settings
  (``robocasa-benchmark/openpi``): ``max_token_len=200`` and a cosine schedule
  with 1k warmup, peak LR 2.5e-5, decaying to 2.5e-6 over 100k steps.

All paths can be overridden with environment variables so the code stays
relocatable (see ``scripts/env.sh``).
"""

from __future__ import annotations

import os
import pathlib

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import MixedLeRobotDataConfig
from openpi.training.config import MixedDataSourceConfig
from openpi.training.config import TrainConfig
from openpi.training import weight_loaders

from robocasa365.data_config import LeRobotRoboCasa365DataConfig
from robocasa365.registry import CODE_ROOT
from robocasa365.registry import OPENPI_ROOT
from robocasa365.registry import WORKSPACE_ROOT


def _env_path(name: str, default: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.environ.get(name, str(default))).expanduser().resolve()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Read-only source: the flat LeRobot v3.0 mirrors downloaded from HuggingFace.
# This directory is never modified.
SOURCE_DATA_ROOT = _env_path("ROBOCASA365_SOURCE_DATA_ROOT", WORKSPACE_ROOT / "data")
# Directory holding the (converted to v2.1) RoboCasa365 target datasets used for
# training.  Kept separate from SOURCE_DATA_ROOT so the source stays untouched.
DATA_ROOT = _env_path("ROBOCASA365_DATA_ROOT", WORKSPACE_ROOT / "data_v2.1")
# Normalization-stats assets live next to this code by default.
ASSETS_DIR = _env_path("ROBOCASA365_ASSETS_DIR", CODE_ROOT / "assets")
# Checkpoints are written here.
CHECKPOINT_DIR = _env_path("ROBOCASA365_CHECKPOINT_DIR", CODE_ROOT / "checkpoints")

# Pretrained omnisf pi0.5 checkpoint (36000-step omnisf_9.0 run).
PRETRAIN_PARAMS = _env_path(
    "ROBOCASA365_PRETRAIN_PARAMS",
    OPENPI_ROOT / "checkpoints/omnisf_9.0_jax/experiments/35999/params",
)
# Alternative base checkpoint, selectable with ROBOCASA365_PRETRAIN=pi05_base.
PI05_BASE_PARAMS = _env_path("ROBOCASA365_PI05_BASE_PARAMS", OPENPI_ROOT / "checkpoints/pi05_base/params")
# omnivggt / VGGT weights used by the spatial-forcing alignment branch.
VGGT_PATH = _env_path("ROBOCASA365_VGGT_PATH", OPENPI_ROOT / "checkpoints/omnivggt")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# split key -> dataset directory name (shared by the v3.0 source and v2.1 output).
SPLIT_DATASET_NAMES = {
    "atomic": "robocasa365-target-atomic",
    "composite_seen": "robocasa365-target-composite-seen",
    "composite_unseen": "robocasa365-target-composite-unseen",
}
# Converted v2.1 datasets (used for training).
DATASET_DIRS = {split: DATA_ROOT / name for split, name in SPLIT_DATASET_NAMES.items()}
# Original v3.0 datasets (read-only; used to compute norm stats directly).
SOURCE_DATASET_DIRS = {split: SOURCE_DATA_ROOT / name for split, name in SPLIT_DATASET_NAMES.items()}

# Config name == f"{CONFIG_PREFIX}_{split}".
CONFIG_PREFIX = "robocasa365_target"

# Config name for the run that mixes all three subsets in a single model.
MIXED_CONFIG_NAME = f"{CONFIG_PREFIX}_mixed"
# Sampling weights for the mixed run, as a comma-separated list ordered like
# SPLIT_DATASET_NAMES.  "1,1,1" => sample proportionally to dataset size (plain
# union); e.g. "3,1,1" oversamples atomic relative to the composites.
_MIXED_WEIGHTS_RAW = os.environ.get("ROBOCASA365_MIXED_WEIGHTS", "1.0,1.0,1.0")
_MIXED_WEIGHT_VALUES = [float(v) for v in _MIXED_WEIGHTS_RAW.split(",")]
if len(_MIXED_WEIGHT_VALUES) != len(SPLIT_DATASET_NAMES):
    raise ValueError(
        f"ROBOCASA365_MIXED_WEIGHTS must have {len(SPLIT_DATASET_NAMES)} comma-separated values, "
        f"got {_MIXED_WEIGHTS_RAW!r}"
    )
MIXED_WEIGHTS = dict(zip(SPLIT_DATASET_NAMES, _MIXED_WEIGHT_VALUES, strict=True))


def config_name_for(split: str) -> str:
    if split not in SPLIT_DATASET_NAMES:
        raise KeyError(f"Unknown RoboCasa365 target split '{split}'. Choices: {sorted(SPLIT_DATASET_NAMES)}")
    return f"{CONFIG_PREFIX}_{split}"


# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
# One node, 8 GPUs.  Physical (micro) batch 512 x gradient-accumulation 2
# => effective batch 1024 (64 samples per device per micro step).
BATCH_SIZE = int(os.environ.get("ROBOCASA365_BATCH_SIZE", "512"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("ROBOCASA365_GRAD_ACCUM", "2"))
FSDP_DEVICES = int(os.environ.get("ROBOCASA365_FSDP_DEVICES", "8"))
NUM_WORKERS = int(os.environ.get("ROBOCASA365_NUM_WORKERS", "8"))
NUM_TRAIN_STEPS = int(os.environ.get("ROBOCASA365_NUM_TRAIN_STEPS", "20000"))
# Save only at the end of the 20k run (the loop additionally always saves the
# final step, num_train_steps - 1).  Each checkpoint is ~42 GB, so a 20000-step
# run with save_interval=5000 would write 4 of them.
SAVE_INTERVAL = int(os.environ.get("ROBOCASA365_SAVE_INTERVAL", "20000"))
KEEP_PERIOD = int(os.environ.get("ROBOCASA365_KEEP_PERIOD", "10000"))
EMA_DECAY = float(os.environ.get("ROBOCASA365_EMA_DECAY", "0.999"))
STATE_DROPOUT_PROB = float(os.environ.get("ROBOCASA365_STATE_DROPOUT", "0.25"))

# Official RoboCasa365 pi0.5 optimization schedule.
LR_WARMUP_STEPS = int(os.environ.get("ROBOCASA365_LR_WARMUP_STEPS", "1000"))
LR_PEAK = float(os.environ.get("ROBOCASA365_LR_PEAK", "2.5e-5"))
LR_DECAY_STEPS = int(os.environ.get("ROBOCASA365_LR_DECAY_STEPS", "100000"))
LR_DECAY = float(os.environ.get("ROBOCASA365_LR_DECAY", "2.5e-6"))

# omnisf spatial-forcing alignment.
ALIGN_VLA_LAYER = int(os.environ.get("ROBOCASA365_ALIGN_VLA_LAYER", "12"))
ALIGN_VGGT_LAYER = int(os.environ.get("ROBOCASA365_ALIGN_VGGT_LAYER", "-1"))
ALIGN_LOSS_COEFF = float(os.environ.get("ROBOCASA365_ALIGN_LOSS_COEFF", "0.5"))
ALIGN_FEATURE_BATCH_SIZE = int(os.environ.get("ROBOCASA365_ALIGN_FEATURE_BATCH_SIZE", "32"))
ALIGN_VGGT_DEVICES = int(os.environ.get("ROBOCASA365_ALIGN_VGGT_DEVICES", str(FSDP_DEVICES)))


def _pretrain_params() -> pathlib.Path:
    """Pretrained checkpoint selected via ``ROBOCASA365_PRETRAIN``."""
    choice = os.environ.get("ROBOCASA365_PRETRAIN", "omnisf").strip().lower()
    if choice in {"omnisf", "omnisf_9.0", "omnisf_9.0_jax"}:
        return PRETRAIN_PARAMS
    if choice in {"pi05", "pi05_base"}:
        return PI05_BASE_PARAMS
    # Fall back to treating the value as a literal path.
    return _env_path("ROBOCASA365_PRETRAIN", pathlib.Path(choice))


def _build_train_config(name: str, data) -> TrainConfig:
    """Shared omnisf post-training TrainConfig; only the data factory differs."""
    model = pi0_config.Pi0Config(pi05=True)

    return TrainConfig(
        name=name,
        project_name="robocasa365_omnisf",
        model=model,
        data=data,
        state_dropout_prob=STATE_DROPOUT_PROB,
        freeze_filter=model.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_pretrain_params())),
        assets_base_dir=str(ASSETS_DIR),
        checkpoint_base_dir=str(CHECKPOINT_DIR),
        num_train_steps=NUM_TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        keep_period=KEEP_PERIOD,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_workers=NUM_WORKERS,
        fsdp_devices=FSDP_DEVICES,
        ema_decay=EMA_DECAY,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=LR_WARMUP_STEPS,
            peak_lr=LR_PEAK,
            decay_steps=LR_DECAY_STEPS,
            decay_lr=LR_DECAY,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # ---- omnisf spatial-forcing alignment ----
        align_enabled=True,
        align_target_model="auto",
        vggt_weight_path=str(VGGT_PATH),
        vla_layers_align=ALIGN_VLA_LAYER,
        vla_adapt_layers_align=False,
        vggt_layers_align=ALIGN_VGGT_LAYER,
        vggt_adapt_layers_align=False,
        pooling_func="bilinear",
        use_vggt_pe=True,
        use_vlm_norm=True,
        align_loss_coeff=ALIGN_LOSS_COEFF,
        ignore_img_padding_area=True,
        # omnivggt does not receive camera parameters.
        use_camera_params=False,
        # alignment device placement (shared GPUs on a single 8-GPU node).
        align_device_mode="shared",
        align_vggt_devices=ALIGN_VGGT_DEVICES,
        align_feature_batch_size=ALIGN_FEATURE_BATCH_SIZE,
    )


def build_train_config(split: str) -> TrainConfig:
    """Post-training config for a single target subset (one run per subset)."""
    name = config_name_for(split)
    data = LeRobotRoboCasa365DataConfig(
        repo_id=str(DATASET_DIRS[split]),
        assets=AssetsConfig(
            assets_dir=str(ASSETS_DIR),
            asset_id=name,
        ),
        base_config=DataConfig(prompt_from_task=True),
        use_quantile_norm=True,
    )
    return _build_train_config(name, data)


def build_mixed_train_config() -> TrainConfig:
    """Mixed config: one model trained on all three target subsets together.

    Sampling is proportional to ``weight * len(subset)``; with the default
    ``ROBOCASA365_MIXED_WEIGHTS=1,1,1`` this is a plain union of the three
    datasets (dominated by the two composite splits, which have far more frames).
    All subsets share the same normalisation statistics, taken from the merged
    asset ``robocasa365_target_mixed``.
    """
    sources = [
        MixedDataSourceConfig(
            name=split,
            weight=float(MIXED_WEIGHTS[split]),
            data_factory=LeRobotRoboCasa365DataConfig(
                repo_id=str(DATASET_DIRS[split]),
                base_config=DataConfig(prompt_from_task=True),
                use_quantile_norm=True,
            ),
        )
        for split in SPLIT_DATASET_NAMES
    ]
    data = MixedLeRobotDataConfig(
        repo_id=str(DATA_ROOT / "robocasa365-target-mixed"),
        assets=AssetsConfig(assets_dir=str(ASSETS_DIR), asset_id=MIXED_CONFIG_NAME),
        base_config=DataConfig(prompt_from_task=True),
        sources=sources,
        use_quantile_norm=True,
        fast_local_loading=True,
    )
    return _build_train_config(MIXED_CONFIG_NAME, data)


CONFIGS: list[TrainConfig] = [build_train_config(split) for split in SPLIT_DATASET_NAMES]
CONFIGS.append(build_mixed_train_config())

__all__ = [
    "ASSETS_DIR",
    "CHECKPOINT_DIR",
    "CONFIGS",
    "CONFIG_PREFIX",
    "DATASET_DIRS",
    "DATA_ROOT",
    "MIXED_CONFIG_NAME",
    "MIXED_WEIGHTS",
    "PRETRAIN_PARAMS",
    "SOURCE_DATASET_DIRS",
    "SOURCE_DATA_ROOT",
    "SPLIT_DATASET_NAMES",
    "VGGT_PATH",
    "build_mixed_train_config",
    "build_train_config",
    "config_name_for",
]
