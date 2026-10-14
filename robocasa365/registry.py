"""Helpers to plug the RoboCasa365 configs into the installed openpi-omnisf checkout.

The omnisf training framework (``scripts/train_align.py``) lives in an external
checkout that must not be modified.  We therefore:

1. import ``openpi.training.config`` (which itself loads the omnisf
   ``external_config`` directory), then
2. append our own :class:`~openpi.training.config.TrainConfig` objects to the
   registry and rebuild ``_CONFIGS_DICT`` so ``tyro`` can select them, and
3. dynamically load the omnisf entry-point modules from
   ``$OPENPI_ROOT/scripts/`` and call their ``main``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig

CODE_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent

# Root of the openpi-omnisf checkout that provides `openpi` and `scripts/`.
OPENPI_ROOT = pathlib.Path(os.environ.get("OPENPI_ROOT", "/root/nas/lfh/omnisf")).expanduser().resolve()

_TRAIN_ENTRY = "scripts/train_align.py"
_NORM_STATS_ENTRY = "scripts/compute_norm_stats.py"


def ensure_code_on_path() -> None:
    """Make sure ``import robocasa365`` works regardless of the current directory."""
    code_root = str(CODE_ROOT)
    if code_root not in sys.path:
        sys.path.insert(0, code_root)


def load_module_from_path(module_name: str, file_path: str | pathlib.Path):
    path = pathlib.Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot load module '{module_name}': file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def register_configs(configs: Sequence["TrainConfig"]) -> list[str]:
    """Register ``configs`` with the openpi registry and return their names."""
    from openpi.training import config as _config  # noqa: PLC0415

    incoming = {config.name: config for config in configs}

    # Remove any stale duplicates from a previous registration, then append.
    _config._CONFIGS[:] = [config for config in _config._CONFIGS if config.name not in incoming]  # noqa: SLF001
    _config._CONFIGS.extend(configs)
    _config._CONFIGS_DICT = {config.name: config for config in _config._CONFIGS}  # noqa: SLF001

    for name in incoming:
        logging.info("Registered RoboCasa365 train config: %s", name)
    return list(incoming)


def _require_file(path: pathlib.Path, what: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{what} not found at {path}. "
            "Set OPENPI_ROOT to the openpi-omnisf checkout (default: /root/nas/lfh/omnisf)."
        )


def load_train_align_main():
    """Return ``scripts/train_align.py:main`` from the omnisf checkout."""
    _require_file(OPENPI_ROOT / _TRAIN_ENTRY, "train entry point")
    module = load_module_from_path("_robocasa365_omnisf_train_align", OPENPI_ROOT / _TRAIN_ENTRY)
    return module.main


def load_compute_norm_stats_main():
    """Return ``scripts/compute_norm_stats.py:main`` from the omnisf checkout."""
    _require_file(OPENPI_ROOT / _NORM_STATS_ENTRY, "norm-stats entry point")
    module = load_module_from_path("_robocasa365_omnisf_compute_norm_stats", OPENPI_ROOT / _NORM_STATS_ENTRY)
    return module.main


_LOCAL_PREFIXES = ("/", "./", "../", "~")


def enable_fast_local_loading() -> bool:
    """Use openpi's fast local LeRobot reader for local datasets.

    The stock ``LeRobotDataset`` path, when a local dataset is given, validates
    every per-episode parquet/video file (~36k ``is_file()`` calls for the
    8k-episode RoboCasa365 target splits) and materialises the full
    ``timestamp``/``episode_index`` columns to check timestamp sync -- and it
    does this once per training rank.  The framework's ``_FastLocalLeRobotDataset``
    skips that validation, which is safe here because our conversion script and
    ``preflight.py`` already validated each dataset, and training uses
    externally computed norm stats.

    Only local repo ids are affected. Disable with
    ``ROBOCASA365_FAST_LOCAL_LOADING=0``. Returns whether the patch is active.
    """
    if os.environ.get("ROBOCASA365_FAST_LOCAL_LOADING", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False

    from openpi.training import data_loader as _data_loader  # noqa: PLC0415

    original = _data_loader.create_transformed_single_torch_dataset
    if getattr(original, "_robocasa365_fast_local", False):
        return True

    def patched(
        data_config,
        action_horizon,
        model_config,
        *,
        skip_norm_stats: bool = False,
        tail_transforms=(),
        fast_local_loading: bool = False,
        torch_dataset=None,
    ):
        repo_id = getattr(data_config, "repo_id", None) or ""
        is_local = repo_id.startswith(_LOCAL_PREFIXES)
        return original(
            data_config,
            action_horizon,
            model_config,
            skip_norm_stats=skip_norm_stats,
            tail_transforms=tail_transforms,
            fast_local_loading=fast_local_loading or is_local,
            torch_dataset=torch_dataset,
        )

    patched._robocasa365_fast_local = True  # type: ignore[attr-defined]
    _data_loader.create_transformed_single_torch_dataset = patched
    logging.info("Enabled fast local LeRobot loading (ROBOCASA365_FAST_LOCAL_LOADING).")
    return True
