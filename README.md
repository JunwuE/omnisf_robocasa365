# RoboCasa365 post-training (openpi-omnisf)

在 `code/` 下用 openpi-omnisf 框架对 **RoboCasa365 target 数据集**做 omnisf 后训练。
训练框架（`openpi` + `scripts/train_align.py`）复用 `/root/nas/lfh/omnisf`，**不修改该仓库**；
本目录只提供配置、策略、数据处理与启动脚本。

> **源数据只读**：`data/` 下的 v3.0 原始数据不会被改动或删除。
> 转换结果写到 `data_v2.1/`（可用环境变量改），两步预处理（转换 / norm stats）都只用 CPU。

## 目录结构

```
code/
├── robocasa365/
│   ├── policy.py              # 12 维动作 policy（RoboCasa365Inputs / RoboCasa365Outputs）
│   ├── data_config.py         # LeRobotRoboCasa365DataConfig（repack + 数据/模型变换）
│   ├── configs.py             # 3 个 TrainConfig + 所有超参/路径（支持环境变量覆盖）
│   └── registry.py            # 把配置注册进 openpi，并加载 omnisf 的 train_align/compute_norm_stats
├── train.py                   # 训练入口（等价 scripts/train_align.py，额外注册本目录配置）
├── compute_norm_stats_fast.py # 快速 norm stats（直接读 Parquet 列，不解码视频、不需要 GPU）
├── compute_norm_stats.py      # openpi 原版 norm stats（走数据集管线，作为对照）
├── check_datasets.py          # 校验三个子集能否被读取（structure / full 两种模式）
├── preflight.py               # 训练前就绪检查（数据/统计量/权重/入口/真实样本）
└── scripts/
    ├── env.sh                  # 共享环境变量
    ├── convert_v30_to_v21.sh   # data/ 下 v3.0 → v2.1（源数据只读）
    ├── fix_video_timestamps.sh # 把转换出的每集视频 PTS 归零（并行重封装）
    ├── compute_norm_stats.sh   # 计算 3 个 split 的 norm stats（fast 版）
    ├── train_8gpu.sh           # 单机 8 卡训练启动脚本
    └── submit_detached.sh      # 脱离终端提交长任务（setsid+nohup，断连不影响）
```

## 数据流

```
data/                               data_v2.1/                        code/assets/
robocasa365-target-atomic (v3.0)  ──转换──▶  robocasa365-target-atomic (v2.1) ──训练读取
robocasa365-target-composite-*    (只读)      robocasa365-target-composite-*
        │
        └──(只读 Parquet 两列: observation.state / action)──▶ norm_stats.json
```

norm stats 只需要 `observation.state`(16) 和 `action`(12) 两列，所以**直接从源 Parquet 计算**，
不用等转换、也不读任何一帧视频。

## 三个训练配置

| 配置名 | 数据集目录 | 说明 |
| --- | --- | --- |
| `robocasa365_target_atomic` | `robocasa365-target-atomic` | Atomic-Seen 18 个原子任务 |
| `robocasa365_target_composite_seen` | `robocasa365-target-composite-seen` | Composite-Seen 16 个组合任务 |
| `robocasa365_target_composite_unseen` | `robocasa365-target-composite-unseen` | Composite-Unseen 16 个组合任务 |

三个 split 各训练一个模型（官方 Foundation-Model Learning 协议）。

## 默认训练设置

| 项 | 值 |
| --- | --- |
| 起点权重 | `checkpoints/omnisf_9.0_jax/experiments/35999/params`（omnisf 预训练） |
| 模型 | π0.5，`action_dim=32`（原生 12 维零填充），`action_horizon=50`，`max_token_len=200` |
| batch | micro batch **256** × 梯度累积 **2** = 有效 batch **512** |
| 并行 | `fsdp_devices=8`（单机 8 卡） |
| 训练步数 | 30000（`save_interval=5000`，`keep_period=10000`） |
| 优化器 | AdamW（grad clip 1.0），EMA 0.999 |
| LR | 官方 π0.5 RoboCasa 配置：warmup 1000 → 2.5e-5，余弦衰减到 2.5e-6 |
| 归一化 | 分位数归一化（`use_quantile_norm=True`），state/action 用 q01/q99 |
| 正则 | `state_dropout_prob=0.25` |
| 对齐 | omnisf 空间对齐开启：VLA 第 12 层 ↔ VGGT 第 -1 层，`align_loss_coeff=0.5`，bilinear pooling，`use_vlm_norm=True`，`use_camera_params=False`，8 卡共享 |

## 使用流程（前两步只用 CPU，不需要 GPU）

### 0. 环境

* openpi-omnisf 环境：`/root/nas/lfh/omnisf/.venv/bin/python`（默认 `$OPENPI_PYTHON`）。
* 转换需要 conda 环境 `lerobot`（新版 LeRobot API + `datasets`/`jsonlines`）以及 `ffmpeg`。

### 1. 数据转换：LeRobot v3.0 → v2.1（不改动源数据）

omnisf 的 lerobot 固定为 v2.1（`CODEBASE_VERSION="v2.1"`），所以要先把 v3.0 转成 v2.1。
脚本在「只包含指向源数据的软链接」的 staging 目录上做转换，产物写到 `data_v2.1/`，源目录保持原样。

```bash
cd /root/nas/ejw/robocasa365/code

bash scripts/convert_v30_to_v21.sh --dry-run      # 只打印计划
bash scripts/convert_v30_to_v21.sh                # 转换 3 个 split
bash scripts/convert_v30_to_v21.sh robocasa365-target-atomic   # 只转一个
bash scripts/convert_v30_to_v21.sh --overwrite    # 重新转换（覆盖 data_v2.1 下的旧产物）
```

转换耗时较长（每集每相机一次 `ffmpeg -c copy`），推荐**脱离终端提交**，SSH 断开也会继续跑：

```bash
bash scripts/submit_detached.sh convert bash scripts/convert_v30_to_v21.sh
# 提交后会打印日志路径和 PID（写在 code/logs/convert.pid）
tail -f code/logs/convert_*.log
```

完成后：源数据仍在 `data/`（v3.0，未改动），训练用数据在 `data_v2.1/`（v2.1）。
重新提交前先确认上一个任务已结束：`ps -fp "$(cat code/logs/convert.pid)"`。

### 1b. 修复每集视频的 PTS 起点（必须，CPU，并行约 1~2 小时）

any4lerobot 的转换器用 `ffmpeg -ss <起点> -c copy` 拆视频，但**没有把时间戳平移回 0**，
所以每集 mp4 的 `start_time` = 该集在拼接视频里的 `from_timestamp`（第一集是 0.1s）。
LeRobot 取样本时会做 1e-4s 精度的时间戳校验，会直接报错。用普通重封装归零（不重编码、帧数不变）：

```bash
bash scripts/submit_detached.sh tsfix bash scripts/fix_video_timestamps.sh --workers 16
# 只修一个 split / 调整并发
bash scripts/fix_video_timestamps.sh robocasa365-target-atomic --workers 32
```

脚本是幂等的（已经是 0 的会跳过），可重复运行/断点续跑；只改 `data_v2.1/`，源数据不动。
每个文件先写临时文件并校验（`start_time==0` 且帧数不变）再替换。

### 2. 计算归一化统计量（norm stats，CPU，几秒~几分钟）

```bash
bash scripts/compute_norm_stats.sh                                # 3 个 split 全算
bash scripts/compute_norm_stats.sh robocasa365_target_atomic      # 只算一个
bash scripts/compute_norm_stats.sh --max-frames 1000000           # 抽样加速（调试）
```

输出：`code/assets/<config_name>/norm_stats.json`。

### 3. 训练（单机 8 卡）

```bash
bash scripts/train_8gpu.sh                                                        # 默认 composite_seen
bash scripts/train_8gpu.sh robocasa365_target_atomic atomic_run0 --overwrite
bash scripts/train_8gpu.sh robocasa365_target_composite_unseen unseen_run0
```

日志：`code/logs/<config>_<时间>.out|.err`
权重：`code/checkpoints/<config_name>/<exp_name>/<step>/{params,assets}/`

续训 / 覆盖：

```bash
bash scripts/train_8gpu.sh robocasa365_target_atomic atomic_run0 --resume
bash scripts/train_8gpu.sh robocasa365_target_atomic atomic_run0 --overwrite
```

## 环境变量（`scripts/env.sh`）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `OPENPI_ROOT` | `/root/nas/lfh/omnisf` | omnisf 仓库根目录 |
| `OPENPI_PYTHON` | `$OPENPI_ROOT/.venv/bin/python` | 训练用解释器 |
| `ROBOCASA365_SOURCE_DATA_ROOT` | `../data` | v3.0 源数据（只读，不修改） |
| `ROBOCASA365_DATA_ROOT` | `../data_v2.1` | 转换后的 v2.1 训练数据 |
| `ROBOCASA365_STAGING_ROOT` | `../.cache/robocasa365_v21_staging` | 转换用 staging（软链接，自动清理） |
| `ROBOCASA365_ASSETS_DIR` | `code/assets` | norm stats 输出 |
| `ROBOCASA365_CHECKPOINT_DIR` | `code/checkpoints` | 权重输出 |
| `ROBOCASA365_PRETRAIN` | `omnisf` | `omnisf` / `pi05_base` / 任意路径 |
| `ROBOCASA365_BATCH_SIZE` | `256` | micro batch |
| `ROBOCASA365_GRAD_ACCUM` | `2` | 梯度累积步数 |
| `ROBOCASA365_FSDP_DEVICES` | `8` | FSDP 切分卡数 |
| `ROBOCASA365_NUM_WORKERS` | `8` | 每进程 dataloader worker 数 |
| `ROBOCASA365_NUM_TRAIN_STEPS` | `30000` | 训练步数 |
| `ROBOCASA365_SAVE_INTERVAL` / `ROBOCASA365_KEEP_PERIOD` | `5000` / `10000` | 保存/保留策略 |
| `ROBOCASA365_STATE_DROPOUT` | `0.25` | state dropout 概率 |
| `ROBOCASA365_EMA_DECAY` | `0.999` | EMA 衰减 |
| `ROBOCASA365_LR_WARMUP_STEPS` / `_LR_PEAK` / `_LR_DECAY_STEPS` / `_LR_DECAY` | `1000` / `2.5e-5` / `100000` / `2.5e-6` | LR 调度 |
| `ROBOCASA365_ALIGN_VLA_LAYER` / `_ALIGN_VGGT_LAYER` / `_ALIGN_LOSS_COEFF` | `12` / `-1` / `0.5` | 空间对齐 |
| `ROBOCASA365_ALIGN_VGGT_DEVICES` / `_ALIGN_FEATURE_BATCH_SIZE` | `8` / `32` | 对齐设备/特征 batch |

## 转换后的数据格式（LeRobot v2.1，写在 `data_v2.1/`）

```
robocasa365-target-<split>/
├── meta/
│   ├── info.json            # codebase_version = "v2.1"，features / total_episodes / chunks_size ...
│   ├── tasks.jsonl          # {"task_index": int, "task": "..."}
│   ├── episodes.jsonl       # 每条 episode 的长度、task_index 等
│   └── episodes_stats.jsonl # 每条 episode 的 min/max/mean/std（转换器生成，训练不用）
├── data/chunk-000/
│   └── episode_000000.parquet   # 一个 episode 一个文件
└── videos/chunk-000/
    ├── observation.images.robot0_agentview_left/episode_000000.mp4
    ├── observation.images.robot0_agentview_right/episode_000000.mp4
    └── observation.images.robot0_eye_in_hand/episode_000000.mp4
```

* 特征 schema 与原始 v3.0 一致，**没有降采样、没有过滤**（所有帧保留）：
  * `observation.state`: `float[16]` = `base_pos(3) + base_quat(4) + eef_pos_rel(3) + eef_quat_rel(4) + gripper_qpos(2)`
  * `action`: `float[12]` = `base_motion(4) + control_mode(1) + eef_delta_pos(3) + eef_delta_aa(3) + gripper(1)`
  * 三路视频 256×256（left / right / eye_in_hand）
  * `task_index`（配合 `prompt_from_task=True` 得到语言指令）
* 与 v3.0 的区别：v3.0 是 `data/chunk-000/file-000.parquet`、`videos/<cam>/chunk-000/file-000.mp4`、
  `meta/episodes/chunk-000/file-000.parquet`、`meta/tasks.parquet`；v2.1 拆成 per-episode 文件 +
  JSONL 元数据。视频拆分用 `ffmpeg -c copy`（流拷贝，不重编码）。
* staging 用软链接指向源数据，转换结束后 staging 只删软链接本身，源数据不受影响。

## norm stats 如何计算（fast 版）

入口：`compute_norm_stats_fast.py`（`scripts/compute_norm_stats.sh` 调用它）。它**只读两列 Parquet**：

1. 按 `meta/info.json` 找到 `data/**/*.parquet`（v3.0 / v2.1 都适用）；
2. 只读 `observation.state`(16) 与 `action`(12) 两列（列式读取，**不读视频、不逐帧走数据集**）；
3. 逐维计算 `mean / std / q01 / q99`（`np.quantile` 精确分位数）；
4. 把统计量右侧补零到模型 `action_dim=32`（常量填充维 `mean=std=q01=q99=0`，与 openpi 一致）；
5. 用 openpi 的 `normalize.save` 写成 `{"norm_stats": {"state": {...}, "actions": {...}}}`。

训练时按 `use_quantile_norm=True` 做 q01/q99 归一化：
`x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1`（`transforms.Normalize`）。
填充维（q01=q99=0）会映射到 `-1`，与 openpi 原生管线完全一致。
`checkpoints.save_state` 还会把 norm stats 一并写进每个 checkpoint 的 `assets/` 目录。

说明：

* norm stats 是**每个 split 单独一份**，不能跨数据集复用。
* 与 openpi 原版 `compute_norm_stats.py` 的区别：原版按 `action_horizon=50` 组动作块（episode 末尾会重复最后一帧），
  并且逐帧走 HF 数据集；fast 版统计的是**逐帧边缘分布**，数值上几乎一致，但快得多。
  需要完全复刻原版时可以改用 `compute_norm_stats.py --config-name <name>`（要求先完成 v2.1 转换）。
* 因为动作含 base motion，归一化覆盖 action 第 0–4 维（atomic 数据里 base_motion 常见 q01=q99=0，
  `control_mode` 的 q01=q99=-1，表示手臂激活）。

## 动作空间与相机

* 采用官方 RoboCasa365 口径：原生 **12 维动作**（含 base motion），state 用原始 **16 维**，
  两者都零填充到 32 维；不做 gripper/旋转缩放。
* π0.5 使用三路相机：`agentview_left → base_0_rgb`、`eye_in_hand → left_wrist_0_rgb`、
  `agentview_right → right_wrist_0_rgb`（右相机缺失会报错）。

## 备注

* 训练入口会调用 omnisf 的 `scripts/train_align.py`，因此要求 `align_enabled=True`（默认满足）。
* `train.py` 暴露的是 openpi 全部配置（含 omnisf 内置/外部配置），可以顺带选其它配置。
* 复现官方设置时，也可把 `ROBOCASA365_NUM_TRAIN_STEPS` 调到 100000。
