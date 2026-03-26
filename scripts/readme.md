
# Scripts 说明

这个目录包含数据转换、W&B 上传/回放，以及 RSL-RL 训练/播放相关脚本。


## 1) csv_to_npz.py

**作用**：
把 LAFAN/自定义 CSV 动作（root pose + dof_pos）重采样，并在 IsaacSim 中回放后导出成 tracking 任务可用的 motion npz（主要面向 G1 的示例流程）。

**常用参数**：
- `--input_file`：CSV 路径
- `--input_fps`：CSV 的帧率
- `--frame_range START END`：截取帧范围（从 1 开始，闭区间）
- `--output_name`：输出文件名（脚本内部决定输出目录/结构）
- `--output_fps`：输出 npz 帧率（默认 50Hz）

**示例**：

`/home/idear/Documents/IsaacLab/isaaclab.sh -p scripts/csv_to_npz.py --input_file datasets/motions_csv/dance1_subject2.csv --input_fps 30 --output_name dance1_subject2 --headless`

---

## 2) pkl_to_npz_mujoco.py

**作用**：
把 retarget 后的 `.pkl`（通常只有 `root_pos/root_rot/dof_pos`）通过 **MuJoCo FK** 补齐 body 级状态，并生成 BeyondMimic/Tracking 任务需要的 `motion.npz`：

- `fps`
- `joint_pos/joint_vel`
- `body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w`
- 额外写入 `joint_names/body_names`（用于在 IsaacLab 端按名字对齐）

**依赖**：只依赖 `mujoco` + `numpy`，不需要 IsaacSim。

**示例（X2）**：

`python scripts/pkl_to_npz_mujoco.py --input_pkl datasets/motions_pkl/x2_ultra/aiming1_subject.pkl --mjcf source/whole_body_tracking/whole_body_tracking/assets/x2_ultra/x2_ultra.xml --output_npz /tmp/motion_x2_aiming1.npz --output_fps 50`

---

## 3) upload_npz.py

**作用**：
把本地 `motion.npz` 上传到 W&B Registry（artifact type 默认 `motions`），并链接到 registry collection。

**实现细节**：
脚本会把你提供的 npz 文件在 artifact 内 **统一命名为 `motion.npz`**，避免训练脚本找不到文件名。

**示例**：

`python scripts/upload_npz.py --npz_path /tmp/motion_x2_aiming1.npz --collection_name aiming1_subject_x2 --registry_name motions --project csv_to_npz`

上传后训练端的 `--registry_name` 形如：

`<entity>/<project>/<collection_name>:latest`

例如：

`idear-fudan-university/csv_to_npz/aiming1_subject_x2:latest`

---

## 4) replay_npz.py

**作用**：
从 W&B Registry 下载某个 motion artifact 并在 IsaacSim 中回放（当前脚本使用 G1 配置作为演示）。

**参数**：
- `--registry_name`：W&B registry artifact 标识（会自动补 `:latest`）

**示例**：

`/home/idear/Documents/IsaacLab/isaaclab.sh -p scripts/replay_npz.py --registry_name idear-fudan-university/csv_to_npz/dance1_subject2:latest`

---

## 5) smoke_env.py

**作用**：
不依赖 W&B，直接用本地 `motion_file` 创建 env 并 `reset()` 一次后退出。

用于快速验证：
- URDF 导入是否正常
- default joint pos 是否超限
- motion.npz key/shape 是否符合要求

**示例（X2，headless）**：

`/home/idear/Documents/IsaacLab/isaaclab.sh -p scripts/smoke_env.py --task Tracking-Flat-X2-v0 --motion_file /tmp/motion_x2_aiming1.npz --num_envs 1 --headless`

---

## 6) scripts/rsl_rl/ 目录

这里是训练/播放相关脚本（RSL-RL PPO）。

### 6.1 cli_args.py

**作用**：
统一定义 train/play 共用的命令行参数，并把参数写回 `agent_cfg`（例如 resume、logger、wandb project）。

**注意**：
- 本仓库同时支持 `--checkpoint` 和 `--load_checkpoint`（两者等价）。

### 6.2 train.py

**作用**：
训练 Tracking 任务。会从 W&B Registry 下载 motion artifact 到本地 `artifacts/`，然后把本地路径写入 `env_cfg.commands.motion.motion_file`。

**示例（X2）**：

`/home/idear/Documents/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Tracking-Flat-X2-v0 --registry_name idear-fudan-university/csv_to_npz/aiming1_subject_x2:latest --num_envs 4096 --headless --logger wandb --log_project_name X2_Tracking --run_name aiming1`

### 6.3 play.py

**作用**：
播放（推理）已训练的策略。

两种使用方式：

1) **本地 logs 加载（推荐，离线、不依赖网络）**
- `--load_run`：run 文件夹名（位于 `logs/rsl_rl/<experiment_name>/` 下）
- `--checkpoint`/`--load_checkpoint`：如 `model_28500.pt`
- `--motion_file`：推荐显式给一个本地 npz（避免再次从 artifact 拉取）

示例：

`/home/idear/Documents/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task Tracking-Flat-X2-v0 --num_envs 2 --load_run 2026-03-25_21-45-41_aiming1 --load_checkpoint model_28500.pt --motion_file /tmp/motion_x2_aiming1.npz`

2) **从 W&B run 拉模型（会走网络下载，可能超时）**
- `--wandb_path`：形如 `entity/project/run_id`

示例：

`/home/idear/Documents/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task Tracking-Flat-X2-v0 --num_envs 2 --wandb_path idear-fudan-university/X2_Tracking/455exzjg --motion_file /tmp/motion_x2_aiming1.npz`

（说明：如果你同时提供 `--motion_file`，脚本会优先使用它，不再强制改回 artifact 里的 `motion.npz`。）

