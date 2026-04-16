from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """Load one or more motion NPZ files and concatenate them into a single dataset.

    When multiple files are provided, each file is treated as an independent motion clip.
    All clips are concatenated along the time axis into single tensors for GPU-friendly indexing.
    Per-clip metadata (``clip_starts``, ``clip_lengths``) is stored so that the caller can
    map a global frame index back to (clip_index, local_frame).
    """

    def __init__(self, motion_files: str | list[str], body_indexes: Sequence[int], device: str = "cpu"):
        # 支持单文件路径或多文件路径列表
        if isinstance(motion_files, str):
            motion_files = [motion_files]
        assert len(motion_files) > 0, "At least one motion file must be provided."
        for f in motion_files:
            assert os.path.isfile(f), f"Invalid file path: {f}"

        # 用于逐 clip 累积数据的临时列表
        all_joint_pos = []
        all_joint_vel = []
        all_body_pos_w = []
        all_body_quat_w = []
        all_body_lin_vel_w = []
        all_body_ang_vel_w = []
        clip_lengths: list[int] = []  # 每个 clip 的帧数

        self.joint_names: list[str] | None = None
        self.body_names: list[str] | None = None
        self.fps: float = 0.0

        for motion_file in motion_files:
            data = np.load(motion_file, allow_pickle=True)
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if self.fps == 0.0:
                self.fps = fps
            else:
                # 所有 clip 必须具有相同的帧率，否则拼接后时间语义不一致
                assert abs(self.fps - fps) < 1e-3, (
                    f"FPS mismatch across motion files: {self.fps} vs {fps} in {motion_file}"
                )

            # 关节名 / body 名仅从第一个包含该字段的文件中读取（所有文件关节结构一致）
            if self.joint_names is None and "joint_names" in data:
                self.joint_names = [str(x) for x in np.asarray(data["joint_names"]).tolist()]
            if self.body_names is None and "body_names" in data:
                self.body_names = [str(x) for x in np.asarray(data["body_names"]).tolist()]

            all_joint_pos.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=device))
            all_joint_vel.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=device))
            all_body_pos_w.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
            all_body_quat_w.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
            all_body_lin_vel_w.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device))
            all_body_ang_vel_w.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))
            clip_lengths.append(all_joint_pos[-1].shape[0])

        # 将所有 clip 沿时间轴拼接为单一大张量，便于 GPU 单次索引
        self.joint_pos = torch.cat(all_joint_pos, dim=0)
        self.joint_vel = torch.cat(all_joint_vel, dim=0)
        self._body_pos_w = torch.cat(all_body_pos_w, dim=0)
        self._body_quat_w = torch.cat(all_body_quat_w, dim=0)
        self._body_lin_vel_w = torch.cat(all_body_lin_vel_w, dim=0)
        self._body_ang_vel_w = torch.cat(all_body_ang_vel_w, dim=0)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]  # 所有 clip 的总帧数

        # 每个 clip 的元数据：帧数和在拼接张量中的起始帧索引
        self.num_clips = len(clip_lengths)
        self.clip_lengths = torch.tensor(clip_lengths, dtype=torch.long, device=device)
        starts = [0]
        for l in clip_lengths[:-1]:
            starts.append(starts[-1] + l)
        self.clip_starts = torch.tensor(starts, dtype=torch.long, device=device)  # 每个 clip 的全局起始帧

        print(f"[MotionLoader] Loaded {self.num_clips} clip(s), "
              f"total frames: {self.time_step_total}, fps: {self.fps}")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.robot_body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # 优先使用 motion_files 列表，若为空则回退到单文件 motion_file（向后兼容）
        motion_files = self.cfg.motion_files if self.cfg.motion_files else [self.cfg.motion_file]

        # 用第一个文件解析 body 索引（假设所有文件的 body 结构一致）
        motion_body_indexes = self._resolve_motion_body_indexes(motion_files[0], self.cfg.body_names)
        self.motion_body_indexes = torch.tensor(motion_body_indexes, dtype=torch.long, device=self.device)

        self.motion = MotionLoader(motion_files, self.motion_body_indexes, device=self.device)
        self._align_motion_joint_order_if_needed()

        # 每个 env 在当前 clip 内的本地时间步（非全局帧索引）
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 每个 env 当前正在播放的 clip 索引
        self.clip_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # --- SONIC 风格的 bin-based 自适应采样初始化 ---
        # 每个 clip 按固定时长（1秒）划分 bin，跨所有 clip 汇总为全局 bin 列表
        # frames_per_second：每秒对应的动作帧数（= 环境控制频率，因为每个策略步前进 1 帧）
        frames_per_second = max(1.0, 1.0 / (env.cfg.decimation * env.cfg.sim.dt))
        self._bins_per_clip = torch.zeros(self.motion.num_clips, dtype=torch.long, device=self.device)
        for i in range(self.motion.num_clips):
            # 每个 clip 的 bin 数 ≈ clip 时长（秒），至少 1 个 bin
            self._bins_per_clip[i] = int(self.motion.clip_lengths[i].item() // frames_per_second) + 1
        self.bin_count = int(self._bins_per_clip.sum().item())  # 全局 bin 总数
        # 每个 clip 的 bin 在全局 bin 列表中的起始偏移（用于 clip↔bin 双向映射）
        self._bin_offsets = torch.zeros(self.motion.num_clips, dtype=torch.long, device=self.device)
        for i in range(1, self.motion.num_clips):
            self._bin_offsets[i] = self._bin_offsets[i - 1] + self._bins_per_clip[i - 1]

        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_clip_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_active_clips"] = torch.zeros(self.num_envs, device=self.device)

    def _global_time_steps(self) -> torch.Tensor:
        """将 per-env 的 (clip_index, 本地时间步) 转换为拼接张量的全局帧索引。"""
        # clip_starts[clip_indices]：当前 clip 在拼接数据中的起始帧
        # + time_steps：clip 内的本地偏移
        return self.motion.clip_starts[self.clip_indices] + self.time_steps

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self._global_time_steps()]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self._global_time_steps()]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._global_time_steps()] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._global_time_steps()]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._global_time_steps()]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._global_time_steps()]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._global_time_steps(), self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._global_time_steps(), self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._global_time_steps(), self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._global_time_steps(), self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_body_indexes]

    def _resolve_motion_body_indexes(self, motion_file: str, body_names: Sequence[str]) -> list[int]:
        """Resolve body indices into the motion file.

        Backward compatible behavior:
        - If the motion file does not contain `body_names`, we assume motion bodies follow the simulator's body index
          ordering (i.e., produced by IsaacLab logger), and use the robot body indices directly.
        - If `body_names` exists, we map by name and return indices in the same order as `body_names`.
        """
        data = np.load(motion_file, allow_pickle=True)
        if "body_names" not in data:
            return self.robot.find_bodies(list(body_names), preserve_order=True)[0]

        motion_body_names = [str(x) for x in np.asarray(data["body_names"]).tolist()]
        name_to_index = {name: i for i, name in enumerate(motion_body_names)}

        # Handle common naming differences across exporters (e.g., MuJoCo vs URDF).
        # The simulator might use `base_link` while the motion uses `pelvis`.
        aliases: dict[str, list[str]] = {
            "base_link": ["pelvis"],
            "pelvis": ["base_link"],
        }

        resolved_indexes: list[int] = []
        missing: list[str] = []
        for name in body_names:
            if name in name_to_index:
                resolved_indexes.append(name_to_index[name])
                continue
            # try aliases
            found = False
            for alt in aliases.get(name, []):
                if alt in name_to_index:
                    resolved_indexes.append(name_to_index[alt])
                    found = True
                    break
            if not found:
                missing.append(name)

        if missing:
            raise ValueError(
                "Motion file is missing required bodies: "
                + ", ".join(missing)
                + f". Available example: {motion_body_names[:10]}..."
            )

        return resolved_indexes

    def _align_motion_joint_order_if_needed(self):
        """Align motion joint order to the simulator joint order when motion file stores joint names.

        Older motions produced by IsaacLab typically omit `joint_names` and already match simulator ordering.
        """
        if self.motion.joint_names is None:
            return

        robot_joint_names = list(self.robot.joint_names)
        name_to_index = {name: i for i, name in enumerate(self.motion.joint_names)}
        missing = [name for name in robot_joint_names if name not in name_to_index]
        if missing:
            raise ValueError(
                "Motion file is missing required joints: "
                + ", ".join(missing)
                + f". Available example: {self.motion.joint_names[:10]}..."
            )
        reindex = torch.tensor([name_to_index[name] for name in robot_joint_names], device=self.motion.joint_pos.device)
        self.motion.joint_pos = self.motion.joint_pos[:, reindex]
        self.motion.joint_vel = self.motion.joint_vel[:, reindex]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """SONIC 风格的 bin-based 自适应动作采样。

        对每个 bin 追踪失败率 f_i，将其 cap 到 beta * f_bar（平均失败率的 beta 倍），
        防止极难 bin 被过度采样。最终采样概率将归一化的 cap 后失败分布与均匀分布混合：

            p_i = alpha * p_hat_i + (1 - alpha) * (1 / N)

        其中 p_hat_i = capped_f_i / sum(capped_f_i)，N 为 bin 总数。
        """
        # 检查哪些 env 是因失败（terminated）而触发 resample 的
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            # 将失败 env 的 (clip_index, 本地时间步) 映射到全局 bin 索引，统计各 bin 的失败次数
            clip_ids = self.clip_indices[env_ids][episode_failed]
            local_ts = self.time_steps[env_ids][episode_failed]
            bins_per_clip_for_env = self._bins_per_clip[clip_ids]
            clip_lengths_for_env = self.motion.clip_lengths[clip_ids]
            # 本地 bin 索引 = 本地时间步 / clip 帧数 * clip bin 数，clamp 防止越界
            local_bin = torch.min(
                (local_ts * bins_per_clip_for_env) // torch.clamp(clip_lengths_for_env, min=1),
                bins_per_clip_for_env - 1,
            )
            local_bin = torch.max(local_bin, torch.zeros_like(local_bin))
            # 加上该 clip 的全局 bin 偏移，得到全局 bin 索引
            global_bin = self._bin_offsets[clip_ids] + local_bin
            self._current_bin_failed[:] = torch.bincount(global_bin, minlength=self.bin_count).float()

        # --- SONIC 采样公式 ---
        f = self.bin_failed_count.clone()  # 各 bin 的 EMA 平滑失败率
        f_bar = f.mean()  # 所有 bin 的平均失败率
        beta = self.cfg.adaptive_beta
        # Step 1：将失败率 cap 到 beta * f_bar，避免极端 bin 主导采样
        if f_bar > 0:
            f = torch.clamp(f, max=beta * f_bar)

        # Step 2：卷积平滑，使相邻 bin 的采样概率更连续（kernel_size > 1 时启用）
        if self.cfg.adaptive_kernel_size > 1:
            f = torch.nn.functional.pad(
                f.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            f = torch.nn.functional.conv1d(f, self.kernel.view(1, 1, -1)).view(-1)

        # Step 3：归一化 cap 后的失败率得到 p_hat，再与均匀分布以 alpha 混合
        f_sum = f.sum()
        alpha = self.cfg.adaptive_uniform_ratio  # alpha：自适应分量的权重
        N = float(self.bin_count)
        if f_sum > 0:
            p_hat = f / f_sum  # 归一化的 cap 失败率分布
            sampling_probabilities = alpha * p_hat + (1.0 - alpha) / N  # 混合均匀分布
        else:
            # 尚无失败数据时退化为纯均匀采样
            sampling_probabilities = torch.ones(self.bin_count, device=self.device) / N

        # 最终归一化保证概率之和为 1
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        # 按概率多项式采样全局 bin
        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        # --- 将全局 bin 反映射回 (clip_index, 本地时间步) ---
        # _bin_offsets 单调递增，用 searchsorted 快速定位每个 bin 属于哪个 clip
        clip_for_bin = torch.searchsorted(self._bin_offsets, sampled_bins, right=True) - 1
        clip_for_bin = torch.clamp(clip_for_bin, 0, self.motion.num_clips - 1)
        # 计算 bin 在该 clip 内的局部索引
        local_bin = sampled_bins - self._bin_offsets[clip_for_bin]
        bins_in_clip = self._bins_per_clip[clip_for_bin]
        clip_len = self.motion.clip_lengths[clip_for_bin]

        # 在 bin 内均匀抖动，得到本地时间步（避免所有 env 都从 bin 的起始帧开始）
        self.clip_indices[env_ids] = clip_for_bin
        local_ts = (
            (local_bin.float() + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / bins_in_clip.float() * (clip_len.float() - 1.0)
        ).long()
        self.time_steps[env_ids] = torch.min(torch.max(local_ts, torch.zeros_like(local_ts)), clip_len - 1)

        # --- 记录 bin 级采样分布指标 ---
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(max(self.bin_count, 2))  # 归一化到 [0, 1]
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm          # bin 分布熵（越小越集中）
        self.metrics["sampling_top1_prob"][:] = pmax          # 最高概率 bin 的值
        self.metrics["sampling_top1_bin"][:] = imax.float() / max(self.bin_count, 1)  # 最难 bin 的相对位置

        # --- 记录 clip 级采样分布指标 ---
        # 将每个 clip 对应的所有 bin 概率求和，得到 clip 的边际采样概率
        clip_probs = torch.zeros(self.motion.num_clips, device=self.device)
        for i in range(self.motion.num_clips):
            s = int(self._bin_offsets[i].item())
            e = s + int(self._bins_per_clip[i].item())
            clip_probs[i] = sampling_probabilities[s:e].sum()
        clip_probs = clip_probs / clip_probs.sum().clamp(min=1e-12)
        H_clip = -(clip_probs * (clip_probs + 1e-12).log()).sum()
        H_clip_norm = H_clip / math.log(max(self.motion.num_clips, 2))  # 归一化到 [0, 1]
        self.metrics["sampling_clip_entropy"][:] = H_clip_norm  # clip 分布熵（越接近 1 越均匀）

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        # 检查哪些 env 的本地时间步已超过当前 clip 的长度，触发重新采样（换到新的 clip 和起始帧）
        clip_len = self.motion.clip_lengths[self.clip_indices]
        env_ids = torch.where(self.time_steps >= clip_len)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

        # 统计当前所有 env 中正在播放的不重复 clip 数量，反映 clip 多样性
        # 值越高说明各 env 在同时体验不同动作片段，训练数据更多样
        active_clips = float(self.clip_indices.unique().shape[0])
        self.metrics["sampling_active_clips"][:] = active_clips

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    motion_files: list[str] = []
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # 卷积平滑窗口大小。对 bin 失败率做滑动平均，使相邻 bin 概率更连续。设为 1 表示不平滑（禁用）
    adaptive_kernel_size: int = 1  
    # EMA 平滑系数，越大越重视历史失败数据。0.8 表示当前失败占 20%，历史失败占 80%（仅 kernel_size > 1 时有效）
    adaptive_lambda: float = 0.8
    # beta：失败率 cap 的倍数，防止极难 bin 被过度采样。调低可降低采样集中度，调高可更专注于难样本。3.0 表示 cap 在平均失败率的 3 倍处。
    adaptive_beta: float = 3.0
    # uniform_ratio：在自适应采样中，保持一定比例的均匀采样，防止过度集中在难样本。0.8 表示 80% 的采样为均匀分布。
    adaptive_uniform_ratio: float = 0.8
    # alpha：自适应采样的学习率，控制失败率更新的速度。0.001 表示较慢的更新。
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
