"""Convert a retargeted motion stored as .pkl into BeyondMimic's motion.npz format.

This repo's tracking task expects a reference motion file with the following keys:
- fps: scalar (int/float)
- joint_pos: (T, num_joints)
- joint_vel: (T, num_joints)
- body_pos_w: (T, num_bodies, 3)
- body_quat_w: (T, num_bodies, 4)  # MuJoCo uses wxyz
- body_lin_vel_w: (T, num_bodies, 3)
- body_ang_vel_w: (T, num_bodies, 3)

This script:
1) Loads a motion dict from a .pkl produced by retargeting (e.g. GMR).
2) Resamples it from input fps to output fps (default 50Hz to match policy dt).
3) Runs MuJoCo forward kinematics to compute per-body pose.
4) Numerically differentiates to get velocities.

Example:
    python scripts/pkl_to_npz_mujoco.py \
      --input_pkl datasets/motions_pkl/x2_ultra/walk1_subject1.pkl \
      --mjcf source/whole_body_tracking/whole_body_tracking/assets/x2_ultra/x2_ultra.xml \
      --output_npz datasets/motions_npz/walk1_subject1.npz \
      --output_fps 50

Then upload:
    python scripts/upload_npz.py --npz_path /tmp/motion.npz --collection_name aiming1_subject_x2
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that maps NumPy 2.x internal module paths to NumPy 1.x.

    Some motion pkls are produced with NumPy 2.x, which records classes from
    `numpy._core.*`. IsaacLab environments often pin NumPy 1.x.

    This remapping allows loading such pkls without upgrading NumPy.
    """

    def find_class(self, module: str, name: str):  # noqa: D401
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def _normalize_quat_wxyz(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, eps)


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in wxyz format."""

    aw, ax, ay, az = np.split(a, 4, axis=-1)
    bw, bx, by, bz = np.split(b, 4, axis=-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.concatenate([w, x, y, z], axis=-1)


def _axis_angle_from_quat_wxyz(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert unit quaternion (wxyz) to axis-angle vector (axis * angle)."""

    q = _normalize_quat_wxyz(q, eps=eps)
    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:]
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)

    angle = 2.0 * np.arctan2(v_norm, np.maximum(w[..., None], eps))
    axis = v / np.maximum(v_norm, eps)

    # When v_norm ~ 0, angle ~ 0 and axis is ill-defined; axis_angle -> 0.
    axis_angle = axis * angle
    axis_angle = np.where(v_norm > 1e-6, axis_angle, 0.0)
    return axis_angle


def _so3_derivative_from_quat_wxyz(rotations: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference SO(3) derivative.

    Args:
        rotations: (T, ..., 4) unit quaternions in wxyz.
        dt: timestep.
    Returns:
        (T, ..., 3) angular velocity in world frame.
    """

    if rotations.shape[0] < 3:
        return np.zeros(rotations.shape[:-1] + (3,), dtype=np.float32)

    q_prev = rotations[:-2]
    q_next = rotations[2:]
    q_rel = _quat_mul_wxyz(q_next, _quat_conj_wxyz(q_prev))

    omega = _axis_angle_from_quat_wxyz(q_rel) / (2.0 * dt)
    omega = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
    return omega.astype(np.float32)


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Slerp between two unit quaternions (wxyz).

    Args:
        q0: (..., 4)
        q1: (..., 4)
        t: (...,) blend factor in [0, 1]
    """

    q0 = _normalize_quat_wxyz(q0)
    q1 = _normalize_quat_wxyz(q1)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    # Take shortest path
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)

    # If very close, fall back to lerp
    close = dot > 1.0 - 1e-4
    q = np.empty_like(q0)

    if np.any(close):
        q_lerp = q0 + (q1 - q0) * t[..., None]
        q_lerp = _normalize_quat_wxyz(q_lerp)
        q = np.where(close, q_lerp, 0.0)

    if np.any(~close):
        theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta0 = np.sin(theta0)
        theta = theta0 * t[..., None]
        s0 = np.sin(theta0 - theta) / np.maximum(sin_theta0, eps)
        s1 = np.sin(theta) / np.maximum(sin_theta0, eps)
        q_slerp = s0 * q0 + s1 * q1
        q_slerp = _normalize_quat_wxyz(q_slerp)
        q = np.where(close, q, q_slerp)

    return q


def _resample_motion(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if math.isclose(input_fps, output_fps):
        return root_pos.astype(np.float32), root_quat_wxyz.astype(np.float32), dof_pos.astype(np.float32)

    input_dt = 1.0 / float(input_fps)
    output_dt = 1.0 / float(output_fps)
    input_frames = root_pos.shape[0]
    duration = (input_frames - 1) * input_dt

    times_out = np.arange(0.0, duration, output_dt, dtype=np.float64)
    phase = times_out / max(duration, 1e-12)

    idx0 = np.floor(phase * (input_frames - 1)).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, input_frames - 1)
    blend = (phase * (input_frames - 1) - idx0).astype(np.float64)

    rp = (1.0 - blend)[:, None] * root_pos[idx0] + blend[:, None] * root_pos[idx1]
    dp = (1.0 - blend)[:, None] * dof_pos[idx0] + blend[:, None] * dof_pos[idx1]
    rq = _slerp_wxyz(root_quat_wxyz[idx0], root_quat_wxyz[idx1], blend.astype(np.float64))

    return rp.astype(np.float32), rq.astype(np.float32), dp.astype(np.float32)


def _ensure_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _auto_quat_format_xyzw_or_wxyz(q: np.ndarray) -> Literal["xyzw", "wxyz"]:
    """Heuristic: if last component dominates, assume xyzw."""

    mean_abs = np.mean(np.abs(q.reshape(-1, 4)), axis=0)
    if mean_abs[3] > 0.5 and mean_abs[3] > mean_abs[0]:
        return "xyzw"
    if mean_abs[0] > 0.5 and mean_abs[0] > mean_abs[3]:
        return "wxyz"
    # fallback: most retargeters in this repo use xyzw
    return "xyzw"


@dataclass
class _MjcfInfo:
    joint_names: list[str]
    body_names: list[str]


def _get_mjcf_info(model: mujoco.MjModel) -> _MjcfInfo:
    joint_names: list[str] = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if j == 0 and name == "floating_base_joint":
            continue
        joint_names.append(name)

    body_names: list[str] = []
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if b == 0 and name == "world":
            continue
        body_names.append(name)

    return _MjcfInfo(joint_names=joint_names, body_names=body_names)


def main():
    parser = argparse.ArgumentParser(description="Convert motion .pkl to BeyondMimic motion.npz using MuJoCo FK.")
    parser.add_argument("--input_pkl", type=str, required=True, help="Path to the input motion .pkl.")
    parser.add_argument(
        "--mjcf",
        type=str,
        default="source/whole_body_tracking/whole_body_tracking/assets/x2_ultra/x2_ultra.xml",
        help="Path to the MuJoCo MJCF (xml) model.",
    )
    parser.add_argument("--output_npz", type=str, required=True, help="Output path for motion.npz.")
    parser.add_argument("--output_fps", type=float, default=50.0, help="Output fps (default: 50).")
    parser.add_argument(
        "--quat_format",
        type=str,
        choices=["auto", "xyzw", "wxyz"],
        default="auto",
        help="Quaternion format of root_rot inside the pkl.",
    )
    args = parser.parse_args()

    motion = _load_pkl(args.input_pkl)
    input_fps = float(motion.get("fps", 30))

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)

    if args.quat_format == "auto":
        quat_format: Literal["xyzw", "wxyz"] = _auto_quat_format_xyzw_or_wxyz(root_rot)
    else:
        quat_format = args.quat_format  # type: ignore[assignment]

    if quat_format == "xyzw":
        root_quat_wxyz = root_rot[:, [3, 0, 1, 2]]
    else:
        root_quat_wxyz = root_rot
    root_quat_wxyz = _normalize_quat_wxyz(root_quat_wxyz)

    # Resample to match policy dt (typically 50Hz).
    root_pos, root_quat_wxyz, dof_pos = _resample_motion(
        root_pos, root_quat_wxyz, dof_pos, input_fps=input_fps, output_fps=float(args.output_fps)
    )

    # Load model and run FK.
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    mjcf_info = _get_mjcf_info(model)

    expected_dofs = model.nq - 7
    if dof_pos.shape[1] != expected_dofs:
        raise ValueError(
            f"DOF mismatch: pkl dof_pos has {dof_pos.shape[1]} dims, but model expects {expected_dofs} (nq={model.nq})."
        )

    T = root_pos.shape[0]
    B = model.nbody - 1  # exclude 'world'
    J = expected_dofs

    body_pos_w = np.zeros((T, B, 3), dtype=np.float32)
    body_quat_w = np.zeros((T, B, 4), dtype=np.float32)

    qpos = np.zeros(model.nq, dtype=np.float64)
    for t in range(T):
        qpos[0:3] = root_pos[t]
        qpos[3:7] = root_quat_wxyz[t]
        qpos[7:] = dof_pos[t]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        body_pos_w[t] = data.xpos[1:].astype(np.float32)
        body_quat_w[t] = _normalize_quat_wxyz(data.xquat[1:].astype(np.float32))

        if (t + 1) % 500 == 0:
            print(f"[INFO] FK {t+1}/{T}")

    dt = 1.0 / float(args.output_fps)

    joint_pos = dof_pos.astype(np.float32)
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)

    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
    body_ang_vel_w = _so3_derivative_from_quat_wxyz(body_quat_w, dt)

    _ensure_dir(args.output_npz)
    np.savez(
        args.output_npz,
        fps=np.array([float(args.output_fps)], dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_names=np.asarray(mjcf_info.joint_names, dtype=object),
        body_names=np.asarray(mjcf_info.body_names, dtype=object),
        source_pkl=os.path.abspath(args.input_pkl),
        mjcf=os.path.abspath(args.mjcf),
        input_fps=np.array([input_fps], dtype=np.float32),
        quat_format=np.asarray([quat_format], dtype=object),
    )

    print(f"[OK] Saved: {args.output_npz}")
    print(f"     frames={T}, fps={args.output_fps}, joints={J}, bodies={B}")


if __name__ == "__main__":
    main()
