"""Replay an X2 motion in Isaac Sim.

Examples:
    python scripts/replay_npz.py --motion_file datasets/motions_npz/aiming1_subject1.npz
    python scripts/replay_npz.py --registry_name entity/project/aiming1_subject1:latest
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pathlib

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
motion_source = parser.add_mutually_exclusive_group(required=True)
motion_source.add_argument("--motion_file", type=str, help="Path to a local motion.npz file.")
motion_source.add_argument("--registry_name", type=str, help="W&B registry artifact containing motion.npz.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.x2_ultra import X2_ULTRA_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for an X2 replay scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = X2_ULTRA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _resolve_motion_file() -> str:
    if args_cli.motion_file is not None:
        motion_file = pathlib.Path(args_cli.motion_file).expanduser().resolve()
        if not motion_file.is_file():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        return str(motion_file)

    registry_name = args_cli.registry_name
    if ":" not in registry_name:
        registry_name += ":latest"

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    return str(pathlib.Path(artifact.download()) / "motion.npz")


def _read_motion_fps(motion_file: str) -> float:
    data = np.load(motion_file, allow_pickle=True)
    return float(np.asarray(data["fps"]).reshape(-1)[0])


def _load_motion(motion_file: str, device: str) -> MotionLoader:
    data = np.load(motion_file, allow_pickle=True)
    num_bodies = int(data["body_pos_w"].shape[1])
    body_indexes = torch.arange(num_bodies, dtype=torch.long, device=device)
    return MotionLoader(motion_file, body_indexes, device)


def _align_motion_joint_order_if_needed(robot: Articulation, motion: MotionLoader):
    if motion.joint_names is None:
        return

    robot_joint_names = list(robot.joint_names)
    name_to_index = {name: i for i, name in enumerate(motion.joint_names)}
    missing = [name for name in robot_joint_names if name not in name_to_index]
    if missing:
        raise ValueError(
            "Motion file is missing required joints: "
            + ", ".join(missing)
            + f". Available example: {motion.joint_names[:10]}..."
        )

    reindex = torch.tensor([name_to_index[name] for name in robot_joint_names], device=motion.joint_pos.device)
    motion.joint_pos = motion.joint_pos[:, reindex]
    motion.joint_vel = motion.joint_vel[:, reindex]


def _resolve_root_body_index(motion: MotionLoader) -> int:
    if motion.body_names is None:
        return 0

    name_to_index = {name: i for i, name in enumerate(motion.body_names)}
    for candidate in ("pelvis", "base_link"):
        if candidate in name_to_index:
            return name_to_index[candidate]

    raise ValueError(
        "Could not determine the root body from motion body_names. "
        f"Expected one of ['pelvis', 'base_link'], got example: {motion.body_names[:10]}..."
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion: MotionLoader):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    _align_motion_joint_order_if_needed(robot, motion)
    root_body_index = _resolve_root_body_index(motion)
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        current_steps = time_steps

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[current_steps][:, root_body_index] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[current_steps][:, root_body_index]
        root_states[:, 7:10] = motion.body_lin_vel_w[current_steps][:, root_body_index]
        root_states[:, 10:] = motion.body_ang_vel_w[current_steps][:, root_body_index]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[current_steps], motion.joint_vel[current_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        # pos_lookat = root_states[0, :3].cpu().numpy()
        # sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)
        time_steps = (time_steps + 1) % motion.time_step_total


def main():
    motion_file = _resolve_motion_file()
    motion_fps = _read_motion_fps(motion_file)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / motion_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    motion = _load_motion(motion_file, sim.device)
    # Run the simulator
    run_simulator(sim, scene, motion)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
