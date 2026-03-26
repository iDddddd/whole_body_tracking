"""Minimal env smoke test.

This script launches Isaac Sim (headless by default) and tries to construct the env once.
It is useful to validate robot import, joint limits, and motion file compatibility without W&B.

Example:
    /path/to/IsaacLab/isaaclab.sh -p scripts/smoke_env.py --task Tracking-Flat-X2-v0 \
        --motion_file /tmp/motion_x2_aiming1.npz --num_envs 2 --headless
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Smoke test for IsaacLab env creation")
parser.add_argument("--task", type=str, required=True, help="Gym task id")
parser.add_argument("--motion_file", type=str, required=True, help="Path to motion.npz")
parser.add_argument("--num_envs", type=int, default=2, help="Number of envs")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "env_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg=None,
):
    _ = agent_cfg
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = args_cli.motion_file

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # One reset + one step is enough to trigger most import/limit checks.
    obs = env.reset()
    _ = obs
    env.close()


if __name__ == "__main__":
    try:
        cast(Callable[[], None], main)()
    finally:
        simulation_app.close()
