from isaaclab.utils import configclass

from whole_body_tracking.robots.x2_ultra import X2_ACTION_SCALE, X2_ULTRA_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class X2FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = X2_ULTRA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = X2_ACTION_SCALE

        # Motion tracking body set should match the motion.npz `body_names`.
        # Use a compact subset for stability and faster convergence.
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            # X2 URDF uses `base_link` as the root body (older motion files may call it `pelvis`).
            "base_link",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
            "head_yaw_link",
        ]
