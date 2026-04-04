import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

X2_ULTRA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/x2_ultra/x2_ultra_simple_collision.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Match the MuJoCo MJCF default pose in `assets/x2_ultra/x2_ultra.xml`.
        # - Root height in MJCF: body pelvis pos="0 0 0.68"
        # - Joint defaults: MJCF has no explicit `qpos` defaults, so they are zero.
        pos=(0.0, 0.0, 0.68),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Lower body + waist (kp/kd come from your provided arrays)
        "lower_body": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_pitch_joint",
                "waist_roll_joint",
            ],
            effort_limit_sim={
                # legs
                "left_hip_pitch_joint": 120.0,
                "left_hip_roll_joint": 120.0,
                "left_hip_yaw_joint": 120.0,
                "left_knee_joint": 120.0,
                "left_ankle_pitch_joint": 36.0,
                "left_ankle_roll_joint": 24.0,
                "right_hip_pitch_joint": 120.0,
                "right_hip_roll_joint": 120.0,
                "right_hip_yaw_joint": 120.0,
                "right_knee_joint": 120.0,
                "right_ankle_pitch_joint": 36.0,
                "right_ankle_roll_joint": 24.0,
                # waist
                "waist_yaw_joint": 120.0,
                "waist_pitch_joint": 48.0,
                "waist_roll_joint": 48.0,
            },
            velocity_limit_sim=50.0,
            stiffness={
                # kp
                "left_hip_pitch_joint": 120.0,
                "left_hip_roll_joint": 120.0,
                "left_hip_yaw_joint": 120.0,
                "left_knee_joint": 150.0,
                "left_ankle_pitch_joint": 40.0,
                "left_ankle_roll_joint": 30.0,
                "right_hip_pitch_joint": 120.0,
                "right_hip_roll_joint": 120.0,
                "right_hip_yaw_joint": 120.0,
                "right_knee_joint": 150.0,
                "right_ankle_pitch_joint": 40.0,
                "right_ankle_roll_joint": 30.0,
                "waist_yaw_joint": 120.0,
                "waist_pitch_joint": 40.0,
                "waist_roll_joint": 40.0,
            },
            damping={
                # kd
                "left_hip_pitch_joint": 5.0,
                "left_hip_roll_joint": 5.0,
                "left_hip_yaw_joint": 5.0,
                "left_knee_joint": 5.0,
                "left_ankle_pitch_joint": 2.0,
                "left_ankle_roll_joint": 2.0,
                "right_hip_pitch_joint": 5.0,
                "right_hip_roll_joint": 5.0,
                "right_hip_yaw_joint": 5.0,
                "right_knee_joint": 5.0,
                "right_ankle_pitch_joint": 2.0,
                "right_ankle_roll_joint": 2.0,
                "waist_yaw_joint": 5.0,
                "waist_pitch_joint": 3.0,
                "waist_roll_joint": 3.0,
            },
            armature=0.03,
        ),
        # Arms + head
        "upper_body": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_yaw_joint",
                "left_wrist_pitch_joint",
                "left_wrist_roll_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_yaw_joint",
                "right_wrist_pitch_joint",
                "right_wrist_roll_joint",
                "head_yaw_joint",
                "head_pitch_joint",
            ],
            effort_limit_sim={
                "left_shoulder_pitch_joint": 36.0,
                "left_shoulder_roll_joint": 36.0,
                "left_shoulder_yaw_joint": 24.0,
                "left_elbow_joint": 24.0,
                "left_wrist_yaw_joint": 24.0,
                "left_wrist_pitch_joint": 4.8,
                "left_wrist_roll_joint": 4.8,
                "right_shoulder_pitch_joint": 36.0,
                "right_shoulder_roll_joint": 36.0,
                "right_shoulder_yaw_joint": 24.0,
                "right_elbow_joint": 24.0,
                "right_wrist_yaw_joint": 24.0,
                "right_wrist_pitch_joint": 4.8,
                "right_wrist_roll_joint": 4.8,
                "head_yaw_joint": 2.6,
                "head_pitch_joint": 0.6,
            },
            velocity_limit_sim=50.0,
            stiffness={
                "left_shoulder_pitch_joint": 20.0,
                "left_shoulder_roll_joint": 20.0,
                "left_shoulder_yaw_joint": 20.0,
                "left_elbow_joint": 20.0,
                "left_wrist_yaw_joint": 20.0,
                "left_wrist_pitch_joint": 5.0,
                "left_wrist_roll_joint": 5.0,
                "right_shoulder_pitch_joint": 20.0,
                "right_shoulder_roll_joint": 20.0,
                "right_shoulder_yaw_joint": 20.0,
                "right_elbow_joint": 20.0,
                "right_wrist_yaw_joint": 20.0,
                "right_wrist_pitch_joint": 5.0,
                "right_wrist_roll_joint": 5.0,
                # head: mild stabilization
                "head_yaw_joint": 5.0,
                "head_pitch_joint": 5.0,
            },
            damping={
                "left_shoulder_pitch_joint": 1.0,
                "left_shoulder_roll_joint": 1.0,
                "left_shoulder_yaw_joint": 1.0,
                "left_elbow_joint": 1.0,
                "left_wrist_yaw_joint": 1.0,
                "left_wrist_pitch_joint": 0.5,
                "left_wrist_roll_joint": 0.5,
                "right_shoulder_pitch_joint": 1.0,
                "right_shoulder_roll_joint": 1.0,
                "right_shoulder_yaw_joint": 1.0,
                "right_elbow_joint": 1.0,
                "right_wrist_yaw_joint": 1.0,
                "right_wrist_pitch_joint": 0.5,
                "right_wrist_roll_joint": 0.5,
                "head_yaw_joint": 0.5,
                "head_pitch_joint": 0.5,
            },
            armature=0.03,
        ),
    },
)


X2_ACTION_SCALE: dict[str, float] = {}
for actuator in X2_ULTRA_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    names = actuator.joint_names_expr
    if not isinstance(effort, dict):
        effort = {n: effort for n in names}
    if not isinstance(stiffness, dict):
        stiffness = {n: stiffness for n in names}
    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            X2_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
