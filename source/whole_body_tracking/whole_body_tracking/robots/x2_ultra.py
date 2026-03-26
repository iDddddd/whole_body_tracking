import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# Reuse the same PD design heuristics as G1 for a reasonable starting point.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


X2_ULTRA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/x2_ultra/x2_ultra.urdf",
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
        pos=(0.0, 0.0, 0.76),
        # NOTE: IsaacLab validates default joint positions are within URDF limits.
        # The X2 elbow joints have limits [-2.356, 0], so positive defaults will error.
        joint_pos={
            # legs (order matches your provided default_position)
            "left_hip_pitch_joint": -0.2,
            "left_hip_roll_joint": 0.3,
            "left_hip_yaw_joint": 0.5,
            "left_knee_joint": 0.6,
            "left_ankle_pitch_joint": -0.27,
            "left_ankle_roll_joint": -0.13,
            "right_hip_pitch_joint": -0.29,
            "right_hip_roll_joint": -0.15,
            "right_hip_yaw_joint": 0.27,
            "right_knee_joint": 0.65,
            "right_ankle_pitch_joint": -0.4,
            "right_ankle_roll_joint": 0.14,
            # waist
            "waist_yaw_joint": 0.84,
            "waist_pitch_joint": 0.0,
            "waist_roll_joint": 0.0,
            # left arm
            "left_shoulder_pitch_joint": -0.53,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": -0.37,
            "left_elbow_joint": -2.3,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_pitch_joint": -0.26,
            "left_wrist_roll_joint": -0.13,
            # right arm
            "right_shoulder_pitch_joint": -0.64,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.43,
            "right_elbow_joint": -2.0,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_pitch_joint": -0.256,
            "right_wrist_roll_joint": 0.0,
            # head (not provided in your list; keep neutral)
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
        },
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
            armature={
                "left_hip_pitch_joint": ARMATURE_7520_14,
                "left_hip_roll_joint": ARMATURE_7520_22,
                "left_hip_yaw_joint": ARMATURE_7520_14,
                "left_knee_joint": ARMATURE_7520_22,
                "left_ankle_pitch_joint": ARMATURE_5020,
                "left_ankle_roll_joint": ARMATURE_5020,
                "right_hip_pitch_joint": ARMATURE_7520_14,
                "right_hip_roll_joint": ARMATURE_7520_22,
                "right_hip_yaw_joint": ARMATURE_7520_14,
                "right_knee_joint": ARMATURE_7520_22,
                "right_ankle_pitch_joint": ARMATURE_5020,
                "right_ankle_roll_joint": ARMATURE_5020,
                "waist_yaw_joint": ARMATURE_7520_14,
                "waist_pitch_joint": ARMATURE_5020,
                "waist_roll_joint": ARMATURE_5020,
            },
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
            armature={
                "left_shoulder_pitch_joint": ARMATURE_5020,
                "left_shoulder_roll_joint": ARMATURE_5020,
                "left_shoulder_yaw_joint": ARMATURE_5020,
                "left_elbow_joint": ARMATURE_5020,
                "left_wrist_yaw_joint": ARMATURE_5020,
                "left_wrist_pitch_joint": ARMATURE_4010,
                "left_wrist_roll_joint": ARMATURE_4010,
                "right_shoulder_pitch_joint": ARMATURE_5020,
                "right_shoulder_roll_joint": ARMATURE_5020,
                "right_shoulder_yaw_joint": ARMATURE_5020,
                "right_elbow_joint": ARMATURE_5020,
                "right_wrist_yaw_joint": ARMATURE_5020,
                "right_wrist_pitch_joint": ARMATURE_4010,
                "right_wrist_roll_joint": ARMATURE_4010,
                "head_yaw_joint": 0.25 * ARMATURE_4010,
                "head_pitch_joint": 0.25 * ARMATURE_4010,
            },
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
