import argparse  # 解析命令行参数用于配置运行  
import time  # 提供计时与延时功能  

import mujoco.viewer  # MuJoCo 可视化查看器接口  
import mujoco  # MuJoCo 物理引擎接口  
import numpy as np  # 数值计算与数组处理  
import onnxruntime  # ONNX 运行时推理引擎  
import onnx  # ONNX 模型解析加载  
import torch  # PyTorch 张量与模型工具  
import os  # 文件路径操作  

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # 仓库根目录  
XML_PATH = os.path.join(_REPO_ROOT, "source/whole_body_tracking/whole_body_tracking/assets/x2_ultra/x2_ultra.xml")  # MuJoCo 模型 XML 路径  
SIMULATION_DURATION = 300.0  # 总仿真时长（秒）  
SIMULATION_DT = 0.002  # 物理仿真步长  
CONTROL_DECIMATION = 10  # 控制更新降采样倍数  
NUM_ACTIONS = 31  # 动作维度数量  
NUM_OBS = 170  # 观测维度数量  
BODY_NAME = "torso_link"  # 机器人参考刚体名称  

JOINT_XML = [  # MuJoCo 关节顺序列表（X2, qpos 顺序, 31 关节）  
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
]  # 关节顺序列表结束  


def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b):  # 计算 A 到 B 的相对位姿  
    rotm_a = np.zeros(9)  # 初始化旋转矩阵扁平数组  
    mujoco.mju_quat2Mat(rotm_a, quat_a)  # 四元数转旋转矩阵  
    rotm_a = rotm_a.reshape(3, 3)  # 重塑为 3x3 矩阵  
    rel_pos = rotm_a.T @ (pos_b - pos_a)  # 计算相对位置向量  
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), quat_b)  # 计算相对旋转四元数  
    rel_quat = rel_quat / np.linalg.norm(rel_quat)  # 归一化四元数  
    return rel_pos, rel_quat  # 返回相对位置和相对姿态  

def quaternion_conjugate(q):  # 计算四元数共轭  
    return np.array([q[0], -q[1], -q[2], -q[3]])  # 标量不变向量取负  

def quaternion_multiply(q1, q2):  # 计算四元数乘积  
    w1, x1, y1, z1 = q1  # 解包第一个四元数  
    w2, x2, y2, z2 = q2  # 解包第二个四元数  
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # 计算结果 w 分量  
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # 计算结果 x 分量  
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # 计算结果 y 分量  
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # 计算结果 z 分量  
    
    return np.array([w, x, y, z])  # 返回乘积四元数  


def pd_control(target_q, q, kp, target_dq, dq, kd):  # 计算 PD 控制力矩  
    return (target_q - q) * kp + (target_dq - dq) * kd  # 位置与速度误差加权求和  


def parse_args():  # 解析命令行参数  
    parser = argparse.ArgumentParser()  # 创建参数解析器  
    parser.add_argument("--motion_file", type=str, default=os.path.join(_REPO_ROOT, "artifacts/walk1_subject1_x2:v0/motion.npz"), help="motion npz file")  # 动作文件路径参数  
    parser.add_argument("--policy_path", type=str, default=os.path.join(_REPO_ROOT, "logs/rsl_rl/x2_flat/2026-04-01_20-41-43_walk1_subject1_v1/2026-04-01_20-41-43_walk1_subject1_v1.onnx"), help="onnx policy")  # 策略模型路径参数  
    return parser.parse_args()  # 返回解析后的参数  


def load_policy_metadata(policy_path, joint_xml):  # 读取策略元数据并重排关节  
    model = onnx.load(policy_path)  # 加载 ONNX 模型  
    joint_seq = []  # 初始化关节名称序列  
    joint_pos_array_seq = None  # 初始化默认关节角度序列  
    joint_pos_array = None  # 初始化重排后默认角度  
    stiffness_array = None  # 初始化重排后刚度  
    damping_array = None  # 初始化重排后阻尼  
    action_scale = None  # 初始化动作缩放  
    anchor_body_name = None  # 初始化锚点刚体名称  
    body_names = None  # 初始化刚体名称列表  
    for prop in model.metadata_props:  # 遍历模型元数据属性  
        values = prop.value.split(",")  # 解析属性字符串为列表  
        if prop.key == "joint_names":  # 处理关节名称字段  
            joint_seq = values  # 保存关节名称顺序  
        elif prop.key == "default_joint_pos":  # 处理默认关节角度  
            joint_pos_array_seq = np.array([float(x) for x in values])  # 转为浮点数组  
            joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])  # 按 MuJoCo 顺序重排  
        elif prop.key == "joint_stiffness":  # 处理关节刚度  
            stiffness_array = np.array([float(x) for x in values])  # 转为浮点数组  
            stiffness_array = np.array([stiffness_array[joint_seq.index(joint)] for joint in joint_xml])  # 按 MuJoCo 顺序重排  
        elif prop.key == "joint_damping":  # 处理关节阻尼  
            damping_array = np.array([float(x) for x in values])  # 转为浮点数组  
            damping_array = np.array([damping_array[joint_seq.index(joint)] for joint in joint_xml])  # 按 MuJoCo 顺序重排  
        elif prop.key == "action_scale":  # 处理动作缩放  
            action_scale = np.array([float(x) for x in values])  # 转为浮点数组  
        elif prop.key == "anchor_body_name":  # 处理锚点刚体名称  
            anchor_body_name = prop.value  # 记录锚点刚体名称  
        elif prop.key == "body_names":  # 处理刚体名称列表  
            body_names = values  # 记录刚体名称列表  
        print(f"{prop.key}: {prop.value}")  # 打印元数据方便检查  

    if not joint_seq:  # 检查关节名称是否缺失  
        raise ValueError("ONNX 元数据缺少 joint_names，无法对齐关节顺序")  # 抛出缺失错误  
    if joint_pos_array_seq is None or joint_pos_array is None:  # 检查默认关节角度是否缺失  
        raise ValueError("ONNX 元数据缺少 default_joint_pos，无法初始化关节角度")  # 抛出缺失错误  
    if stiffness_array is None or damping_array is None or action_scale is None:  # 检查控制参数是否缺失  
        raise ValueError("ONNX 元数据缺少关节参数或动作缩放，请检查 joint_stiffness/joint_damping/action_scale")  # 抛出缺失错误  

    return (  # 返回整理后的策略参数  
        joint_seq,  # 关节名称顺序  
        joint_pos_array_seq,  # 默认关节角度原序列  
        joint_pos_array,  # 默认关节角度 MuJoCo 顺序  
        stiffness_array,  # 关节刚度 MuJoCo 顺序  
        damping_array,  # 关节阻尼 MuJoCo 顺序  
        action_scale,  # 动作缩放系数  
        anchor_body_name,  # 锚点刚体名称  
        body_names,  # 刚体名称列表  
    )  # 返回结束  


def load_motion(motion_file):  # 读取动作数据文件  
    motion = np.load(motion_file, allow_pickle=True)  # 加载 npz 数据  
    motion_pos = motion["body_pos_w"]  # 读取所有刚体位置序列  
    motion_quat = motion["body_quat_w"]  # 读取所有刚体姿态四元数序列  
    motion_input_pos = motion["joint_pos"]  # 读取关节位置序列  
    motion_input_vel = motion["joint_vel"]  # 读取关节速度序列  
    # 从 npz body_names 动态查找锚点刚体索引  
    motion_body_index = None
    if "body_names" in motion:
        body_names_list = [str(x) for x in motion["body_names"]]
        if BODY_NAME in body_names_list:
            motion_body_index = body_names_list.index(BODY_NAME)
            print(f"[motion] Found {BODY_NAME} at body index {motion_body_index} in npz body_names")
        else:
            raise ValueError(f"{BODY_NAME} not found in npz body_names: {body_names_list}")
    else:
        print("[motion] npz has no body_names field, falling back to hard-coded index")
    # 查找根刚体（pelvis）索引，用于初始化机器人全局位姿
    root_body_index = None
    if "body_names" in motion:
        if "pelvis" in body_names_list:
            root_body_index = body_names_list.index("pelvis")
    # 读取 npz 关节名称（用于后续重排到策略顺序）  
    motion_joint_names = None
    if "joint_names" in motion:
        motion_joint_names = [str(x) for x in motion["joint_names"]]
    return motion_pos, motion_quat, motion_input_pos, motion_input_vel, motion_body_index, motion_joint_names, root_body_index  


if __name__ == "__main__":  # 主程序入口  
    args = parse_args()  # 解析命令行参数  
    motion_file = args.motion_file  # 获取动作文件路径  
    policy_path = args.policy_path  # 获取策略模型路径  

    motion_pos, motion_quat, motion_input_pos, motion_input_vel, motion_body_index_from_npz, motion_joint_names, root_body_index = load_motion(motion_file)  # 加载动作数据  

    (  # 解包策略元数据  
        joint_seq,  # 关节名称顺序  
        joint_pos_array_seq,  # 默认关节角度原序列  
        joint_pos_array,  # 默认关节角度 MuJoCo 顺序  
        stiffness_array,  # 关节刚度 MuJoCo 顺序  
        damping_array,  # 关节阻尼 MuJoCo 顺序  
        action_scale,  # 动作缩放系数  
        anchor_body_name,  # 锚点刚体名称  
        body_names,  # 刚体名称列表  
    ) = load_policy_metadata(policy_path, JOINT_XML)  # 加载策略元数据
    print(f"Loaded policy metadata with {len(joint_seq)} joints and body names: {body_names}")  # 打印加载信息

    # 将 npz 关节数据从 npz 顺序（MuJoCo DFS）重排到策略顺序（IsaacLab BFS）  
    # 训练时 MotionCommand._align_motion_joint_order_if_needed() 做了同样的事  
    if motion_joint_names is not None:
        npz_to_policy = [motion_joint_names.index(j) for j in joint_seq]
        motion_input_pos = motion_input_pos[:, npz_to_policy]
        motion_input_vel = motion_input_vel[:, npz_to_policy]
        print(f"[motion] Reordered motion joints from npz order to policy order ({len(npz_to_policy)} joints)")
    else:
        print("[motion] WARNING: npz has no joint_names, assuming order matches policy")

    obs = np.zeros(NUM_OBS, dtype=np.float32)  # 初始化观测向量  
    counter = 0  # 控制器计数器  

    m = mujoco.MjModel.from_xml_path(XML_PATH)  # 加载 MuJoCo 模型  
    d = mujoco.MjData(m)  # 创建 MuJoCo 数据对象  
    m.opt.timestep = SIMULATION_DT  # 设置仿真时间步长  

    # 覆写动力学参数以匹配训练侧（XML 默认 armature=0.03, frictionloss=0.3）  
    m.dof_armature[6:] = 0.01  # 训练侧 armature  
    m.dof_frictionloss[6:] = 0.0  # 训练侧 frictionloss  
    print(f"[override] dof_armature[6:]={m.dof_armature[6]}, dof_frictionloss[6:]={m.dof_frictionloss[6]}")  

    # 覆写腕部力矩上限以匹配训练侧 effort_limit_sim（XML 中 wrist pitch/roll 为 ±2.2）
    wrist_effort_limits = {
        "left_wrist_pitch_joint": 4.8,
        "left_wrist_roll_joint": 4.8,
        "right_wrist_pitch_joint": 4.8,
        "right_wrist_roll_joint": 4.8,
    }
    for act_id in range(m.nu):
        joint_id = m.actuator_trnid[act_id, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name in wrist_effort_limits:
            lim = wrist_effort_limits[joint_name]
            m.actuator_ctrlrange[act_id, 0] = -lim
            m.actuator_ctrlrange[act_id, 1] = lim
    print("[override] wrist pitch/roll ctrlrange -> +/-4.8 Nm")

    # 使用ONNX Runtime库创建了一个推理会话，用于加载和执行预训练的神经网络策略模型
    policy = onnxruntime.InferenceSession(policy_path)  

    timestep = 0  # 初始化动作序列索引  

    # --- 初始化：teleport 到 motion 参考第一帧（与训练侧 _resample_command 行为对齐） ---  
    # motion_input_pos 已是策略顺序，需转为 MuJoCo qpos 顺序写入 d.qpos  
    init_joint_pos_policy = motion_input_pos[0, :]  # 第一帧关节位置（策略顺序）  
    init_joint_pos_xml = np.array([init_joint_pos_policy[joint_seq.index(joint)] for joint in JOINT_XML])  # 转 MuJoCo 顺序  
    init_joint_vel_policy = motion_input_vel[0, :]  # 第一帧关节速度（策略顺序）
    init_joint_vel_xml = np.array([init_joint_vel_policy[joint_seq.index(joint)] for joint in JOINT_XML])  # 转 MuJoCo 顺序
    # 初始化根刚体位姿：从 motion 参考的第一帧 pelvis 位置/朝向
    # 训练侧 _resample_command 会 teleport root 到 motion 参考位姿
    if root_body_index is not None:
        d.qpos[0:3] = motion_pos[0, root_body_index, :]  # pelvis 世界位置
        d.qpos[3:7] = motion_quat[0, root_body_index, :]  # pelvis 世界朝向
        print(f"[init] Root pose from motion: pos={d.qpos[0:3]}, quat={d.qpos[3:7]}")
    else:
        d.qpos[2] = 0.7  # fallback: 仅设高度
    d.qpos[7:] = init_joint_pos_xml  # 写入 motion 参考第一帧关节角度  
    d.qvel[6 : 6 + NUM_ACTIONS] = init_joint_vel_xml  # 写入 motion 参考第一帧关节速度
    mujoco.mj_forward(m, d)  # 刷新派生状态，确保观测使用一致初始状态
    target_dof_pos = init_joint_pos_xml.copy()  # PD 目标也设为参考姿态  
    # 初始化 action_buffer 使其对应当前姿态，避免 INIT→TRACK 跳变  
    action_buffer = ((init_joint_pos_policy - joint_pos_array_seq) / np.where(action_scale != 0, action_scale, 1.0)).astype(np.float32)  
    print(f"[init] Teleported to motion frame 0, action_buffer range: [{action_buffer.min():.3f}, {action_buffer.max():.3f}]")  

    body_name = anchor_body_name or BODY_NAME  # 使用元数据锚点名称或默认名称  
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)  # 获取锚点 ID  
    if body_id == -1:  # 检查锚点是否存在  
        raise ValueError(f"Body {body_name} not found in model")  # 抛出错误提示  
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")  # root body ID（用于 base_lin_vel）  
    if motion_body_index_from_npz is not None:  # 优先使用 npz 动态查找的索引  
        motion_body_index = motion_body_index_from_npz
    else:
        raise ValueError("npz 中没有 body_names，无法确定锚点索引，请检查 motion 文件")

    with mujoco.viewer.launch_passive(m, d) as viewer:  # 启动被动可视化窗口  
        
        viewer.cam.type = 1  # mjCAMERA_TRACKING  # 跟踪相机
        viewer.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")  # 跟踪pelvis身体
        viewer.cam.lookat[2] = 1.5  # 跟踪点高度
        viewer.cam.distance = 3  # 距离
        viewer.cam.azimuth = 135  # 角度
        viewer.cam.elevation = -20  # 仰角

    
        start = time.time()  # 记录仿真起始时间  
        while viewer.is_running() and time.time() - start < SIMULATION_DURATION:  # 主循环条件  
            step_start = time.time()  # 记录本步起始时间  

            # ---- 策略更新（在 mj_step 之前，与训练侧一致） ----  
            if counter % CONTROL_DECIMATION == 0:  # 到达控制周期更新策略  
                position = d.xpos[body_id]  # 获取仿真中锚点刚体位置  
                quaternion = d.xquat[body_id]  # 获取仿真中锚点刚体姿态  
                motion_input = np.concatenate(  # 拼接目标关节位置与速度  
                    (motion_input_pos[timestep, :], motion_input_vel[timestep, :]),  # 使用策略关节顺序  
                    axis=0,  # 进行拼接  
                )  # 拼接结束  
                motion_pos_current = motion_pos[timestep, motion_body_index, :]  # 读取当前参考锚点动作位置  
                motion_quat_current = motion_quat[timestep, motion_body_index, :]  # 读取当前参考锚点动作姿态  
                anchor_pos, anchor_quat = subtract_frame_transforms_mujoco(  # 计算相对锚点位置与姿态  
                    position, quaternion, motion_pos_current, motion_quat_current  # 输入位姿参数  
                )  # 获取相对位置与旋转结果  
                anchor_ori = np.zeros(9)  # 初始化锚点旋转矩阵  
                mujoco.mju_quat2Mat(anchor_ori, anchor_quat)  # 仿真锚点相对于参考锚点的相对姿态四元数转旋转矩阵  
                anchor_ori = anchor_ori.reshape(3, 3)[:, :2]  # 取旋转矩阵前两列  
                anchor_ori = anchor_ori.reshape(-1,)  # 展平为向量  
                # base_lin_vel 需要用 root body（pelvis）旋转，而非 anchor body（torso_link）  
                # 训练侧 mdp.base_lin_vel = robot.data.root_lin_vel_b（pelvis 坐标系）  
                pelvis_rot = np.zeros(9)  
                mujoco.mju_quat2Mat(pelvis_rot, d.xquat[pelvis_id])  # pelvis 四元数转旋转矩阵  
                pelvis_rot = pelvis_rot.reshape(3, 3)  
                base_lin_vel = pelvis_rot.T @ d.qvel[0:3]  # 世界系线速度 → pelvis 本体系  
                base_ang_vel = d.qvel[3:6]  # MuJoCo free joint 角速度已在 body 本体系，无需旋转  

                offset = 0  # 观测向量写入偏移量  
                obs[offset : offset + 2 * NUM_ACTIONS] = motion_input  # 写入目标关节指令  
                offset += 2 * NUM_ACTIONS  # 更新偏移量  
                obs[offset : offset + 3] = anchor_pos  # 写入参考锚点相对位置  
                offset += 3  # 更新偏移量  
                obs[offset : offset + 6] = anchor_ori  # 写入参考锚点相对姿态
                offset += 6  # 更新偏移量  

                obs[offset : offset + 3] = base_lin_vel  # 写入基座root/freejoint线速度  
                offset += 3  # 更新偏移量  
                obs[offset : offset + 3] = base_ang_vel  # 写入基座角速度  
                offset += 3  # 更新偏移量  
                qpos_xml = d.qpos[7 : 7 + NUM_ACTIONS]  # 读取 MuJoCo 顺序关节角度  
                qpos_seq = np.array([qpos_xml[JOINT_XML.index(joint)] for joint in joint_seq])  # 变换到策略关节顺序  
                obs[offset : offset + NUM_ACTIONS] = qpos_seq - joint_pos_array_seq  # 写入关节角度偏差  
                offset += NUM_ACTIONS  # 更新偏移量  
                qvel_xml = d.qvel[6 : 6 + NUM_ACTIONS]  # 读取 MuJoCo 顺序关节速度  
                qvel_seq = np.array([qvel_xml[JOINT_XML.index(joint)] for joint in joint_seq])  # 变换到策略关节顺序  
                obs[offset : offset + NUM_ACTIONS] = qvel_seq  # 写入关节速度  
                offset += NUM_ACTIONS  # 更新偏移量  
                obs[offset : offset + NUM_ACTIONS] = action_buffer  # 写入上一动作  

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # 转为批量张量输入  
                action = policy.run(  # 执行 ONNX 推理  
                    ["actions"],  # 指定输出节点名称  
                    {"obs": obs_tensor.numpy(), "time_step": np.array([timestep], dtype=np.float32).reshape(1, 1)},  # 传入观测与时间步  
                )[0]  # 获取推理输出  
                action = np.asarray(action).reshape(-1)  # 转为一维数组  
                action_buffer = action.copy()  # 更新上一动作缓存  
                target_dof_pos = action * action_scale + joint_pos_array_seq  # 计算目标关节角度  
                target_dof_pos = target_dof_pos.reshape(-1,)  # 确保为一维向量  
                target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in JOINT_XML])  # 重排到 MuJoCo 顺序  
                if timestep < 5:  # 前 5 个策略周期打印调试信息  
                    print(f"[step {timestep}] base_lin_vel={base_lin_vel}, base_ang_vel={base_ang_vel}")
                    print(f"[step {timestep}] anchor_pos={anchor_pos}, anchor_ori_norm={np.linalg.norm(anchor_ori):.3f}")
                    print(f"[step {timestep}] action min={action.min():.3f} max={action.max():.3f} mean={action.mean():.3f}")
                    shoulder_indices = [joint_seq.index(j) for j in ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]]
                    print(f"[step {timestep}] shoulder_pitch action: L={action[shoulder_indices[0]]:.3f}, R={action[shoulder_indices[1]]:.3f}")
                timestep += 1  # 更新动作序列索引  

            # ---- PD 控制 + 物理仿真（每个 sim step 都执行） ----  
            tau = pd_control(  # 计算关节控制力矩  
                target_dof_pos,  # 目标关节角度  
                d.qpos[7:],  # freejoint 后面的所有关节位置  
                stiffness_array,  # 关节刚度  
                np.zeros_like(damping_array),  # 目标关节速度置零  
                d.qvel[6:],  # freejoint 后面的所有关节角速度  
                damping_array,  # 关节阻尼  
            )  # 控制力矩计算结束  
            d.ctrl[:] = tau  # 写入力矩控制输入  
            mujoco.mj_step(m, d)  # 进行一步物理仿真  
            counter += 1  # 更新控制器计数  

            viewer.sync()  # 同步可视化显示  
            time_until_next_step = m.opt.timestep - (time.time() - step_start)  # 计算剩余睡眠时间  
            if time_until_next_step > 0:  # 若剩余时间为正则休眠  
                time.sleep(time_until_next_step)  # 休眠保持实时步长    
