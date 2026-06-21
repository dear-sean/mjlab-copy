import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml


def quat_rotate_inverse(quat, world_vec):
    """
    将世界坐标系的向量旋转到本体坐标系（使用四元数的逆/共轭）。
    quat: [w, x, y, z] - 表示从本体到世界的旋转
    world_vec: 世界坐标系下的向量
    返回: 本体坐标系下的向量
    """
    w, x, y, z = quat
    q_vec = np.array([x, y, z])
    
    t = np.cross(q_vec, world_vec) * 2.0
    return world_vec + w * t + np.cross(q_vec, t)


def projected_gravity(quat):
    """
    将世界重力向量 [0, 0, -1] 投影到本体坐标系。
    注意：这里"投影"实际含义是"旋转到本体坐标系"，不是数学上的平面投影。
    """
    world_gravity = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, world_gravity)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    policy_path = '/home/robot/mjlab-copy/logs/rsl_rl/rlboy_velocity/2026-06-21_22-04-23/model_1500.pt'
    xml_path = '/home/robot/mjlab-copy/src/mjlab/asset_zoo/robots/RL_BOY/RLBOY.xml'

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running():
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # ============================================
                # 观测值构建（根据图中定义，共 7 项）
                # ============================================
                
                # 1. base_lin_vel: 机器人基座线速度（IMU，本体坐标系）
                #    注意：MuJoCo free joint 中 qvel[0:3] 是线速度，qvel[3:6] 是角速度
                #    两者都在世界坐标系下，需要通过四元数逆旋转转换到本体坐标系
                quat = d.qpos[3:7]  # [w, x, y, z]
                world_lin_vel = d.qvel[0:3]  # 世界坐标系线速度
                base_lin_vel = quat_rotate_inverse(quat, world_lin_vel) * cmd_scale  # 本体坐标系线速度
                
                # 2. base_ang_vel: 机器人基座角速度（IMU，本体坐标系）
                #    注意：MuJoCo中角速度也在世界坐标系，需要转换到本体坐标系
                world_ang_vel = d.qvel[3:6]  # 世界坐标系角速度
                base_ang_vel = quat_rotate_inverse(quat, world_ang_vel) * ang_vel_scale  # 本体坐标系角速度
                
                # 3. projected_gravity: 重力向量投影到基座坐标系
                #    注意：这里"投影"实际含义是旋转到本体坐标系，不是数学上的平面投影
                gravity_orientation = projected_gravity(quat)
                
                # 4. joint_pos: 关节位置（相对默认姿态）
                qj = d.qpos[7:]  # 关节位置
                joint_pos = (qj - default_angles) * dof_pos_scale
                
                # 5. joint_vel: 关节速度（相对默认速度，默认速度为0）
                dqj = d.qvel[6:]  # 关节速度
                joint_vel = dqj * dof_vel_scale
                
                # 6. actions: 上一时刻的动作输出
                #    直接使用上一轮迭代中 policy 输出的 action
                
                # 7. command: 当前速度命令（twist）
                #    从配置文件读取的 cmd，通常包含 [vx, vy, yaw_rate] 等指令
                
                # ============================================
                # 组装观测向量（顺序需与训练时一致！）
                # ============================================
                # 观测维度 = 3 + 3 + 3 + num_actions + num_actions + num_actions + 3
                #         = 12 + 3 * num_actions
                
                idx = 0
                obs[idx:idx+3] = base_lin_vel
                idx += 3
                obs[idx:idx+3] = base_ang_vel
                idx += 3
                obs[idx:idx+3] = gravity_orientation
                idx += 3
                obs[idx:idx+num_actions] = joint_pos
                idx += num_actions
                obs[idx:idx+num_actions] = joint_vel
                idx += num_actions
                obs[idx:idx+num_actions] = action
                idx += num_actions
                obs[idx:idx+3] = cmd
                idx += 3
                
                # 备注：请务必确认 obs 的维度与 num_obs 一致，
                # 且顺序与训练策略时使用的观测顺序完全相同！
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)