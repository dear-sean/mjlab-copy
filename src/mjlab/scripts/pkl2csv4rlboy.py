import pickle
import csv
import argparse
import numpy as np


def pkl_to_csv_rlboy(pkl_path: str, csv_path: str):
    """
    RLBoy 专用：PKL 运动数据 转 CSV
    仅保留必要字段：基座位置 + 基座四元数(qx,qy,qz,qw) + 20个关节角度
    输出格式完全兼容 csv_to_npz_rlboy.py
    """
    # 1. 加载 pkl 文件
    with open(pkl_path, "rb") as f:
        motion_data = pickle.load(f)

    # 2. 校验数据结构（根据常规运动pkl格式适配）
    # 约定 pkl 内部字段（可根据你的实际pkl键名修改）：
    #   root_pos: 基座位置 [N, 3] (x,y,z)
    #   root_quat: 基座四元数 [N, 4] (qx, qy, qz, qw)
    #   dof_pos: 关节角度 [N, 20] 顺序与RLBoy关节列表一致
    try:
        root_pos = np.array(motion_data["root_pos"])
        root_quat = np.array(motion_data["root_rot"])
        dof_pos = np.array(motion_data["dof_pos"])
    except KeyError as e:
        raise RuntimeError(f"PKL 缺少必要字段: {e}\n"
                           "请检查pkl内是否包含 root_pos / root_quat / dof_pos")

    # 维度校验
    n_frames = root_pos.shape[0]
    assert root_pos.shape == (n_frames, 3), "基座位置维度必须为 [帧数, 3]"
    assert root_quat.shape == (n_frames, 4), "基座四元数维度必须为 [帧数, 4]"
    assert dof_pos.shape == (n_frames, 20), "关节角度必须为 [帧数, 20] (RLBoy 20DoF)"

    print(f"加载完成，总帧数: {n_frames}")
    print(f"开始写入 CSV: {csv_path}")

    # 3. 逐帧拼接数据并写入 CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        for frame_idx in range(n_frames):
            # 拼接一行: pos(3) + quat(4) + dof(20) → 共27列
            row = []
            # 基座位置 x,y,z
            row.extend(root_pos[frame_idx].tolist())
            # 基座四元数 qx, qy, qz, qw (和原脚本要求一致)
            row.extend(root_quat[frame_idx].tolist())
            # 20个关节角度
            row.extend(dof_pos[frame_idx].tolist())

            writer.writerow(row)

    print(f"转换成功！文件已保存至: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="RLBoy PKL 转 CSV 工具")
    parser.add_argument("--pkl", type=str, required=True, help="输入 .pkl 文件路径")
    parser.add_argument("--csv", type=str, required=True, help="输出 .csv 文件路径")
    args = parser.parse_args()

    pkl_to_csv_rlboy(args.pkl, args.csv)


if __name__ == "__main__":
    main()
