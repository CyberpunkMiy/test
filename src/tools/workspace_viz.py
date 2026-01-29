import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体（可选，如果乱码可注释掉或换成英文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def main():
    # 1. 加载模型
    xml_path = '../assets/robot.xml'
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 定义关节和末端
    joint_names = ['huizhuan_joint', 'updown_joint', 'front_back_joint']
    site_name = 'cutter_tip'

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    qpos_indices = [model.jnt_qposadr[jid] for jid in joint_ids]
    joint_ranges = [model.jnt_range[jid] for jid in joint_ids]

    # 3. 采样 (为了绘图清晰，采样点设为 15，总点数 15^3 = 3375)
    steps = 15 
    q1_vals = np.linspace(joint_ranges[0][0], joint_ranges[0][1], steps)
    q2_vals = np.linspace(joint_ranges[1][0], joint_ranges[1][1], steps)
    q3_vals = np.linspace(joint_ranges[2][0], joint_ranges[2][1], steps)

    print(f"开始计算运动学... 预计生成 {steps**3} 个点")

    points = []
    original_qpos = data.qpos.copy()

    # 遍历计算
    for q1 in q1_vals:
        for q2 in q2_vals:
            for q3 in q3_vals:
                data.qpos[qpos_indices[0]] = q1
                data.qpos[qpos_indices[1]] = q2
                data.qpos[qpos_indices[2]] = q3
                mujoco.mj_kinematics(model, data)
                points.append(data.site_xpos[site_id].copy())

    data.qpos = original_qpos
    points_np = np.array(points)
    
    xs, ys, zs = points_np[:, 0], points_np[:, 1], points_np[:, 2]
    
    print(f"计算完成。X范围: [{xs.min():.2f}, {xs.max():.2f}]")
    print(f"          Y范围: [{ys.min():.2f}, {ys.max():.2f}]")
    print(f"          Z范围: [{zs.min():.2f}, {zs.max():.2f}]")

    # ================= 绘图部分 =================
    # 创建一个大图，包含 4 个子图
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'掘进机工作空间分析 (Roadheader Workspace Analysis)', fontsize=16)

    # --- 子图 1: 3D 视图 ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    sc = ax1.scatter(xs, ys, zs, c=zs, cmap='viridis', s=5, alpha=0.5)
    ax1.set_xlabel('X (Side/Left-Right)')
    ax1.set_ylabel('Y (Forward)')
    ax1.set_zlabel('Z (Height)')
    ax1.set_title('3D Workspace Point Cloud')
    plt.colorbar(sc, ax=ax1, label='Height Z (m)', shrink=0.5)

    # --- 子图 2: X-Y 平面 (俯视图 - Top View) ---
    # 展示左右摆动和前后伸缩的覆盖面积
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(xs, ys, c=zs, cmap='viridis', s=5, alpha=0.5)
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Top View (X-Y Projection)\nCoverage Area')
    ax2.grid(True)
    ax2.axis('equal')

    # --- 子图 3: X-Z 平面 (正视图 - Front View) ---
    # 展示挖掘断面的形状（最常用）
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(xs, zs, c=ys, cmap='magma', s=5, alpha=0.5)
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Z Position (m)')
    ax3.set_title('Front View (X-Z Projection)\nCutting Cross-section')
    ax3.grid(True)
    ax3.axis('equal')

    # --- 子图 4: Y-Z 平面 (侧视图 - Side View) ---
    # 展示挖掘高度随距离的变化
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(ys, zs, c=xs, cmap='plasma', s=5, alpha=0.5)
    ax4.set_xlabel('Y Position (m)')
    ax4.set_ylabel('Z Position (m)')
    ax4.set_title('Side View (Y-Z Projection)\nHeight vs Reach')
    ax4.grid(True)
    ax4.axis('equal')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()