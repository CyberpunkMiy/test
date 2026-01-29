import time
import mujoco
import mujoco.viewer
import numpy as np
import threading
import queue
import sys
import matplotlib.pyplot as plt

# ================= 配置参数 =================
XML_PATH = "../assets/robot.xml"
# XML_PATH = "../output/merged_result.xml"
SITE_NAME = "cutter_tip"  # 末端执行器 Site 名称
# 参与控制的关节名称列表 (顺序需要对应)
JOINT_NAMES = ["huizhuan_joint", "updown_joint", "front_back_joint"]
ACTUATOR_NAMES = ["act_left_right", "act_up_down", "act_front_back"]

MOVEMENT_DURATION = 3.0  # 每次运动的耗时 (秒)
CONTROL_FREQ = 50.0      # 控制频率 (Hz), 即每秒做多少次 IK 计算
CONTROL_DT = 1.0 / CONTROL_FREQ

def plot_trajectory(points):
    """绘制 3D 轨迹及其在三个平面上的投影"""
    if len(points) < 2:
        return
    points = np.array(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('End-Effector Trajectory Analysis')
    
    # 3D View
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(x, y, z, 'b-', linewidth=2, label='Path')
    ax3d.scatter(x[0], y[0], z[0], c='g', marker='o', s=50, label='Start')
    ax3d.scatter(x[-1], y[-1], z[-1], c='r', marker='x', s=50, label='End')
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory')
    ax3d.legend()
    
    # XY Plane
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(x, y, 'b-')
    ax_xy.plot(x[0], y[0], 'go', label='Start')
    ax_xy.plot(x[-1], y[-1], 'rx', label='End')
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('XY Plane Projection')
    ax_xy.grid(True)
    ax_xy.axis('equal')

    # XZ Plane
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.plot(x, z, 'b-')
    ax_xz.plot(x[0], z[0], 'go')
    ax_xz.plot(x[-1], z[-1], 'rx')
    ax_xz.set_xlabel('X (m)')
    ax_xz.set_ylabel('Z (m)')
    ax_xz.set_title('XZ Plane Projection')
    ax_xz.grid(True)
    ax_xz.axis('equal')

    # YZ Plane
    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.plot(y, z, 'b-')
    ax_yz.plot(y[0], z[0], 'go')
    ax_yz.plot(y[-1], z[-1], 'rx')
    ax_yz.set_xlabel('Y (m)')
    ax_yz.set_ylabel('Z (m)')
    ax_yz.set_title('YZ Plane Projection')
    ax_yz.grid(True)
    ax_yz.axis('equal')
    
    plt.tight_layout()
    print(">> 正在显示轨迹图，请关闭图表窗口以继续...")
    plt.show()

class RoadheaderController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # 获取关节和执行器的索引
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self.dof_ids = [model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES]
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
        
        # 验证是否找到所有组件
        if -1 in self.joint_ids or -1 in self.actuator_ids or self.site_id == -1:
            raise ValueError("无法在模型中找到指定的关节、执行器或 Site，请检查名称。")

    def get_current_site_pos(self):
        """获取当前末端的三维坐标"""
        return self.data.site_xpos[self.site_id].copy()

    def get_joint_positions(self):
        """获取当前受控关节的角度/位置"""
        return np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.joint_ids])

    def solve_ik(self, target_pos, init_q=None, max_steps=50, tol=1e-3):
        """
        计算逆运动学。
        :param target_pos: 目标三维坐标 (x, y, z)
        :param init_q: 初始关节角度猜测值 (如果为 None，则使用当前实际角度)
        :return: 对应的关节角度数组，若未收敛则返回 None
        """
        ik_data = mujoco.MjData(self.model)
        
        # 同步当前状态到 ik_data
        ik_data.qpos[:] = self.data.qpos[:]
        mujoco.mj_fwdPosition(self.model, ik_data)

        # 如果提供了特定的初始猜测
        if init_q is not None:
             for i, jid in enumerate(self.joint_ids):
                q_adr = self.model.jnt_qposadr[jid]
                ik_data.qpos[q_adr] = init_q[i]
        
        step_size = 0.5 # 阻尼系数/步长
        
        for _ in range(max_steps):
            # 1. 计算正运动学，获取当前末端位置
            mujoco.mj_fwdPosition(self.model, ik_data)
            current_pos = ik_data.site_xpos[self.site_id]
            
            # 2. 计算误差
            error = target_pos - current_pos
            if np.linalg.norm(error) < tol:
                res_q = []
                for jid in self.joint_ids:
                    q_adr = self.model.jnt_qposadr[jid]
                    res_q.append(ik_data.qpos[q_adr])
                return np.array(res_q)

            # 3. 计算雅可比矩阵
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, ik_data, jacp, jacr, self.site_id)
            
            # 4. 提取我们控制的那 3 个自由度对应的列
            J_reduced = np.zeros((3, 3))
            for i, dof_idx in enumerate(self.dof_ids):
                J_reduced[:, i] = jacp[:, dof_idx]
                
            # 5. 求解 dq
            dq = np.linalg.pinv(J_reduced) @ error
            
            # 6. 更新关节角度
            for i, jid in enumerate(self.joint_ids):
                q_adr = self.model.jnt_qposadr[jid]
                ik_data.qpos[q_adr] += dq[i] * step_size
                
                # 检查并应用关节限制
                range_min = self.model.jnt_range[jid][0]
                range_max = self.model.jnt_range[jid][1]
                if ik_data.qpos[q_adr] < range_min: ik_data.qpos[q_adr] = range_min
                if ik_data.qpos[q_adr] > range_max: ik_data.qpos[q_adr] = range_max

        res_q = []
        for jid in self.joint_ids:
            q_adr = self.model.jnt_qposadr[jid]
            res_q.append(ik_data.qpos[q_adr])
        return np.array(res_q)

    def control_actuators(self, target_q):
        """将目标关节角度发送给执行器"""
        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = target_q[i]

def user_input_thread(input_queue, stop_event, idle_event):
    """单独的线程用于接收用户输入"""
    # 稍作延时，等待主程序打印初始化信息
    time.sleep(1.0)
    print("\n==========================================")
    print(">>> 终端控制模式就绪 <<<")
    print("请输入目标坐标 (x y z)，例如: 2.5 0.0 1.2")
    print("输入 'q' 或 'exit' 退出程序")
    print("==========================================\n")
    
    while not stop_event.is_set():
        # 等待机器人处于空闲状态
        idle_event.wait()
        
        if stop_event.is_set():
            break

        try:
            # 获取用户输入
            user_str = input("请输入目标 (x y z) >> ")
            
            if stop_event.is_set():
                break
            
            if user_str.strip().lower() in ['q', 'exit']:
                input_queue.put('exit')
                break
            
            parts = user_str.strip().split()
            if len(parts) == 3:
                try:
                    target = np.array([float(p) for p in parts])
                    input_queue.put(target)
                    # 输入有效后，清除空闲标志，等待机器人执行完毕
                    idle_event.clear()
                except ValueError:
                    print("错误: 坐标必须是有效的数字。")
            else:
                if user_str.strip(): # 忽略空回车
                    print("错误: 请输入 3 个数值，用空格分隔。")
                
        except (EOFError, KeyboardInterrupt):
            input_queue.put('exit')
            break

def main():
    # 1. 加载模型
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 初始化控制器
    controller = RoadheaderController(model, data)
    
    # 初始化仿真
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # --- 线程通信设置 ---
    input_queue = queue.Queue()
    stop_event = threading.Event()
    idle_event = threading.Event()
    idle_event.set() # 初始状态为空闲
    
    # 启动输入监视线程
    input_t = threading.Thread(target=user_input_thread, args=(input_queue, stop_event, idle_event), daemon=True)
    input_t.start()

    # 轨迹相关初始变量
    start_pos = controller.get_current_site_pos()
    target_pos_global = start_pos.copy() # 默认呆在原地
    
    trajectory_start_time = 0.0
    is_moving = False
    current_trajectory = [] # 用于存储轨迹点
    
    print(f"仿真已启动。当前末端位置: {start_pos}")

    # 使用 with 语句启动 Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # 初始等待物理稳定
        start_wait = time.time()
        while viewer.is_running() and time.time() - start_wait < 1.0:
            mujoco.mj_step(model, data)
            viewer.sync()
        
        # 主循环
        last_control_time = 0
        
        while viewer.is_running():
            step_start = time.time()

            # --- 1. 处理用户输入 ---
            try:
                while not input_queue.empty():
                    msg = input_queue.get_nowait()
                    if isinstance(msg, str) and msg == 'exit':
                        print("退出程序...")
                        stop_event.set()
                        viewer.close()
                        sys.exit(0)
                    elif isinstance(msg, np.ndarray):
                        # 收到新目标点
                        target_pos_global = msg
                        start_pos = controller.get_current_site_pos()
                        
                        # 重置轨迹记录
                        current_trajectory = [start_pos]
                        
                        trajectory_start_time = data.time
                        is_moving = True
                        print(f"-> 收到新目标: {target_pos_global}，开始运动...")
            except queue.Empty:
                pass

            # --- 2. 控制逻辑 (固定频率) ---
            if data.time - last_control_time >= CONTROL_DT:
                last_control_time = data.time
                
                # 计算当前时刻的期望笛卡尔位置
                current_desired_pos = target_pos_global
                
                if is_moving:
                    # 记录实际轨迹（物理位置）
                    current_trajectory.append(controller.get_current_site_pos())
                    
                    # 计算进度
                    elapsed = data.time - trajectory_start_time
                    progress = elapsed / MOVEMENT_DURATION
                    progress = min(max(progress, 0.0), 1.0)
                    
                    # 直线插值 P(t)
                    current_desired_pos = start_pos + progress * (target_pos_global - start_pos)
                    
                    # 检查是否完成
                    if progress >= 1.0:
                        print(f"-> 已到达目标点: {target_pos_global}")
                        is_moving = False
                        
                        # 绘制轨迹
                        plot_trajectory(current_trajectory)
                        
                        # 任务完成，标记为空闲，允许输入下一个点
                        idle_event.set()
                
                # 执行 IK 并控制执行器
                # 无论是运动中还是静止，都维持在 current_desired_pos
                q_cmd = controller.solve_ik(current_desired_pos, tol=1e-3)
                controller.control_actuators(q_cmd)

            # --- 3. 物理步进 ---
            mujoco.mj_step(model, data)

            # --- 4. 渲染与交互 ---
            viewer.user_scn.ngeom = 0 
            # 渲染目标点 红球
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0.05, 0.05], 
                pos=target_pos_global,
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.0, 0.0, 0.5] 
            )
            viewer.user_scn.ngeom = 1
            viewer.sync()

            # 帧率控制
            time_until_next_step = model.opt.timestep
            if (time.time() - step_start) < time_until_next_step:
                time.sleep(time_until_next_step)
    
    # 确保退出时停止线程
    stop_event.set()

if __name__ == "__main__":
    main()
