import time
import mujoco
import mujoco.viewer
import numpy as np
import threading
import queue
import sys
from digging_system_mesh import MeshDiggingSystem

# ================= 线程函数 =================
def user_input_thread(cmd_queue, stop_event):
    """
    后台线程：监听用户输入
    """
    # 延时一下，避免和主线程的启动日志混在一起
    time.sleep(1.0)
    print("\n------------------------------------------------")
    print("⌨️  交互指令已就绪: 在终端输入 'yes' 并回车以重置墙壁")
    print("------------------------------------------------\n")
    
    while not stop_event.is_set():
        try:
            # 阻塞式等待输入，不会占用 CPU
            user_in = input()
            if user_in.strip().lower() == "yes":
                cmd_queue.put("reset")
                print("-> 收到重置指令，正在执行...")
        except EOFError:
            break

def main():
    xml_path = "../output/merged_result.xml"
    print(f"🚀 正在加载模型: {xml_path}")
    
    # 1. 加载模型
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 初始化挖掘系统
    YOUR_CUTTER_MESH_NAME = "jiegetou_link" 
    YOUR_CUTTER_BODY_NAME = "jiegetou_link" 
    
    print("🔧 初始化挖掘系统...")
    try:
        digging = MeshDiggingSystem(
            model, 
            data, 
            mesh_name=YOUR_CUTTER_MESH_NAME,
            scene_body_name="voxel_target", 
            clean_threshold=[3.0, 0.65, 0.65]
        )
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # --- 线程通信设置 ---
    cmd_queue = queue.Queue()
    stop_event = threading.Event()
    
    # 启动输入监视线程 (Daemon=True 表示主程序退出时它也会自动退出)
    input_t = threading.Thread(target=user_input_thread, args=(cmd_queue, stop_event), daemon=True)
    input_t.start()

    # 3. 启动 Viewer
    print("🎥 启动模拟器...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [1.0, 0, 1.0]
        viewer.cam.distance = 5.0
        
        step_counter = 0
        
        while viewer.is_running():
            step_start = time.time()
            step_counter += 1

            # ================= [功能 2: 处理多线程重置指令] =================
            try:
                # 检查队列是否有消息（非阻塞）
                while not cmd_queue.empty():
                    msg = cmd_queue.get_nowait()
                    if msg == "reset":
                        if hasattr(digging, 'reset'):
                            # 1. 重置挖掘系统状态 (恢复体素)
                            digging.reset()
                            # 2. 重置物理系统状态 (机器人归位)
                            mujoco.mj_resetData(model, data)
                            # 3. 刷新一下模型计算
                            mujoco.mj_forward(model, data)
                            print("✅ 环境已重置！")
                        else:
                            print("⚠️ 错误: MeshDiggingSystem 中未找到 reset() 方法")
            except queue.Empty:
                pass
            # ==========================================================

            # --- 核心逻辑：执行挖掘 ---
            reward = digging.perform_cutting(YOUR_CUTTER_BODY_NAME)
            if reward > 0:
                print(f"⛏️ 挖掘中! 消除体素: {reward}")

            # ================= [功能 1: 实时调用质心获取] =================
            # 注意：确保 digging_system_mesh.py 中有 get_remaining_voxel_center 方法
            current_center = np.zeros(3)
            if hasattr(digging, 'get_remaining_voxel_center'):
                current_center = digging.get_remaining_voxel_center()
                
                # 为了防止终端刷屏太快，每 60 帧打印一次
                if step_counter % 60 == 0:
                    rem_count = len(digging.active_voxels)
                    print(f"[状态监控] 剩余体素: {rem_count} | 质心位置: {current_center}")
            
            # --- 可视化质心 (画一个绿色小球) ---
            viewer.user_scn.ngeom = 0 # 清除上一帧的几何体
            if hasattr(digging, 'active_voxels') and len(digging.active_voxels) > 0:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.1, 0.1, 0.1],      # 球的大小
                    pos=current_center,        # 球的位置 (刚刚获取的质心)
                    mat=np.eye(3).flatten(),
                    rgba=[0.0, 1.0, 0.0, 0.6]  # 绿色，半透明
                )
                viewer.user_scn.ngeom = 1
            # ==========================================================

            # --- 物理步进 ---
            mujoco.mj_step(model, data)
            viewer.sync()

            # 保持实时帧率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # 退出时通知线程结束
    stop_event.set()

if __name__ == "__main__":
    main()