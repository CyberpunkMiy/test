import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
import mink

# === 配置参数 ===
XML_PATH = 'robot.xml'
SITE_NAME = 'cutter_tip'
JOINT_NAMES = ['huizhuan_joint', 'updown_joint', 'front_back_joint']

# 全局变量
current_target = np.array([0.0, 1.5, 0.5]) 
input_running = True

def user_input_thread():
    global current_target, input_running
    print(">>> 调试模式启动")
    while input_running:
        try:
            user_str = input()
            if user_str.strip().lower() == 'q': break
            parts = user_str.strip().split()
            if len(parts) == 3:
                current_target = np.array(list(map(float, parts)))
                print(f"--> 新目标: {current_target}")
        except: pass
    input_running = False

def main():
    global input_running
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)
    
    # 任务定义
    end_effector_task = mink.FrameTask(
        frame_name=SITE_NAME,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )

    t = threading.Thread(target=user_input_thread, daemon=True)
    t.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 5
        viewer.cam.azimuth = 140
        dt = 0.01 
        solver = "quadprog" 

        print("=== 仿真开始 ===")
        print(f"当前使用的求解器: {solver}")
        print("如果下方报错 ModuleNotFoundError，请 pip install quadprog")

        while viewer.is_running():
            step_start = time.time()
            
            # 更新 Kinematics
            configuration.update(data.qpos)
            
            # 计算目标
            current_ee_pos = configuration.get_transform_frame_to_world(SITE_NAME, "site").translation()
            error_vec = current_target - current_ee_pos
            dist = np.linalg.norm(error_vec)
            
            # 简单的轨迹插值
            max_speed = 0.5 
            step_dist = max_speed * dt
            if dist > step_dist:
                intermediate_target = current_ee_pos + (error_vec / dist) * step_dist
            else:
                intermediate_target = current_target
            
            end_effector_task.set_target(mink.SE3.from_translation(intermediate_target))
            
            # === 核心调试点：移除 try-catch ===
            # 计算速度
            vel = mink.solve_ik(
                configuration, 
                [end_effector_task], 
                dt, 
                solver, 
                damping=1e-2 # 稍微增加阻尼以提高数值稳定性
            )
            
            # 打印调试信息 (如果速度全是0，说明计算有问题)
            # if np.linalg.norm(vel) < 1e-6 and dist > 0.01:
            #     print(f"警告: 距离目标 {dist:.3f}m，但计算出的速度为 0！可能遇到关节限制或求解失败。")

            # 积分并应用
            configuration.integrate_inplace(vel, dt)
            
            for i, name in enumerate(JOINT_NAMES):
                jid = model.joint(name).id
                # 查找 Actuator ID
                act_id = -1
                for aid in range(model.nu):
                    if model.actuator_trnid[aid, 0] == jid:
                        act_id = aid
                        break
                
                if act_id != -1:
                    q_idx = model.jnt_qposadr[jid]
                    data.ctrl[act_id] = configuration.q[q_idx]
            
            mujoco.mj_step(model, data)
            
            # 可视化
            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0, 0],   
                pos=current_target,  
                mat=np.eye(3).flatten(),
                rgba=[0, 1, 0, 0.5] 
            )
            viewer.sync()
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    input_running = False

if __name__ == "__main__":
    main()