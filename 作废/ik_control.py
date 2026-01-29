import time
import threading
import mujoco
import mujoco.viewer
import numpy as np
import sys

# === 配置参数 ===
XML_PATH = 'robot.xml'
# 降低数学阻尼，因为我们已经有很好的物理阻尼(kv)了，让计算更灵敏
IK_DAMPING = 0.02 

# 全局变量
current_target = np.array([0.0, 1.5, 0.5])
input_running = True

def get_joint_addresses(model, joint_names):
    """获取关节索引和执行器ID"""
    dof_ids = []
    actuator_ids = []
    for name in joint_names:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof_ids.append(model.jnt_dofadr[jid])
            found = False
            for act_id in range(model.nu):
                if model.actuator_trnid[act_id, 0] == jid:
                    actuator_ids.append(act_id)
                    found = True
                    break
            if not found:
                print(f"Warning: No actuator found for joint {name}")
        except:
            print(f"Error: Joint {name} not found")
    return dof_ids, actuator_ids

def calculate_ik(jac, target_pos, current_pos, dof_ids, damping):
    """阻尼最小二乘法 IK"""
    error = target_pos - current_pos
    jacp = jac[:3, dof_ids] 
    lambda_sq = damping ** 2
    J_T = jacp.T
    inv_term = np.linalg.inv(jacp @ J_T + lambda_sq * np.eye(3))
    dq = J_T @ inv_term @ error
    return dq

def user_input_thread():
    """输入监听线程"""
    global current_target, input_running
    print(">>> 极速控制模式已就绪！")
    print(">>> 输入坐标 (例如: 0.5 2.0 1.0) 并回车。输入 'q' 退出。")

    while input_running:
        try:
            user_str = input()
            if user_str.strip().lower() == 'q':
                print("停止输入监听。")
                break
            parts = user_str.strip().split()
            if len(parts) == 3:
                x, y, z = map(float, parts)
                current_target = np.array([x, y, z])
                print(f"--> 目标更新: [{x}, {y}, {z}]")
            else:
                print("格式错误！请输入三个数字。")
        except:
            pass

def main():
    global input_running
    
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"无法加载模型: {e}")
        return

    site_name = 'cutter_tip'
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    except:
        print("XML中未找到 'cutter_tip' site。")
        return

    control_joints = ['huizhuan_joint', 'updown_joint', 'front_back_joint']
    dof_ids, actuator_ids = get_joint_addresses(model, control_joints)
    jac = np.zeros((6, model.nv))

    # 启动输入线程
    t = threading.Thread(target=user_input_thread, daemon=True)
    t.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 5
        viewer.cam.azimuth = 140
        
        while viewer.is_running():
            step_start = time.time()
            
            # 1. 获取状态
            current_pos = data.site_xpos[site_id]
            dist = np.linalg.norm(current_target - current_pos)
            
            # 2. 控制逻辑
            if dist > 0.002: 
                # 计算向量
                vector_to_target = current_target - current_pos
                full_distance = np.linalg.norm(vector_to_target)
                
                # === 修改 1：加大前瞻距离 ===
                # 之前是 0.05 (5cm)。在高阻尼下这太小了，电机推不动。
                # 改为 0.15 (15cm)，制造更大的误差，产生更大的扭矩。
                segment_length = 0.15 
                
                if full_distance > segment_length:
                    direction = vector_to_target / full_distance 
                    virtual_target = current_pos + direction * segment_length
                else:
                    virtual_target = current_target

                # IK 计算
                mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
                dq = calculate_ik(jac, virtual_target, current_pos, dof_ids, IK_DAMPING)
                
                # 速度限制逻辑
                step_magnitude = np.linalg.norm(dq)
                
                # === 修改 2：稍微放宽一点最大步长 ===
                # 配合大的 segment_length，稍微给一点速度空间
                max_allowable_step = 0.08 
                braking_dist = 0.25
                
                if dist < braking_dist:
                    limit = max_allowable_step * (dist / braking_dist) 
                else:
                    limit = max_allowable_step

                if step_magnitude > 1e-4:
                    scale = min(1.0, limit / step_magnitude)
                else:
                    scale = 1.0

                for i, act_id in enumerate(actuator_ids):
                    joint_id = model.actuator_trnid[act_id, 0]
                    qpos_adr = model.jnt_qposadr[joint_id]
                    current_val = data.qpos[qpos_adr]
                    new_ctrl = current_val + dq[i] * scale
                    ctrl_range = model.actuator_ctrlrange[act_id]
                    data.ctrl[act_id] = np.clip(new_ctrl, ctrl_range[0], ctrl_range[1])

            # 3. === 修改 3：增加物理步进 (最关键) ===
            # 之前是 1。在高阻尼下，1ms 根本来不及加速。
            # 改回 5。既保证了有力气动，又因为有 Virtual Target 约束，不会震荡。
            physics_steps = 5
                
            for _ in range(physics_steps):
                mujoco.mj_step(model, data)
            
            # 4. 可视化
            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0, 0],   
                pos=current_target,  
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 0.5]  
            )

            viewer.sync()
        
        input_running = False

if __name__ == "__main__":
    main()