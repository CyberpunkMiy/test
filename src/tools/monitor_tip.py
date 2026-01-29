import time
import mujoco
import mujoco.viewer
import numpy as np
import sys

def main():
    xml_path = '../output/merged_result.xml'
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"无法加载模型: {e}")
        return

    # 获取 cutter_tip 的 ID
    site_name = 'cutter_tip'
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    except:
        print(f"错误: 找不到名为 {site_name} 的 site。")
        return

    print("========================================================")
    print("  掘进机末端坐标实时监视器")
    print("  操作说明:")
    print("  1. 在弹出的窗口右侧，展开 'Control' 面板")
    print("  2. 拖动滑动条 (act_left_right, act_up_down, etc.)")
    print("  3. 下方将实时显示 Cutter Tip 的 (X, Y, Z) 坐标")
    print("========================================================")
    
    # 等待用户看清提示
    time.sleep(2)

    # 启动 Passive Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 如果需要，可以将相机中心对准末端（可选）
        viewer.cam.lookat[:] = [0.5, 0, 0.5]
        viewer.cam.distance = 5
        viewer.cam.azimuth = 135
        
        while viewer.is_running():
            step_start = time.time()

            # 1. 执行物理步进 (这是必须的，否则机器不会动)
            # 即使是手动控制，也需要物理引擎计算动力学
            mujoco.mj_step(model, data)

            # 2. 获取实时坐标
            pos = data.site_xpos[site_id]
            x, y, z = pos[0], pos[1], pos[2]

            # 3. 在终端实时刷新显示 (使用 \r 回车符覆盖当前行)
            # 格式化输出：保留3位小数
            status_str = f"📍 Cutter Tip Pos | X: {x:8.3f} | Y: {y:8.3f} | Z: {z:8.3f}"
            sys.stdout.write(f"\r{status_str}")
            sys.stdout.flush()

            # 4. 同步 Viewer 显示
            viewer.sync()

            # 5. 控制帧率 (防止循环跑太快看不清，且占用过多CPU)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()