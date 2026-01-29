import mujoco
import mujoco.viewer
import time
import os
import re

def load_xml_content(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"找不到文件: {filename}")
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def extract_section(xml_content, tag_name):
    """提取 <tag>...</tag> 内部的内容"""
    pattern = re.compile(f"<{tag_name}[^>]*>(.*?)</{tag_name}>", re.DOTALL)
    match = pattern.search(xml_content)
    return match.group(1) if match else ""

def main():
    print("🚀 开始反向合并：把 Scene 融入 Robot...")
    
    # 1. 加载文件
    # 主文件：Robot (保留它的 actuator, sensor, option)
    # 插件文件：Scene (只提取里面的体素 voxels)
    host_xml = load_xml_content("../assets/robot.xml")
    guest_xml = load_xml_content("../assets/scene.xml")

    # ========================================================
    # 2. 提取并清洗 Scene 的 Worldbody (体素)
    # ========================================================
    print("⛏️ 正在提取 Scene 中的体素...")
    guest_body = extract_section(guest_xml, "worldbody")
    
    if not guest_body:
        print("⚠️ Scene 中没有 worldbody，尝试提取全部内容...")
        guest_body = re.sub(r'<mujoco[^>]*>', '', guest_xml)
        guest_body = re.sub(r'</mujoco>', '', guest_body)

    # --------------------------------------------------------
    # [核心清洗]：既然 Robot 是主人，我们要删掉 Scene 里所有可能冲突的环境设施
    # --------------------------------------------------------
    
    # 1. 删掉 Scene 里的地板 (floor) -> 用 Robot 的地板
    guest_body = re.sub(
        r'<geom[^>]*name=["\']floor["\'][^>]*>', 
        '', 
        guest_body, 
        flags=re.IGNORECASE
    )

    # 2. 删掉 Scene 里的灯光 (light) -> 用 Robot 的灯光
    guest_body = re.sub(
        r'<light[^>]*>', 
        '', 
        guest_body, 
        flags=re.IGNORECASE
    )

    # 3. 删掉 Scene 里的 skybox 纹理引用 (如果有)
    # (通常体素只是 geom，不引用 skybox，但为了保险)
    
    # ========================================================
    # 3. 提取并清洗 Scene 的 Assets (如果有特殊材质)
    # ========================================================
    print("🎨 正在提取 Scene 的材质...")
    guest_assets = extract_section(guest_xml, "asset")
    
    # 清洗冲突的 asset
    if guest_assets:
        # 删掉 skybox, desert, plane 等环境纹理
        guest_assets = re.sub(r'<texture[^>]*type=["\']skybox["\'][^>]*>', '', guest_assets, flags=re.IGNORECASE)
        guest_assets = re.sub(r'<texture[^>]*name=["\']desert["\'][^>]*>', '', guest_assets, flags=re.IGNORECASE)
        guest_assets = re.sub(r'<texture[^>]*name=["\']plane["\'][^>]*>', '', guest_assets, flags=re.IGNORECASE)
        guest_assets = re.sub(r'<material[^>]*name=["\']plane["\'][^>]*>', '', guest_assets, flags=re.IGNORECASE)

    # ========================================================
    # 4. 执行合并 (注入到 Robot 中)
    # ========================================================
    print("💉 正在注入...")

    # 4.1 合并 Assets
    if guest_assets.strip():
        if "<asset>" in host_xml:
            # 插入到现有的 asset 块中
            idx = host_xml.rfind("</asset>")
            host_xml = host_xml[:idx] + "\n" + guest_assets + "\n" + host_xml[idx:]
        else:
            # 创建新的 asset 块
            new_asset = f"<asset>\n{guest_assets}\n</asset>"
            idx = host_xml.find("<worldbody>")
            host_xml = host_xml[:idx] + "\n" + new_asset + "\n" + host_xml[idx:]
    
    # ========================================================
    # 设定你想要的偏移量
    # ========================================================
    # 假设原本体素墙在 x=2.0
    # 如果你填 [1.0, 0, 0]，体素墙就会移动到 x=3.0 (2.0 + 1.0)
    # 如果你填 [-0.5, 0, 0]，体素墙就会移动到 x=1.5 (2.0 - 0.5)
    offset_x = 0.0
    offset_y = 2.2
    offset_z = 0  # 如果你想把墙埋深一点，可以设为负数

    # 4.2 合并 Worldbody (把体素加进去)
    # 我们把体素包在一个 body 里，方便管理位置
    # 假设 Robot 在原点，体素在 scene.xml 里原本的位置 (例如 x=2.0)
    # 所以我们这里 pos="0 0 0" 保持原位即可
    voxels_block = f"""
    <body name="imported_scene" pos="{offset_x} {offset_y} {offset_z}">
        {guest_body}
    </body>
    """
    
    idx = host_xml.rfind("</worldbody>")
    if idx == -1:
        raise ValueError("robot.xml 损坏：找不到 </worldbody>")
    
    final_xml = host_xml[:idx] + "\n" + voxels_block + "\n" + host_xml[idx:]

    output_filename = "../output/merged_result.xml"
    with open(output_filename, "w", encoding='utf-8') as f:
        f.write(final_xml)
    print(f"💾 已将合并后的文件保存为: {output_filename}")

    # ========================================================
    # 5. 编译与运行
    # ========================================================
    print("✅ 合并完成，编译中...")
    try:
        spec = mujoco.MjSpec.from_string(final_xml)
        model = spec.compile()
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"❌ 编译失败: {e}")
        with open("../output/debug_robot_with_scene.xml", "w", encoding='utf-8') as f:
            f.write(final_xml)
        print("已保存 debug_robot_with_scene.xml 以供检查。")
        return

    print("🎥 启动 Viewer (按 ESC 退出)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置一下默认视角
        viewer.cam.lookat[:] = [1.5, 0, 0.5] # 看向中间
        viewer.cam.distance = 4.0
        
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()