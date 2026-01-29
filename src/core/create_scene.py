import xml.etree.ElementTree as ET
from xml.dom import minidom
import math
import os

def create_scene():
    mujoco = ET.Element('mujoco', model='digging_tunnel_ellipse_4m')

    # --- 1. 内存配置 ---
    # 分配 4GB 内存，保证加载稳定
    ET.SubElement(mujoco, 'size', memory="4096M")
    
    # 调整统计中心: 高度4m，中心设在2m处
    ET.SubElement(mujoco, 'statistic', extent="8", center="2 0 2.0")
    ET.SubElement(mujoco, 'option', gravity='0 0 -9.81')

    # --- 2. 视觉优化 ---
    visual = ET.SubElement(mujoco, 'visual')
    # 增强光照质感
    ET.SubElement(visual, 'headlight', diffuse="0.7 0.7 0.7", ambient="0.4 0.4 0.4", specular="0.1 0.1 0.1")
    ET.SubElement(visual, 'map', zfar="100") 
    
    worldbody = ET.SubElement(mujoco, 'worldbody')

    # 光照 - 适配 4m 高度
    ET.SubElement(worldbody, 'light', pos='0 0 7', dir='0 0 -1', diffuse='0.8 0.8 0.8')
    ET.SubElement(worldbody, 'light', pos='3 -3 5', dir='0 1 -0.5', diffuse='0.5 0.5 0.5')

    # 地板
    ET.SubElement(worldbody, 'geom', name='floor', type='plane', size='10 10 0.1', rgba='0.8 0.9 0.8 1')

    # --- 3. 生成半椭圆体素 ---
    wall_x_start = 2.0
    
    # 几何参数
    width_radius = 2.25  # 半宽 (Total width = 4.5m)
    height_radius = 4.0  # 高度 (Total height = 4.0m)
    depth_length = 1.0   # 深度
    
    # [分辨率微调]
    # 高度从4.5降到4.0，体积减小，我们可以稍微把精度提高到 0.14m
    # 预计体素数量 ~6500 左右，安全范围内
    spacing = 0.14       
    box_size = spacing / 2 
    
    # 椭圆方程筛选容差
    ellipse_tolerance = 1.05

    # 计算循环步数
    n_depth = int(depth_length / spacing)
    n_height = int(height_radius / spacing) + 2
    n_width_half = int(width_radius / spacing) + 2

    print(f"Generating Tunnel: Height={height_radius}m, Width={width_radius*2}m")
    
    voxel_count = 0

    for k in range(n_depth):
        x = wall_x_start + k * spacing + box_size
        
        for i in range(n_height):
            z = box_size + i * spacing
            
            for j in range(-n_width_half, n_width_half + 1):
                y = j * spacing
                
                # --- 椭圆方程 ---
                # (y/a)^2 + (z/b)^2 <= 1
                norm_dist_sq = (y / width_radius)**2 + (z / height_radius)**2
                
                if norm_dist_sq <= ellipse_tolerance:
                    name = f"vx_{k}_{i}_{j}"
                    body = ET.SubElement(worldbody, 'body', name=name, pos=f"{x:.3f} {y:.3f} {z:.3f}")
                    
                    # 幻影设置 (无碰撞)
                    ET.SubElement(body, 'geom', 
                                  type='box', 
                                  size=f"{box_size} {box_size} {box_size}", 
                                  rgba="0.35 0.35 0.4 0.8", 
                                  contype="0", 
                                  conaffinity="0")
                    voxel_count += 1

    # --- 保存 ---
    xml_str = minidom.parseString(ET.tostring(mujoco)).toprettyxml(indent="  ")
    
    # 获取当前脚本所在的绝对路径 (例如 .../Mujoco_Tunnel_Project/core/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构造目标路径: 上一级目录 -> assets 文件夹
    # 这里的 ".." 表示向上一级，"assets" 是目标文件夹名
    assets_dir = os.path.join(current_dir, "../assets")
    
    # (可选) 如果 assets 文件夹不存在，自动创建它，防止报错
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        print(f"Created directory: {assets_dir}")
    
    # 组合完整的文件路径
    filename = os.path.join(assets_dir, "scene.xml")
    
    # 写入文件
    with open(filename, "w", encoding='utf-8') as f:
        f.write(xml_str)
    
    print(f"Generated Tunnel (H=4.0m) with {voxel_count} voxels.")
    print(f"Saved to: {filename}") # 打印完整路径确认

if __name__ == "__main__":
    create_scene()