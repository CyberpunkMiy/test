import mujoco
import numpy as np
from digging_system_mesh import MeshDiggingSystem

# 1. 这里填你的 xml 路径
xml_path = "../output/merged_result.xml" 

# 2. 这里填你的 Mesh 名字 (xml <mesh file="..."> 对应的名字)
mesh_name = "jiegetou_link" 

try:
    print("--- 开始测试 Mesh 清洗逻辑 ---")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 3. 初始化系统 (这会自动触发清洗逻辑)
    # 你可以试着修改 clean_threshold 看看效果
    sys = MeshDiggingSystem(model, data, mesh_name, clean_threshold=0.5)
    
    print("--- 测试结束 ---")

except ValueError as e:
    print(e)
except Exception as e:
    print(f"发生错误: {e}")