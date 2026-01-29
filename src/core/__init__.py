# 1. 导入地形生成函数
from .create_scene import create_scene

# 2. 导入模型合并逻辑 (将 main 函数重命名为更有意义的名字)
from .merge_pure import main as merge_robot_and_scene

# 3. 导入挖掘系统类 (直接导入，假设文件已存在)
from .digging_system_mesh import MeshDiggingSystem

# 定义外部调用 from core import * 时能获取到的内容
__all__ = [
    "create_scene",
    "merge_robot_and_scene",
    "MeshDiggingSystem"
]