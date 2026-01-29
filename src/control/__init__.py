import os

# 1. 暴露核心控制类，方便外部直接从包名导入
# 例如：from your_package import RoadheaderController
from .trajectory_control_interactive import RoadheaderController

# 2. 定义常用的默认路径（相对于当前包的位置）
# 这样在不同环境下运行时，模型文件路径不会轻易失效
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML_PATH = os.path.join(PACKAGE_ROOT, "../assets/robot.xml")

# 3. 提取机器人配置元数据，方便 RL 环境获取观测空间(Observation)和动作空间(Action)的大小
ROBOT_CONFIG = {
    "joint_names": ["huizhuan_joint", "updown_joint", "front_back_joint"],
    "actuator_names": ["act_left_right", "act_up_down", "act_front_back"],
    "site_name": "cutter_tip",
    "control_freq": 50.0,
    "action_dim": 3,  # 对应 3 个关节的控制
    "obs_dim": 6      # 例如：3个关节角度 + 3个末端坐标
}

def make_controller(model, data):
    """
    快速实例化控制器的工厂函数
    """
    return RoadheaderController(model, data)

__all__ = ["RoadheaderController", "make_controller", "ROBOT_CONFIG", "DEFAULT_XML_PATH"]