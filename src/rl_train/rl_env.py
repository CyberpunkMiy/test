import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import sys

# ==========================================
# 1. 路径与模块自动修复
# ==========================================
# 获取当前文件 (src/core/rl_env.py) 的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 src 目录 (即 core 的上一级)
src_dir = os.path.dirname(current_dir)

# 将 src 加入系统路径，这样就能通过 "from control.xxx" 导入了
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from control.trajectory_control_interactive import RoadheaderController
    from core.digging_system_mesh import MeshDiggingSystem
except ImportError as e:
    raise ImportError(f"❌ 模块导入失败! 请检查目录结构。\n错误信息: {e}")

class RoadheaderDiggingEnv(gym.Env):
    """
    【强化学习环境】掘进机局部导航挖掘 (Roadheader Local Navigation)
    
    核心特性：
    1. 恒定速度控制：AI 只能决定方向，无法决定快慢。
    2. 局部目标引导：使用 'get_local_target' 引导机器人贴壁挖掘，避免全局质心导致的“指空”问题。
    3. 撞墙惩罚机制：检测机器人是否卡死，给予重罚以加速收敛。
    
    状态空间 (10维): [末端位置(3), 关节角度(3), 局部目标点(3), 任务进度(1)]
    动作空间 (3维):  [dx, dy, dz] (归一化方向向量)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, xml_path=None, mesh_name="jiegetou_link", body_name="jiegetou_link", render_mode=None):
        super().__init__()
        
        # --- 1. 路径配置 ---
        if xml_path is None:
            # 默认去 src/output/merged_result.xml 找模型
            self.xml_path = os.path.join(src_dir, "output", "merged_result.xml")
        else:
            self.xml_path = xml_path

        self.mesh_name = mesh_name
        self.body_name = body_name
        self.render_mode = render_mode
        
        print(f"🌍 [RL_Env] 环境正在初始化... 模型路径: {self.xml_path}")
        
        # --- 2. 加载 MuJoCo 模型 ---
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"❌ 找不到 XML 文件: {self.xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # --- 3. 初始化子系统 ---
        # A. 运动控制器 (IK Solver)
        self.controller = RoadheaderController(self.model, self.data)
        
        # B. 挖掘交互系统 (Voxel System)
        self.digger = MeshDiggingSystem(
            self.model, self.data, 
            mesh_name=self.mesh_name, 
            scene_body_name="voxel_target"
        )
        
        # --- 4. 定义空间 (Spaces) ---
        # 动作: [dx, dy, dz] 方向向量，范围 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 观察: [末端(3), 关节(3), 局部目标(3), 进度(1)] = 10维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # --- 5. 仿真超参数 ---
        self.step_size = 0.05       # 恒定速度: 每次移动 5cm
        self.max_steps = 2000       # 最大步数 (防止死循环)
        self.current_step = 0
        self.initial_voxel_count = 1 
        
        # 渲染句柄
        self.viewer = None

    def reset(self, seed=None, options=None):
        """
        环境重置：物理归位 + 体素墙复原
        """
        super().reset(seed=seed)
        
        # 1. MuJoCo 物理重置
        mujoco.mj_resetData(self.model, self.data)
        
        # 2. 挖掘系统重置 (恢复墙壁)
        if hasattr(self.digger, 'reset'):
            self.digger.reset()
        else:
            print("⚠️ [Warning] MeshDiggingSystem 缺少 reset() 方法！")
        
        # 3. 重新统计初始体素 (用于计算进度百分比)
        if hasattr(self.digger, 'active_voxels'):
            self.initial_voxel_count = max(len(self.digger.active_voxels), 1)

        # 4. 刷新前向动力学 (确保所有坐标更新)
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        # 返回初始观测
        return self._get_obs(), {}

    def step(self, action):
        """
        核心步进逻辑：执行动作 -> 物理模拟 -> 计算奖励 -> 返回状态
        """
        self.current_step += 1
        
        # --- A. 动作处理 (实现恒定速度) ---
        # 获取移动前的位置
        pos_before = self.controller.get_current_site_pos()
        
        # 归一化动作向量 (只取方向)
        action_norm = np.linalg.norm(action)
        direction = np.zeros(3)
        if action_norm > 1e-6:
            direction = action / action_norm
            
        # 计算 IK 目标位置 = 当前位置 + 方向 * 固定步长
        target_pos = pos_before + direction * self.step_size
        
        # --- B. 执行控制 ---
        # 1. IK 解算
        q_cmd = self.controller.solve_ik(target_pos)
        
        # 2. 如果有解，驱动电机
        if q_cmd is not None:
            self.controller.control_actuators(q_cmd)
        
        # 3. 物理步进 (Frame Skip = 5，模拟约 0.01~0.05秒的物理过程)
        for _ in range(5): 
            mujoco.mj_step(self.model, self.data)
            
        # --- C. 撞墙/卡顿检测 (Stuck Detection) ---
        pos_after = self.controller.get_current_site_pos()
        actual_dist = np.linalg.norm(pos_after - pos_before)
        
        # 判定逻辑：如果 AI 意图移动 (action > 0.1) 但实际没怎么动 (移动距离 < 步长的 10%)
        # 这通常意味着撞到了关节限位，或者撞到了不可破坏的障碍物
        is_stuck = False
        if action_norm > 0.1 and actual_dist < (self.step_size * 0.1):
            is_stuck = True

        # --- D. 奖励计算 (Reward Shaping) ---
        reward = 0.0
        
        # 1. 挖掘奖励 (主要目标: +5.0 / voxel)
        voxels_removed = self.digger.perform_cutting(self.body_name)
        reward += voxels_removed * 5.0
        
        # 2. 基础时间惩罚 (效率目标: -0.1 / step)
        reward -= 0.1 

        # 3. 状态惩罚与引导
        if is_stuck:
            # 撞墙重罚，迫使 AI 换个方向
            reward -= 1.0 
        elif voxels_removed == 0:
            # 引导奖励：如果这步没挖到土，也没卡住
            # 就计算它离【局部目标】的距离，越近扣分越少
            target = self._get_target_center()
            dist = np.linalg.norm(pos_after - target)
            reward -= dist * 0.05 # 距离引导系数

        # --- E. 结束判定 ---
        terminated = False
        truncated = False
        
        # 任务完成：所有体素被清除
        current_voxel_count = len(self.digger.active_voxels)
        if current_voxel_count == 0:
            terminated = True
            reward += 1000.0 # 胜利大奖
            print(f"🎉 Episode {self.current_step}: 任务完成！所有体素已清除。")
            
        # 超时截断
        if self.current_step >= self.max_steps:
            truncated = True

        # --- F. 渲染 ---
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_target_center(self):
        """
        【关键逻辑】获取导航目标点
        优先使用 'get_local_target' (最近K个中心)，
        如果底层不支持，回退到 'get_remaining_voxel_center' (全局质心)
        """
        # 1. 获取当前截割头位置
        head_pos = self.controller.get_current_site_pos()
        
        # 2. 调用 MeshDiggingSystem 的新方法：获取最近 50 个体素的中心
        # 这能保证目标点始终“贴在墙壁表面”，而不是悬浮在被挖空的中心
        if hasattr(self.digger, 'get_local_target'):
            return self.digger.get_local_target(head_pos, k=50)
            
        # 3. 兼容性回退
        if hasattr(self.digger, 'get_remaining_voxel_center'):
            return self.digger.get_remaining_voxel_center()
            
        return np.zeros(3)

    def _get_obs(self):
        """
        获取观测向量 (10维)
        """
        # 1. 机器人自身状态
        tip_pos = self.controller.get_current_site_pos()
        joint_pos = self.controller.get_joint_positions()
        
        # 2. 局部导航目标 (指向最近的墙壁)
        target_center = self._get_target_center()
        
        # 3. 全局进度 (0.0 ~ 1.0)
        current = len(self.digger.active_voxels) if hasattr(self.digger, 'active_voxels') else 0
        progress = current / max(self.initial_voxel_count, 1)
        
        # 拼接向量
        obs = np.concatenate([
            tip_pos,        # (3,)
            joint_pos,      # (3,)
            target_center,  # (3,) <-- 智能导航点
            [progress]      # (1,)
        ]).astype(np.float32)
        
        return obs

    def render(self):
        """渲染环境"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()