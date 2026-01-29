import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import mujoco
from stable_baselines3 import PPO

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from rl_env import RoadheaderDiggingEnv

def main():
    # 1. 加载最新的模型
    # 请手动修改这里的文件名，或者指向 roadheader_final
    model_path = os.path.join(current_dir, "models", "roadheader_final.zip")
    
    if not os.path.exists(model_path):
        print("❌ 找不到模型文件，请先运行 train.py")
        return

    print(f"📂 加载模型: {model_path}")
    model = PPO.load(model_path)

    # 2. 创建测试环境 (开启 render_mode="human")
    env = RoadheaderDiggingEnv(render_mode="human")
    
    print("🎥 开始演示...")
    obs, _ = env.reset()
    
    total_reward = 0
    while True:
        # deterministic=True 表示不使用随机探索，直接输出最优动作
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 如果需要减慢速度方便观看，可以加 time.sleep(0.01)
        
        if terminated or truncated:
            print(f"🔄 回合结束。总得分: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0

if __name__ == "__main__":
    main()