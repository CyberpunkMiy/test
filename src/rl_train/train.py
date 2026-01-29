import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# ================= 路径修复 =================
# 确保脚本能找到 core 和 control 模块
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/rl_train
parent_dir = os.path.dirname(current_dir)                # src
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 导入你的环境
from rl_env import RoadheaderDiggingEnv

def make_env(rank, seed=0):
    """
    环境工厂函数，用于创建独立的环境实例
    """
    def _init():
        # XML 路径自动定位到 src/output/merged_result.xml
        env = RoadheaderDiggingEnv(render_mode=None) # 训练时不要渲染(None)，速度最快
        # 使用 Monitor 包装环境，记录 Reward/Length 到日志文件
        log_file = os.path.join(current_dir, "logs", str(rank))
        env = Monitor(env, log_file)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    # --- 1. 配置参数 ---
    TRAIN_TIMESTEPS = 2_000_000  # 总训练步数 (建议至少 100万)
    N_ENVS = 1                   # 并行环境数量 (调试用1，生产训练可用 4 或 8)
    LEARNING_RATE = 3e-4         # 学习率
    BATCH_SIZE = 64              # 批次大小
    
    # 路径设置
    models_dir = os.path.join(current_dir, "models")
    logs_dir = os.path.join(current_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # --- 2. 检测 GPU ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*50)
    print(f"🚀 训练设备: {device.upper()}")
    if device == "cuda":
        print(f"   GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 版本: {torch.version.cuda}")
    print("="*50)

    # --- 3. 创建环境 ---
    # 使用 DummyVecEnv (单进程) 或 SubprocVecEnv (多进程并行)
    # 对于 MuJoCo，单进程通常已经很快了，多进程主要用于 CPU 密集型计算
    env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])

    # --- 4. 定义 PPO 模型 ---
    # MlpPolicy: 使用全连接网络 (因为输入是向量状态)
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",          # 👈 强制使用 CPU
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=2048,           # 每次更新采集的步数
        batch_size=BATCH_SIZE,
        n_epochs=10,            # 每次更新优化 10 轮
        gamma=0.99,             # 折扣因子
        gae_lambda=0.95,
        ent_coef=0.01,          # 熵系数：增加一点点随机探索，防止过早收敛
        tensorboard_log=logs_dir
    )

    # --- 5. 设置回调函数 (定期保存) ---
    # 每 50,000 步保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="roadheader_ppo"
    )

    # --- 6. 开始训练 ---
    print(f"🏃 开始训练... 目标步数: {TRAIN_TIMESTEPS}")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True # 显示进度条
        )
    except KeyboardInterrupt:
        print("\n⚠️ 检测到中断，正在保存当前模型...")
    
    # --- 7. 保存最终模型 ---
    final_path = os.path.join(models_dir, "roadheader_final")
    model.save(final_path)
    print(f"✅ 训练完成！最终模型已保存至: {final_path}.zip")
    print(f"⏱️ 耗时: {(time.time() - start_time)/60:.2f} 分钟")

if __name__ == "__main__":
    main()