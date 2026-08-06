import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# --- Configuration ---
ENV_NAME = "BipedalWalker-v3"
SCRATCH_MODEL_PATH = "scratch_sac_bipedal"
LOG_DIR = "./logs/bipedal_scratch/"
TOTAL_TIMESTEPS = 100000 # Must be the SAME as your finetune script

# --- Create and Train SAC from Scratch ---
print("Creating and training SAC agent from scratch...")
env = gym.make(ENV_NAME)

eval_env = gym.make(ENV_NAME)
eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_DIR,
                             log_path=LOG_DIR, eval_freq=5000,
                             deterministic=True, render=False)

# Use the same network architecture for a fair comparison
policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
model = SAC(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./tensorboard_logs/bipedal_scratch/"
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    log_interval=4
)

model.save(SCRATCH_MODEL_PATH)
print("Training from scratch complete.")
