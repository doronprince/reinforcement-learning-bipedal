import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO  # The expert is a PPO agent
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import load_from_hub
from tqdm import tqdm

# --- Configuration ---
ENV_NAME = "BipedalWalker-v3"
# This is the official expert model from the SB3 team
EXPERT_MODEL_REPO_ID = "sb3/ppo-BipedalWalker-v3"
FILENAME = "ppo-BipedalWalker-v3.zip"
N_EPISODES_TO_GENERATE = 100
DATASET_STATES_PATH = "expert_states_bipedal.npy"
DATASET_ACTIONS_PATH = "expert_actions_bipedal.npy"

# --- 1. Load Environment and Expert Model ---
# Note: PPO is an "on-policy" algorithm, so we must use a DummyVecEnv
env = DummyVecEnv([lambda: gym.make(ENV_NAME)])

print(f"Downloading expert model {EXPERT_MODEL_REPO_ID}...")
checkpoint = load_from_hub(
    repo_id=EXPERT_MODEL_REPO_ID,
    filename=FILENAME
)
expert_model = PPO.load(checkpoint, env=env)
print("Expert model loaded.")

# --- 2. Generate and Save Trajectories ---
expert_states = []
expert_actions = []

print(f"Generating {N_EPISODES_TO_GENERATE} expert episodes...")
for _ in tqdm(range(N_EPISODES_TO_GENERATE)):
    obs = env.reset()
    done = False

    while not done:
        # Get expert action
        action, _ = expert_model.predict(obs, deterministic=True)

        # Store the (state, action) pair
        expert_states.append(obs[0])
        expert_actions.append(action[0])

        obs, reward, done, info = env.step(action)

env.close()

# --- 3. Save Data to Files ---
print("Saving dataset...")
np.save(DATASET_STATES_PATH, np.array(expert_states))
np.save(DATASET_ACTIONS_PATH, np.array(expert_actions))

print(f"Successfully generated {len(expert_states)} transitions.")
