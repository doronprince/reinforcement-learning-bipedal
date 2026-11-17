import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# --- Configuration ---
ENV_NAME = "BipedalWalker-v3"
DATASET_STATES_PATH = "expert_states_bipedal.npy"
DATASET_ACTIONS_PATH = "expert_actions_bipedal.npy"
BC_MODEL_SAVE_PATH = "bc_model_bipedal.pth"
FINETUNED_MODEL_PATH = "finetuned_sac_bipedal"
LOG_DIR = "./logs/bipedal_finetune/"
TOTAL_TIMESTEPS = 100000  # More complex task, more steps
BC_EPOCHS = 50


# --- 1. Define the Policy Network ---
# This network must be compatible with SB3's SAC MlpPolicy
# SB3's default is [256, 256] for the actor (pi) network
class BipedalPolicy(nn.Module):
    def __init__(self):
        super(BipedalPolicy, self).__init__()
        # BipedalWalker has 24 states and 4 actions
        self.network = nn.Sequential(
            nn.Linear(24, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # Note: The SAC actor in SB3 has two separate heads
            # We are only pre-training the *main* part.
            # We will load these weights into the SAC actor's 'latent_pi'
        )
        # This part matches the SAC actor's action output layer
        self.action_head = nn.Linear(256, 4)

    def forward(self, x):
        # This part is used for weight loading
        features = self.network(x)
        action = self.action_head(features)
        # BipedalWalker actions are [-1, 1], so Tanh is correct
        return torch.tanh(action)


# --- 2. [IL STEP] Train the Behavior Cloning Model ---
print("--- Starting Behavior Cloning Pre-training ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
states = np.load(DATASET_STATES_PATH)
actions = np.load(DATASET_ACTIONS_PATH)
states_tensor = torch.tensor(states, dtype=torch.float32)
actions_tensor = torch.tensor(actions, dtype=torch.float32)
dataset = TensorDataset(states_tensor, actions_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Create model and optimizer
bc_model = BipedalPolicy().to(device)
optimizer = optim.Adam(bc_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# BC Training loop
for epoch in range(BC_EPOCHS):
    epoch_loss = 0.0
    for batch_states, batch_actions in dataloader:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)

        pred_actions = bc_model(batch_states)
        loss = loss_fn(pred_actions, batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"BC Epoch [{epoch + 1}/{BC_EPOCHS}], Loss: {epoch_loss / len(dataloader):.6f}")

torch.save(bc_model.state_dict(), BC_MODEL_SAVE_PATH)
print("--- Behavior Cloning Finished ---")

# --- 3. [RL STEP] Create SAC Agent and Transfer Weights ---
print("--- Starting RL Fine-Tuning ---")
env = gym.make(ENV_NAME)
eval_env = gym.make(ENV_NAME)
eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_DIR,
                             log_path=LOG_DIR, eval_freq=5000,
                             deterministic=True, render=False)

# Create SAC model with the *same network architecture*
policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
rl_model = SAC(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./tensorboard_logs/bipedal_finetune/",
    device=device,
    # Same SAC hyperparameters as train_sac_scratch.py (SB3 defaults:
    # train_freq=1 step, gradient_steps=1) so the only difference is the BC start.
)

# --- 4. [HYBRID STEP] Load the pre-trained weights ---
# We load the BC weights into the SAC agent's actor network
# Note: The SAC actor has `latent_pi` (our `network`) and `mu` (our `action_head`)
# We need to map them correctly.
bc_state_dict = torch.load(BC_MODEL_SAVE_PATH, map_location=device)
actor_state_dict = rl_model.policy.actor.state_dict()

# Map the BC network weights to the SAC actor's `latent_pi`
actor_state_dict["latent_pi.0.weight"] = bc_state_dict["network.0.weight"]
actor_state_dict["latent_pi.0.bias"] = bc_state_dict["network.0.bias"]
actor_state_dict["latent_pi.2.weight"] = bc_state_dict["network.2.weight"]
actor_state_dict["latent_pi.2.bias"] = bc_state_dict["network.2.bias"]

# Map the BC action head to the SAC actor's `mu` (mean) head
actor_state_dict["mu.weight"] = bc_state_dict["action_head.weight"]
actor_state_dict["mu.bias"] = bc_state_dict["action_head.bias"]

# Load the modified state dict into the actor
rl_model.policy.actor.load_state_dict(actor_state_dict)
# Note: SB3's SAC keeps a target network only for the critic (critic_target),
# not for the actor, so there is no actor target to update here.

print("BC weights successfully transferred to SAC actor.")

# --- 5. Start RL Fine-Tuning ---
print(f"Starting RL fine-tuning for {TOTAL_TIMESTEPS} timesteps...")
rl_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    log_interval=4
)

rl_model.save(FINETUNED_MODEL_PATH)
print("--- Fine-Tuning Complete ---")
