# Reinforcement Learning: Imitation-Bootstrapped SAC vs. Scratch SAC

Comparing a Soft Actor-Critic (SAC) agent trained from scratch against one bootstrapped with behavior cloning from a pretrained PPO expert, on \`BipedalWalker-v3\` (Gymnasium).

## What's here

- \`generate_dataset.py\` — downloads an expert PPO policy from the Hugging Face Hub (\`sb3/ppo-BipedalWalker-v3\`), rolls out episodes, and saves the resulting (state, action) pairs as a behavior-cloning dataset.
- \`finetune_clone.py\` — pretrains a small policy network via behavior cloning on the expert dataset, transfers those weights into an SB3 SAC actor, then fine-tunes with reinforcement learning.
- \`train_sac_scratch.py\` — trains a SAC agent from scratch (same network architecture) as a baseline for comparison.

## Approach

1. Generate expert demonstrations from a pretrained PPO agent.
2. Pretrain a policy via supervised behavior cloning on those demonstrations.
3. Transfer the cloned weights into a SAC actor and continue training with RL (imitation + fine-tuning).
4. Compare against a SAC agent trained purely from scratch, using TensorBoard logs.

## Requirements

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`bash
python generate_dataset.py
python finetune_clone.py
python train_sac_scratch.py
\`\`\`

Training logs are written to \`./logs/\` and \`./tensorboard_logs/\`; view with:

\`\`\`bash
tensorboard --logdir ./tensorboard_logs/
\`\`\`

## Status

Experimental / research project.
