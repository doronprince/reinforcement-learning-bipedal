# Does a behaviour-cloning head start help SAC on BipedalWalker?

A controlled comparison of two Soft Actor-Critic (SAC) agents on
[`BipedalWalker-v3`](https://gymnasium.farama.org/environments/box2d/bipedal_walker/):

- **Bootstrapped:** the actor is first trained by behaviour cloning on demonstrations
  from a pretrained PPO expert, then fine-tuned with SAC.
- **Scratch:** the same SAC agent, same network and same hyperparameters, starting
  from random weights.

The question is whether imitation pretraining speeds up learning, or whether the
cloned policy is forgotten once reinforcement learning starts.

## Method

1. **Demonstrations.** `generate_dataset.py` downloads the expert
   [`sb3/ppo-BipedalWalker-v3`](https://huggingface.co/sb3/ppo-BipedalWalker-v3)
   from the Hugging Face Hub and records 100 deterministic episodes as
   (state, action) pairs.
2. **Behaviour cloning.** `finetune_clone.py` trains a 24 → 256 → 256 → 4 MLP
   with a tanh output on those pairs (MSE loss, Adam, learning rate 1e-3,
   50 epochs, batch size 256).
3. **Weight transfer.** The cloned layers are copied into the SAC actor:
   the two hidden layers into `latent_pi` and the output layer into the mean
   head `mu`. The log-standard-deviation head and the critics start from random
   weights.
4. **Fine-tuning.** SAC trains for 100,000 environment steps.
5. **Baseline.** `train_sac_scratch.py` trains SAC for the same 100,000 steps
   with the same `[256, 256]` actor and critic networks and the same SB3 default
   hyperparameters.

Both runs evaluate the deterministic policy every 5,000 steps with SB3's
`EvalCallback` and log to TensorBoard.

## Run it

```bash
pip install -r requirements.txt     # Box2D may need swig: pip install swig first
python generate_dataset.py          # writes expert_states_bipedal.npy, expert_actions_bipedal.npy
python finetune_clone.py            # BC pretraining, then SAC fine-tuning
python train_sac_scratch.py         # baseline
tensorboard --logdir ./tensorboard_logs/
```

Evaluation results are written to `./logs/bipedal_finetune/` and
`./logs/bipedal_scratch/` (`evaluations.npz`, plus the best model of each run).

## Results

Not published yet. The plan is to report mean evaluation return against
environment steps for both runs, averaged over several random seeds, together
with the PPO expert's return as a reference line.

## Limitations

- A single seed per run cannot separate the effect of pretraining from noise.
- 100,000 steps is short for BipedalWalker; SAC from scratch usually needs more
  to walk reliably, so this measures the early-learning effect.
- Only the actor's mean is pretrained. The critic starts untrained, so its early
  value estimates can pull the actor away from the cloned behaviour.

## Licence

MIT. The expert policy is downloaded from the Hugging Face Hub at run time and
is not included in this repository; see its model card for its terms.
