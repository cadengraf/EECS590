# EECS590

This repository contains a reinforcement learning project focused on grid-based drone delivery. The agent must navigate through pipe-like lane layouts, pick up a package, and deliver it to a goal location. The project includes both classical reinforcement learning methods and Deep RL methods built on top of a shared delivery environment.

For the Version 3 final writeup, including implementation decisions, rejected options, current limitations, and continuing work, see [V3_FINAL_REPORT.md](V3_FINAL_REPORT.md).

## Problem Statement

The main goal of this project is autonomous package delivery with drones. The maze or pipe-grid layout is intentional: it represents constrained air spaces that avoid flying directly over residential areas. Instead of giving the drone freedom to cross any cell in the map, the task asks the agent to stay inside approved lanes, pick up a package, and deliver it to the assigned destination.

The project has evolved from classical grid-world delivery into quadcopter-style physics experiments. The newer quad-physics work keeps the high-level package-delivery goal, but replaces simple grid actions with continuous roll and pitch control, making the learned policy responsible for both navigation and stable movement.

## Acknowledgments and Assistance

I used ChatGPT/Codex as a programming assistant while developing and cleaning this repository. I used it for debugging help, code review, documentation edits, and implementation support. The project direction, algorithm choices, experiment design, and final decisions are my own.

## Quad-Physics Updates

The V3 work is focused on `quad_physics/`, especially curriculum training for fixed and multi-agent package delivery. As well as `drone_rl/ppo.py` and `drone_rl/mappo.py`.  

The current MAPPO fixed-position curriculum can solve many earlier bridge stages, but it is still fragile when the package or deliveries move farther across the pipe map. Some of the issues that I am having: one drone learns to pick up and deliver while the other drone either does not pick up, does not deliver, or follows a stale route from an earlier stage. Training rollouts often show many pickups and deliveries, while deterministic evaluation collapses to only one delivered agent. This could mean that exploration noise is finding useful behavior, but the mean policy has not fully internalized the two-agent coordination pattern.

The fixed curriculum is useful for debugging because it reveals exactly where the policy breaks, but it may also encourage overfitting to one route or one agent assignment. Continuing work should therefore include:

- Returning to MAPPO random-delivery training once the fixed-stage curriculum is stable enough to bootstrap from.
- Comparing fixed delivery, random delivery, and a combination of both.

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/cadengraf/EECS590.git
cd EECS590
python3 -m venv reinforcement_learning 
source reinforcement_learning/bin/activate
pip install -r requirements.txt
```

The classical methods and Deep RL algorithms that don't use drone physics are in `drone_rl/`:

```bash
cd drone_rl
```

Whereas the quad-physics experiments are in `quad_physics/`:

```bash
cd quad_physics
```

## Project Structure

`drone_rl/classical_methods/`
: Classical RL and planning baselines, plus saliency utilities and visual outputs.

`drone_rl/classical_methods/utils/replay.py`
: A simple reusable replay buffer for classical trainers, implemented as a fixed-size circular buffer.

`drone_rl/envs/`
: Shared drone delivery environment used by the Deep RL experiments.

`drone_rl/train_dqn.py`
: Stable-Baselines3 DQN training with curriculum stages and stage-based checkpointing.

`drone_rl/ppo.py`
: Stable-Baselines3 PPO training and evaluation.

`drone_rl/checkpoints/`
: Saved experiment configurations and model checkpoints for DQN and PPO runs.

## Classical Methods

The `drone_rl/classical_methods/` folder contains several tabular and value-based baselines:

- `delivery_markov.py` and related Markov / dynamic-programming scripts: early planning and MDP-style baselines.
- `Q_learning.py`: off-policy Q-learning with a tabular Q-function.
- `monte_carlo.py`: first-visit Monte Carlo control using full-episode returns.
- `sarsa.py`: one-step on-policy SARSA.
- `sarsa_n.py`: n-step SARSA with short multi-step returns.
- `sarsa_backward.py`: backward-view SARSA(lambda) with eligibility traces.
- `sarsa_lambda_forward.py`: forward-view SARSA(lambda) using lambda-returns over stored episodes.
- `tdn_forward.py`: forward-view TD(n) with a state-value function `V(s)`.
- `td_backward.py`: a backward-view TD(lambda)-style value-learning implementation with eligibility traces.
- `td_lambda_forward.py`: forward-view TD(lambda) using truncated lambda-returns for value learning.

These scripts generally:

- build a binary lane map from the pipe/grid utilities,
- define start, package, and delivery positions,
- train an agent with task-specific shaping rewards,
- in selected scripts, store experience tuples `(state, action, reward, next_state, done)` in a simple replay buffer during training,
- roll out the learned policy,
- generate a GIF or visualization of the resulting path,
- and run saliency analysis after training.

If you want short editable descriptions of these algorithms, see [algorithm_summaries.txt](drone_rl/classical_methods/algorithm_summaries.txt).

### Replay Buffer

The classical trainers now share a small replay buffer implementation in [replay.py](drone_rl/classical_methods/utils/replay.py).

The replay buffer: 

- it stores experiences as `(state, action, reward, next_state, done)`,
- it uses a fixed capacity to keep memory bounded,
- it overwrites old entries in a circular manner when full,
- and it supports random batch sampling.

This replay buffer is currently implemented in:

- `Q_learning.py`
- `sarsa.py`
- `monte_carlo.py`
- `sarsa_n.py`

Each of these classical trainers creates `self.replay_buffer` inside the agent class and appends one transition per environment step. This does not change the training update rules yet; it just makes stored experience available for inspection, debugging, or future replay-based experiments.

Not all of these algorithms are off-policy. Q-learning is off-policy, so replay can naturally fit its learning style. SARSA and n-step SARSA are on-policy, and Monte Carlo control is usually treated as on-policy in this project, so their replay buffers are currently for logging/inspection rather than random replay-based updates. Using random replay to update SARSA directly would change the algorithm unless the update was redesigned carefully.

`train_dqn.py` also uses replay, but not through this replay-buffer class. It uses Stable-Baselines3 DQN, which has its own built-in replay buffer because DQN is an off-policy value-based Deep RL algorithm.

## Deep RL Methods

Two main Deep RL approaches are included:

- `train_dqn.py`: DQN training using Stable-Baselines3 with curriculum stages, stage evaluation, and periodic stage checkpoint saves.
- `ppo.py`: PPO training using Stable-Baselines3 with fixed configuration files, evaluation after training, and optional rollout visualization.

At a high level:

- DQN is value-based. It learns Q-values for actions and is a natural fit for discrete movement choices.
- PPO is policy-gradient based. It directly improves a policy while using a value estimate for advantage computation.

## Running the Main Scripts

Classical examples:

```bash
cd drone_rl/classical_methods
python Q_learning.py
python monte_carlo.py
python sarsa.py
python sarsa_n.py
python sarsa_backward.py
python sarsa_lambda_forward.py
python tdn_forward.py
python td_backward.py
python td_lambda_forward.py
```

Deep RL examples:

```bash
cd drone_rl
python train_dqn.py
python ppo.py
```

Inside `quad_physics/`:

```bash
cd quad_physics
python ppo.py
python mappo_fixed.py
python sac.py
python mappo.py
```

Bayesian hyperparameter tuning for fixed-position MAPPO:

```bash
cd quad_physics
python mappo_fixed.py --bayes-tune --tune-trials 20 --tune-storage sqlite:///mappo_fixed_bayes.db
```

To run a full fixed-position MAPPO curriculum with the best parameters after tuning, add `--tune-final-run`.

Resume fixed-position MAPPO from a checkpoint:

```bash
cd quad_physics
python mappo_fixed.py --resume path/to/checkpoint.pt --resume-stage stage_name
```

To auto-resume from the latest safe fixed-position MAPPO checkpoint, run:

```bash
cd quad_physics
python mappo_fixed.py --resume
```

## Saliency Analysis

The classical methods use a shared saliency pipeline in `drone_rl/classical_methods/utils/saliency.py`. After a policy rollout is generated, the scripts call `run_saliency_suite(...)` to create visual explanations of the learned behavior.

The saliency analysis currently includes:

- a visitation heatmap showing which grid cells were visited most often during the rollout,
- an action-preference map for the pre-pickup phase,
- and an action-preference map for the post-pickup phase.

For Q-based agents such as Q-learning and SARSA, the action-preference visualization is built directly from learned Q-values. For value-based TD methods that learn `V(s)` instead of `Q(s, a)`, the saliency utility estimates action preference from successor-state values. Output images are saved under:

`drone_rl/classical_methods/saliency_output/`

This makes the saliency plots useful both for debugging and for explaining what parts of the path or map structure influenced the learned behavior.

## Checkpoints and Saved Models

Checkpoints for Deep RL experiments are stored under:

`drone_rl/checkpoints/dqn/`

`drone_rl/checkpoints/ppo/`

The directory structure is organized by algorithm, task, and run name. For example:

- `drone_rl/checkpoints/dqn/task1/mlp_sparse_curriculum_v2/`
- `drone_rl/checkpoints/ppo/task1/basic_fixed_v2/`

Typical saved files include:

- `config.json`: the experiment configuration used for that run,
- `model_final.zip`: the final saved model after training finishes,
- `model_<stage>_<timesteps>_steps.zip`: intermediate DQN or PPO checkpoints saved during training,
- `best_model.zip`: a best-performing saved PPO checkpoint for runs that track the best model,
- `model.zip`: a generic saved model file for some older runs.

### DQN checkpoint behavior

In `train_dqn.py`, DQN checkpoints are saved periodically during evaluation callbacks. The filename includes both the curriculum stage and the number of steps completed at the time of save, for example:

`model_s4_500000_steps.zip`

The script also writes a `config.json` file for the run and saves a final checkpoint as:

`model_final.zip`

### PPO checkpoint behavior

In `ppo.py`, PPO saves the training configuration to `config.json` at the start of the run and saves the trained model at the end as:

`model_final.zip`

Some older PPO runs in the repository also include:

- `best_model.zip`
- intermediate stage-based checkpoint files

depending on the training script version used for that experiment.
