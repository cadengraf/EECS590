# EECS590

This repository contains a reinforcement learning project focused on grid-based drone delivery. The agent must navigate through pipe-like lane layouts, pick up a package, and deliver it to a goal location. The project includes both classical reinforcement learning methods and Deep RL methods built on top of a shared delivery environment.

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/cadengraf/EECS590.git
cd EECS590
pip install -r requirements.txt
```

Most scripts are run from inside `drone_rl/`:

```bash
cd drone_rl
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
- store experience tuples `(state, action, reward, next_state, done)` in a simple replay buffer during training,
- roll out the learned policy,
- generate a GIF or visualization of the resulting path,
- and run saliency analysis after training.

If you want short editable descriptions of these algorithms, see [algorithm_summaries.txt](/home/megrad/Documents/GitHub/EECS590/drone_rl/classical_methods/algorithm_summaries.txt).

### Replay Buffer

The classical trainers now share a small replay buffer implementation in [replay.py](/home/megrad/Documents/GitHub/EECS590/drone_rl/classical_methods/utils/replay.py).

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

Each of these trainers creates `self.replay_buffer` inside the agent class and appends one transition per environment step. This does not change the training update rules yet; it just makes stored experience available for inspection, debugging, or future replay-based experiments.

## Deep RL Methods

Two main Deep RL approaches are included:

- `train_dqn.py`: DQN training using Stable-Baselines3 with curriculum stages, stage evaluation, and periodic stage checkpoint saves.
- `ppo.py`: PPO training using Stable-Baselines3 with fixed configuration files, evaluation after training, and optional rollout visualization.

At a high level:

- DQN is value-based. It learns Q-values for actions and is a natural fit for discrete movement choices.
- PPO is policy-gradient based. It directly improves a policy while using a value estimate for advantage computation.

Current status notes:

- The DQN pipeline currently works best for fixed environments and fixed-map training setups.
- PPO is still under development. The long-term goal is to move PPO toward a multi-agent setting, so its current implementation should be treated as an ongoing experiment rather than the final intended direction.

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

## Notes

- The classical scripts are mostly self-contained and define their own reward shaping and update rules.
- The Deep RL scripts rely on the shared environment in `drone_rl/envs/drone_env.py`.
- DQN should currently be viewed as the fixed-environment Deep RL baseline.
- Still working on PPO and will try MAPPO when switch to multiple agents
