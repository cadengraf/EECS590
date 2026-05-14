# Version 3 Final Report

Due: May 13, 2026

## Project Goal

The main goal of this project is autonomous package delivery with drones. The environment uses pipe and maze-like lane maps because the drone should not fly directly over residential areas. The lanes represent approved air corridors. A successful policy must stay inside those corridors, reach a package, pick it up, and deliver it to a destination.

The project began as a grid-based drone delivery environment and expanded into quadcopter-style physics. In the current quad-physics experiments, the high-level task is still package delivery, but the policy controls continuous roll and pitch instead of choosing simple grid moves. This makes the problem harder because the agent must learn both movement stability and task completion.

## What I Implemented

### Classical Reinforcement Learning

I implemented several classical methods because they are useful baselines for a discrete grid delivery problem:

- Markov and dynamic-programming style delivery experiments.
- Q-learning.
- Monte Carlo control.
- SARSA.
- n-step SARSA.
- forward and backward TD(lambda)-style methods.
- A shared replay-buffer utility used by several classical trainers.
- Saliency visualization for rollout analysis.

These methods are appropriate for the early grid-world version of the project because the state/action spaces are small enough to inspect directly. They also make it easier to understand whether the map layout, reward structure, and package-delivery objective are behaving correctly before moving to Deep RL.

### Deep Reinforcement Learning

I implemented Deep RL training paths for:

- DQN for discrete grid delivery.
- PPO for single-agent package delivery.
- MAPPO-style training for two-agent package delivery.
- SAC experiments for continuous-control quad-physics delivery.

### Algorithm Choices and Rationale

I chose classical tabular methods first because the original delivery task was discrete and small enough to inspect. Q-learning, SARSA, Monte Carlo, and TD methods are good tools for checking whether the environment dynamics and rewards make sense before adding neural networks. They also provide a clear baseline for path quality and reward behavior.

I chose DQN for the first Deep RL implementation because the original grid environment has discrete action and the Q-Value function would be able to converge to the correct policy. 

I chose PPO for the first policy-gradient implementation because PPO is stable, widely used, and easier to tune than many policy-gradient methods. PPO also made sense once the project started moving toward policies rather than pure value functions.

I chose MAPPO because the package-delivery task eventually became multi-agent. A single-agent method is not enough to study whether multiple drones can coordinate, avoid collisions, and complete assigned deliveries. MAPPO was a good fit because it keeps decentralized action selection for each drone while using a shared training signal.

I chose SAC as a continuing direction because the quad-physics environment uses continuous roll and pitch control. SAC is designed for continuous-control problems and may handle exploration differently from PPO/MAPPO.

### Making `drone_rl/mappo.py` Work

One important V3 step was getting the original non-physics MAPPO implementation in `drone_rl/mappo.py` working before pushing the idea into quad-physics. I built a custom `MultiDroneDeliveryEnv` instead of trying to force the single-agent environment to act multi-agent. The environment gives each drone its own observation, tracks whether each drone has picked up and delivered, handles collisions and swapped positions, and marks success only when all drones have completed their assigned delivery.

The MAPPO model uses a shared actor-critic network. The actor receives each drone's local observation and outputs a categorical distribution over grid actions. The critic receives the flattened global state for all drones, which gives training a team-level view of the episode. This was the key structure that made it closer to MAPPO instead of just running two independent PPO agents.

To make training actually run end-to-end, I added rollout collection, generalized advantage estimation, PPO-style clipped updates, deterministic evaluation, curriculum stages, checkpoint saving, and rollout visualization. The curriculum starts with small fixed T-junction stages and gradually moves toward harder pipe-map stages with random package and delivery settings. This gave me a working two-drone baseline and helped reveal the same kind of coordination issue that later appeared in the quad-physics MAPPO experiments: one drone can learn useful behavior before the full team policy is stable.

The most important V3 work is in `quad_physics/`. This folder contains the current quadcopter-style delivery experiments, including single-agent PPO, two-agent MAPPO, and SAC. I added shared rollout visualization behavior so the quad-physics scripts can produce comparable rollout GIFs and saved rollout images. I also connected the quad-physics visualization path to the existing saliency suite where possible:

```python
from classical_methods.utils.saliency import run_saliency_suite as render_saliency_suite
```

### Curriculum Training and Resume Support

The quad-physics MAPPO curriculum is now broken into many smaller fixed-position stages. This was necessary because jumping directly from short deliveries to far pipe-map deliveries caused the policy to collapse. I added resume support so training can restart from a previous checkpoint or a specific curriculum stage to speed up debugging:

```bash
python mappo_fixed.py --resume path/to/checkpoint.pt --resume-stage stage_name
```

This matters because MAPPO training is expensive, and the later bridge stages can fail after many earlier stages have already succeeded.

## What I Chose Not To Implement

### Full Swarm Coordination

I did not build a full swarm-control system. The project does include a multi-agent MAPPO environment, but the goal is still package delivery through constrained lanes, not emergent swarm behavior. A full swarm implementation would add complexity without directly improving the core delivery objective. For this project, multi-agent coordination is enough to test whether multiple drones can divide the delivery task.

### Isaac Sim or High-Fidelity Robotics Simulation

I did not move the project into Isaac Sim or another high-fidelity robotics simulator. The current simulator is intentionally lightweight so that training iterations are fast enough to debug reward design, curricula, and algorithm choice. Moving to a full robotics simulator would be useful later, but it would distract from the current RL question: can an agent learn package pickup and delivery through constrained corridors?

### Everything-at-Once Algorithm Coverage

I did not try to make every algorithm equally complete. The classical methods are useful baselines, while quad-physics PPO/MAPPO/SAC are the main experimental direction. Keeping every possible algorithm fully polished would make the repository look broader, but less focused. The final version prioritizes the algorithms that best match the current task.

## Bayesian Hyperparameter Tuning

I added Bayesian hyperparameter tuning for fixed-position MAPPO through `quad_physics/mappo_fixed.py --bayes-tune`. The tuning mode uses Optuna's TPE sampler to search over MAPPO parameters such as learning rate, entropy coefficient, PPO clip range, GAE lambda, discount factor, value-loss coefficient, gradient clipping, hidden size, rollout length, minibatch size, epochs, and actor exploration noise.

Because full MAPPO curriculum trials are expensive, the default tuning setup evaluates shorter trials on the early fixed-position curriculum stages. Each trial writes its own checkpoint run, scores the resulting policy using final evaluation metrics and curriculum progress, and saves the best trial configuration under `quad_physics/checkpoints/mappo/tuning/`. After tuning, the best parameters can also be trained on the full curriculum with `--tune-final-run`.

## Current Training Problems

The largest current issue is with fixed-position MAPPO in the later quad-physics pipe stages. The policy can solve many earlier stages, but it often breaks when the package or delivery locations move farther away.

The repeated failure mode is a one-agent local optimum. One drone learns to pick up or deliver, while the other drone does not reliably complete its delivery. Training rollouts may show many pickups and deliveries, but deterministic evaluation often collapses to one delivered agent or zero successful full episodes. This suggests that exploration noise sometimes finds useful behavior, but the learned mean policy has not fully stabilized the two-agent coordination.

Another issue is curriculum sensitivity. If the package location or delivery locations move too far between stages, the policy can forget the previous behavior or overfit to an older route. Fixed stages are useful for debugging because failures are repeatable, but fixed stages may also encourage brittle behavior.

## Continuing Work

The next step is to continue MAPPO with random delivery once fixed-stage training is stable enough to provide a good starting policy. Random delivery is closer to the real package-delivery goal because a useful drone should generalize to multiple destination cells, not only memorize one fixed route.

Future work should compare:

- fixed delivery curriculum,
- random delivery curriculum,
- mixed fixed-to-random curriculum,
- PPO versus MAPPO versus SAC,
- and additional Bayesian hyperparameter tuning once the environment and curriculum are more stable.

SAC is especially worth continuing because the quad-physics task uses continuous control. PPO/MAPPO are useful and already implemented, but SAC may be a better fit for continuous roll/pitch control in some stages.

## Repository Readiness

The repository is organized around the progression of the project:

- `drone_rl/` contains the original grid delivery environment, classical methods, DQN, PPO, and MAPPO experiments.
- `drone_rl/classical_methods/` contains the classical RL baselines and saliency utilities.
- `quad_physics/` contains the current quad-physics experiments.
- `quad_physics/checkpoints/` and `drone_rl/checkpoints/` contain saved experiment outputs and configurations.

The repository still includes checkpoint outputs and training visualizations because they document the actual experiment history. For a public release, some large checkpoint artifacts could be moved out of the repo or replaced with a smaller sample output folder.

## Known Limitations

- Some MAPPO fixed-position stages still fail in deterministic evaluation.
- The later quad-physics curriculum is still being tuned.
- Saliency plots are most informative for tabular or value-based policies. For neural policies, the shared saliency output is useful as rollout/path visualization, but it is not a full neural-network interpretability method.
- The repository contains experiment artifacts from several attempts. These are useful for grading and traceability, but should be cleaned before a polished public release.
