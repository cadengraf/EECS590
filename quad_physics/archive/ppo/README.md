# Quad Physics PPO

This is a separate PPO experiment folder for the same package-pickup and delivery task used by `drone_rl`, but with a simplified quadcopter dynamics layer added to the environment.

The task is still grid based:

- the map is a binary lane map,
- the drone starts at a lane cell,
- it must pick up the package,
- then it must reach the delivery cell,
- PPO still uses four high-level actions: north, south, west, and east.

The difference is that an action no longer teleports the drone by one grid cell. Each action commands a target roll/pitch angle, and the environment integrates position, velocity, altitude, attitude, and angular-rate state over several physics substeps. Rewards are still issued from the familiar grid delivery task when the physical quadcopter crosses into pickup or delivery cells.

## Files

- `envs/quadcopter_drone_env.py`: Gymnasium environment with quadcopter-inspired physics.
- `ppo.py`: PPO curriculum training, evaluation, checkpointing, and rollout visualization.
- `ppo_config.py`: training, reward, physics, and curriculum settings.
- `utils/pipes.py`: local copy of the pipe-lane map generator so this folder does not import from `drone_rl`.

## Run

From the repo root:

```bash
./reinforcement_learning/bin/python -m quad_physics.ppo
```

Or from this folder:

```bash
cd quad_physics
../reinforcement_learning/bin/python ppo.py
```

Checkpoints are saved inside:

```text
quad_physics/checkpoints/ppo/task1/quad_physics_curriculum_v1/
```

## Quick Smoke Test

```bash
./reinforcement_learning/bin/python - <<'PY'
from stable_baselines3.common.env_checker import check_env
from quad_physics.ppo import build_env
from quad_physics.ppo_config import config

stage = {**config, **config["curriculum"][0]}
env = build_env(stage)
check_env(env, warn=True)
print("quad physics env ok")
PY
```

