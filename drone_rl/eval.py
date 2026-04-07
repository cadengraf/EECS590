# eval.py — fixed
import numpy as np
from stable_baselines3 import DQN
from envs.drone_env import DroneEnv
from classical_methods.utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions

grid_size = (12, 12)
pg  = PipeGrid(*grid_size)
vis = PipeVisualizerBW(lanes=2, base=3)
bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))

MODEL_PATH = "checkpoints/dqn/task1/mlp_curriculum_v2/best/best_model.zip"

# --- Test 1: curriculum env (same distribution as training) ---
print("=== Eval on CURRICULUM env (training distribution) ===")
env_c = DroneEnv(bw_map, curriculum=True)
env_c.set_curriculum_radius(40)  # hardest level agent trained on
model = DQN.load(MODEL_PATH, env=env_c)

N = 50
completed, got_pkg, steps_list, rewards, won = 0, 0, [], [], []

for ep in range(N):
    obs, _ = env_c.reset()
    total_reward = 0.0
    done = False
    for step in range(1000):  # match training max_steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env_c.step(int(action))
        total_reward += reward
        if obs[6] > 0.5:
            got_pkg += 1
        if done or truncated:
            break
    steps_list.append(step + 1)
    rewards.append(total_reward)
    if done and not truncated:
        completed += 1
        won.append(total_reward)

print(f"Completed:        {completed}/{N} ({100*completed/N:.0f}%)")
print(f"Got package:      {got_pkg}/{N} ({100*got_pkg/N:.0f}%)")
print(f"Avg steps:        {np.mean(steps_list):.1f}")
print(f"Avg reward:       {np.mean(rewards):.1f}")
print(f"Avg reward (won): {np.mean(won):.1f}" if won else "Avg reward (won): N/A")

# --- Test 2: fixed positions (generalization test) ---
print("\n=== Eval on FIXED positions (generalization test) ===")
env_f = DroneEnv(bw_map, curriculum=False)
model_f = DQN.load(MODEL_PATH, env=env_f)

completed, got_pkg, steps_list, rewards, won = 0, 0, [], [], []

for ep in range(N):
    obs, _ = env_f.reset()
    total_reward = 0.0
    done = False
    picked_this_ep = False

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env_c.step(int(action))
        total_reward += reward

        if not picked_this_ep and obs[6] > 0.5:
            got_pkg += 1
            picked_this_ep = True
        if done or truncated:
            break
    steps_list.append(step + 1)
    rewards.append(total_reward)
    if done and not truncated:
        completed += 1
        won.append(total_reward)

print(f"Completed:        {completed}/{N} ({100*completed/N:.0f}%)")
print(f"Got package:      {got_pkg}/{N} ({100*got_pkg/N:.0f}%)")
print(f"Avg steps:        {np.mean(steps_list):.1f}")
print(f"Avg reward:       {np.mean(rewards):.1f}")
print(f"Avg reward (won): {np.mean(won):.1f}" if won else "Avg reward (won): N/A")
print(f"\nFixed positions:")
print(f"  Start:    {env_f.fixed_start}")
print(f"  Package:  {env_f.fixed_package}")
print(f"  Delivery: {env_f.fixed_delivery}")

# --- Action distribution on curriculum ---
print("\n--- Action distribution (50 steps, curriculum env) ---")
obs, _ = env_c.reset()
actions_taken = []
for _ in range(50):
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    actions_taken.append(action)
    obs, _, done, _, _ = env_c.step(action)
    if done:
        break
print(f"Counts: { {a: actions_taken.count(a) for a in range(4)} }")
print(f"  (0=up 1=down 2=left 3=right)")