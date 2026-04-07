import numpy as np
import random
from envs.drone_env import DroneEnv
from classical_methods.utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions

# --- Build map ---
grid_size = (12, 12)
pg  = PipeGrid(*grid_size)
vis = PipeVisualizerBW(lanes=2, base=3)
bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))

# --- Fixed positions sanity check ---
print("=== Fixed positions (curriculum=False) ===")
env = DroneEnv(bw_map, curriculum=False)
obs, _ = env.reset()
print(f"Start:      {env.fixed_start}")
print(f"Package:    {env.fixed_package}")
print(f"Delivery:   {env.fixed_delivery}")
print(f"Lane cells: {len(env.lane_coords)}")
print(f"Grid shape: {env.grid_shape}")
print(f"Obs:        {obs}")

got_pkg, delivered = 0, 0
for ep in range(200):
    obs, _ = env.reset()
    for _ in range(500):
        a = env.action_space.sample()
        obs, r, done, trunc, _ = env.step(a)
        if obs[6] > 0.5:
            got_pkg += 1
        if done:
            delivered += 1
            break

print(f"\nRandom policy (fixed) over 200 episodes:")
print(f"  Got package: {got_pkg}")
print(f"  Delivered:   {delivered}")
print(f"  {'OK' if delivered > 0 else 'WARNING: delivery never reached — positions too far apart'}")

# --- Curriculum radius sweep ---
print("\n=== Curriculum radius sweep ===")
for radius in [15, 25, 40, 60, 80]:
    env_c = DroneEnv(bw_map, curriculum=True)
    env_c.set_curriculum_radius(radius)
    got_pkg, delivered = 0, 0
    for ep in range(200):
        obs, _ = env_c.reset()
        for _ in range(500):
            a = env_c.action_space.sample()
            obs, r, done, trunc, _ = env_c.step(a)
            if obs[6] > 0.5:
                got_pkg += 1
            if done:
                delivered += 1
                break
    print(f"  radius={radius:>3} | got_package={got_pkg:>3} | delivered={delivered:>3} "
          f"{'← start here' if delivered > 0 and radius == min([r for r in [15,25,40,60,80] if delivered > 0]) else ''}")

print("\nUse the smallest radius where delivered > 0 as your curriculum start_radius in train.py")