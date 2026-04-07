import os
import json
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.drone_env import DroneEnv
from classical_methods.utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions


class ProgressCallback(BaseCallback):
    def __init__(self, print_freq=5000):
        super().__init__()
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"  Timestep {self.num_timesteps:>7} | "
                  f"Episodes: {self.model._episode_num:>5} | "
                  f"Exploration eps: {self.model.exploration_rate:.3f}")
        return True


class QValueDiagnosticCallback(BaseCallback):
    def __init__(self, env, check_freq=20_000):
        super().__init__()
        self.diag_env = env
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            import torch
            obs, _ = self.diag_env.reset()
            obs_tensor = np.array([obs], dtype=np.float32)
            obs_tensor = torch.FloatTensor(obs_tensor).to(self.model.device)
            with torch.no_grad():
                q_vals = self.model.q_net(obs_tensor).cpu().numpy()[0]
            print(f"\n[Q-diag {self.num_timesteps}] "
                  f"Q={np.round(q_vals, 3)}  "
                  f"std={q_vals.std():.4f}  "
                  f"argmax={q_vals.argmax()} (0=up 1=down 2=left 3=right)")
        return True


ACTION_NAMES = ["up", "down", "left", "right"]


def visualize_trained_policy(model, source_env, delay=0.12):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rollout_env = DroneEnv(source_env.bw_map, max_steps=source_env.max_steps)
    rollout_env.fixed_start = source_env.fixed_start
    rollout_env.fixed_package = source_env.fixed_package
    rollout_env.fixed_delivery = source_env.fixed_delivery

    obs, _ = rollout_env.reset()
    trajectory = [(rollout_env.state[0], rollout_env.state[1])]
    total_reward = 0.0
    steps = 0
    done = False
    truncated = False

    cmap = ListedColormap(["#1a1a1a", "#f1efe8"])
    fig, (ax_grid, ax_info) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [2.2, 1]}
    )
    fig.tight_layout(pad=2.0)

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = rollout_env.step(action)
        total_reward += reward
        steps += 1

        r, c, has_pkg = rollout_env.state
        trajectory.append((r, c))

        ax_grid.clear()
        ax_info.clear()

        ax_grid.imshow(rollout_env.bw_map, cmap=cmap, origin="upper", vmin=0, vmax=1)
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        ax_grid.set_title(
            f"Trained Policy Rollout | Step {steps}",
            fontsize=13,
            pad=10,
        )

        if len(trajectory) > 1:
            cols = [pos[1] for pos in trajectory]
            rows = [pos[0] for pos in trajectory]
            ax_grid.plot(cols, rows, color="#3f88c5", linewidth=2.0, alpha=0.80)

        start_r, start_c = rollout_env.fixed_start
        pkg_r, pkg_c = rollout_env.fixed_package
        dst_r, dst_c = rollout_env.fixed_delivery
        ax_grid.scatter(start_c, start_r, s=80, color="#5b8c5a", zorder=4)
        if not has_pkg:
            ax_grid.scatter(pkg_c, pkg_r, marker="s", s=120, color="#f2c14e", zorder=5)
        ax_grid.scatter(dst_c, dst_r, marker="*", s=180, color="#ef476f", zorder=5)

        drone_color = "#ff7f11" if has_pkg else "#118ab2"
        if done:
            drone_color = "#06d6a0"
        elif truncated:
            drone_color = "#8338ec"
        ax_grid.scatter(c, r, s=140, color=drone_color, zorder=6)

        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")
        ax_info.text(0.5, 0.96, "Rollout Info", ha="center", va="top",
                     fontsize=14, fontweight="bold")

        current_goal = rollout_env.fixed_delivery if has_pkg else rollout_env.fixed_package
        status = "delivered" if done else "truncated" if truncated else "running"
        lines = [
            ("action", ACTION_NAMES[int(action)]),
            ("step reward", f"{reward:+.2f}"),
            ("total reward", f"{total_reward:+.2f}"),
            ("position", f"({r}, {c})"),
            ("carrying", "yes" if has_pkg else "no"),
            ("goal", str(current_goal)),
            ("start", str(rollout_env.fixed_start)),
            ("package", str(rollout_env.fixed_package)),
            ("delivery", str(rollout_env.fixed_delivery)),
            ("status", status),
        ]

        y = 0.86
        for key, value in lines:
            ax_info.text(0.08, y, key, ha="left", va="top", fontsize=10, color="#666666")
            ax_info.text(0.92, y, value, ha="right", va="top", fontsize=10, color="#111111")
            y -= 0.085

        plt.pause(delay)

    outcome = "DELIVERED" if done else "TRUNCATED"
    print(f"\nPost-training rollout: {outcome}")
    print(f"  Steps:        {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final state:  {rollout_env.state}")
    print("  Visualization used the exact training map and fixed positions.")

    plt.ioff()
    plt.show()


# --- Build map ---
grid_size = (12, 12)
pg = PipeGrid(*grid_size)
vis = PipeVisualizerBW(lanes=2, base=3)
bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))

# --- Envs ---
train_env = DroneEnv(bw_map, print_freq=100_000)
eval_env_raw = DroneEnv(bw_map)
eval_env = DummyVecEnv([lambda: Monitor(eval_env_raw)])

print("Observation space:", train_env.observation_space)
print(f"Start:      {train_env.fixed_start}")
print(f"Package:    {train_env.fixed_package}")
print(f"Delivery:   {train_env.fixed_delivery}")
print(f"Lane cells: {len(train_env.lane_coords)}")

# --- Config ---
run_name = "mlp_sparse_v1"
save_dir = f"checkpoints/dqn/task1/{run_name}"
os.makedirs(save_dir, exist_ok=True)

config = {
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "batch_size": 128,
    "buffer_size": 100_000,
    "learning_starts": 20_000,
    "train_freq": 4,
    "target_update_interval": 2000,
    "exploration_fraction": 0.4,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.1,
    "timesteps": 2_000_000,
    "viz_delay": 0.12,
}

with open(f"{save_dir}/config.json", "w") as f:
    json.dump(config, f, indent=4)

# --- Callbacks ---
checkpoint_cb = CheckpointCallback(
    save_freq=100_000,
    save_path=save_dir,
    name_prefix="model"
)
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=f"{save_dir}/best",
    log_path=f"{save_dir}/logs",
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1
)
qdiag_cb = QValueDiagnosticCallback(train_env, check_freq=20_000)

# --- Model ---
model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=config["learning_rate"],
    gamma=config["gamma"],
    batch_size=config["batch_size"],
    buffer_size=config["buffer_size"],
    learning_starts=config["learning_starts"],
    train_freq=config["train_freq"],
    target_update_interval=config["target_update_interval"],
    exploration_fraction=config["exploration_fraction"],
    exploration_initial_eps=config["exploration_initial_eps"],
    exploration_final_eps=config["exploration_final_eps"],
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=0
)

model.learn(
    total_timesteps=config["timesteps"],
    callback=[checkpoint_cb, eval_cb, qdiag_cb,
              ProgressCallback(print_freq=5000)]
)

model.save(f"{save_dir}/model_final")
print("Training complete!")
visualize_trained_policy(model, train_env, delay=config["viz_delay"])
