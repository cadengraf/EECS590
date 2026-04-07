import os
import json
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
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
            print(f"\n[Q-diag {self.num_timesteps}] Q={np.round(q_vals, 3)}")
        return True


ACTION_NAMES = ["up", "down", "left", "right"]


def build_bw_map(grid_size, loop_prob=0.25):
    pipe_grid = PipeGrid(grid_size[0], grid_size[1], loop_prob=loop_prob)
    pipe_vis = PipeVisualizerBW(lanes=2, base=3)
    return pipe_vis.render(pipe_grid.to_pipe_ids(PipeOptions()))



def build_env(stage, print_freq=0):
    bw_map = build_bw_map(stage["grid_size"], loop_prob=stage.get("loop_prob", 0.25))
    return DroneEnv(
        bw_map,
        max_steps=stage["max_steps"],
        print_freq=print_freq,
    )



def clone_env(source_env, print_freq=0):
    env = DroneEnv(source_env.bw_map.copy(), max_steps=source_env.max_steps, print_freq=print_freq)
    env.fixed_start = source_env.fixed_start
    env.fixed_package = source_env.fixed_package
    env.fixed_delivery = source_env.fixed_delivery
    return env



def describe_env(tag, env):
    print(f"{tag} observation space: {env.observation_space}")
    print(f"{tag} grid shape:        {env.grid_shape}")
    print(f"{tag} start:             {env.fixed_start}")
    print(f"{tag} package:           {env.fixed_package}")
    print(f"{tag} delivery:          {env.fixed_delivery}")
    print(f"{tag} lane cells:        {len(env.lane_coords)}")



def configure_stage_exploration(model, stage, config):
    exploration_initial_eps = stage.get(
        "exploration_initial_eps",
        config["stage_exploration_initial_eps"],
    )
    exploration_final_eps = stage.get(
        "exploration_final_eps",
        config["stage_exploration_final_eps"],
    )
    exploration_fraction = stage.get(
        "exploration_fraction",
        config["stage_exploration_fraction"],
    )

    model.exploration_initial_eps = exploration_initial_eps
    model.exploration_final_eps = exploration_final_eps
    model.exploration_fraction = exploration_fraction
    model.exploration_rate = exploration_initial_eps

    print(
        "Stage exploration: "
        f"initial={exploration_initial_eps:.2f} | "
        f"final={exploration_final_eps:.2f} | "
        f"fraction={exploration_fraction:.2f}"
    )



def evaluate_stage(model, source_env, n_eval_episodes):
    eval_env = clone_env(source_env, print_freq=0)
    rewards = []
    lengths = []
    successes = 0
    pickups = 0

    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0
        picked_up = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = eval_env.step(action)
            episode_reward += reward
            episode_steps += 1
            picked_up = picked_up or bool(eval_env.state[2])

        rewards.append(episode_reward)
        lengths.append(episode_steps)
        pickups += int(picked_up)

        reached_delivery = tuple(eval_env.state[:2]) == tuple(eval_env.fixed_delivery)
        delivered = done and reached_delivery and bool(eval_env.state[2])
        successes += int(delivered)

    success_rate = successes / n_eval_episodes
    pickup_rate = pickups / n_eval_episodes
    mean_reward = float(np.mean(rewards))
    mean_length = float(np.mean(lengths))
    return success_rate, pickup_rate, mean_reward, mean_length


class StagePromotionCallback(BaseCallback):
    def __init__(
        self,
        stage_name,
        source_env,
        eval_every,
        n_eval_episodes,
        promotion_threshold,
        min_timesteps_before_promotion,
        save_dir,
    ):
        super().__init__()
        self.stage_name = stage_name
        self.source_env = source_env
        self.eval_every = eval_every
        self.n_eval_episodes = n_eval_episodes
        self.promotion_threshold = promotion_threshold
        self.min_timesteps_before_promotion = min_timesteps_before_promotion
        self.save_dir = save_dir
        self.promoted = False
        self.last_eval = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every != 0:
            return True

        success_rate, pickup_rate, mean_reward, mean_length = evaluate_stage(
            self.model,
            self.source_env,
            self.n_eval_episodes,
        )
        self.last_eval = {
            "success_rate": success_rate,
            "pickup_rate": pickup_rate,
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "timesteps": self.num_timesteps,
        }
        print(
            f"Stage eval after {self.num_timesteps:>6} steps | "
            f"success_rate={success_rate:.2%} | "
            f"pickup_rate={pickup_rate:.2%} | "
            f"mean_reward={mean_reward:.2f} | "
            f"mean_length={mean_length:.1f}"
        )

        stage_model_path = os.path.join(
            self.save_dir,
            f"model_{self.stage_name}_{self.num_timesteps}_steps",
        )
        self.model.save(stage_model_path)

        if (
            self.num_timesteps >= self.min_timesteps_before_promotion
            and success_rate >= self.promotion_threshold
        ):
            print(
                f"Promoting from {self.stage_name} early: success rate {success_rate:.2%} "
                f">= threshold {self.promotion_threshold:.2%}"
            )
            self.promoted = True
            return False

        return True



def visualize_trained_policy(model, source_env, delay=0.12):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rollout_env = clone_env(source_env, print_freq=0)

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
    print("  Visualization used the final curriculum stage map and positions.")

    plt.ioff()
    plt.show()


run_name = "mlp_sparse_curriculum_v1"
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
    "viz_delay": 0.12,
    "progress_print_freq": 5000,
    "qdiag_freq": 20_000,
    "env_print_freq": 20_000,
    "stage_eval_episodes": 20,
    "stage_eval_every": 25_000,
    "stage_promotion_success_rate": 0.70,
    "stage_min_timesteps_before_promotion": 50_000,
    "stage_exploration_initial_eps": 1.0,
    "stage_exploration_final_eps": 0.10,
    "stage_exploration_fraction": 0.40,
    "curriculum": [
        {"name": "stage_1", "grid_size": [4, 4], "max_steps": 120, "timesteps": 150_000, "exploration_initial_eps": 1.0, "exploration_fraction": 0.35},
        {"name": "stage_2", "grid_size": [6, 6], "max_steps": 250, "timesteps": 250_000, "exploration_initial_eps": 0.8, "exploration_fraction": 0.40},
        {"name": "stage_3", "grid_size": [8, 8], "max_steps": 500, "timesteps": 400_000, "exploration_initial_eps": 0.8, "exploration_fraction": 0.45},
        {"name": "stage_4", "grid_size": [10, 10], "max_steps": 1000, "timesteps": 550_000, "exploration_initial_eps": 1.0, "exploration_fraction": 0.60, "min_timesteps_before_promotion": 75_000},
        {"name": "stage_5", "grid_size": [12, 12], "max_steps": 2000, "timesteps": 750_000, "exploration_initial_eps": 1.0, "exploration_fraction": 0.60, "min_timesteps_before_promotion": 100_000},
    ],
}
config["timesteps"] = sum(stage["timesteps"] for stage in config["curriculum"])

with open(f"{save_dir}/config.json", "w") as f:
    json.dump(config, f, indent=4)

first_stage = config["curriculum"][0]
train_env = build_env(first_stage, print_freq=config["env_print_freq"])
describe_env("Initial", train_env)

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

final_stage_env = train_env
for stage_idx, stage in enumerate(config["curriculum"], start=1):
    print(f"\n=== Curriculum Stage {stage_idx}/{len(config['curriculum'])}: {stage['name']} ===")
    stage_train_env = build_env(stage, print_freq=config["env_print_freq"])
    describe_env("Stage", stage_train_env)
    print(f"Stage max timesteps: {stage['timesteps']}")

    model.set_env(stage_train_env)
    configure_stage_exploration(model, stage, config)

    progress_cb = ProgressCallback(print_freq=config["progress_print_freq"])
    qdiag_cb = QValueDiagnosticCallback(stage_train_env, check_freq=config["qdiag_freq"])
    promotion_cb = StagePromotionCallback(
        stage_name=stage["name"],
        source_env=stage_train_env,
        eval_every=stage.get("eval_every", config["stage_eval_every"]),
        n_eval_episodes=stage.get("eval_episodes", config["stage_eval_episodes"]),
        promotion_threshold=stage.get("promotion_success_rate", config["stage_promotion_success_rate"]),
        min_timesteps_before_promotion=stage.get(
            "min_timesteps_before_promotion",
            config["stage_min_timesteps_before_promotion"],
        ),
        save_dir=save_dir,
    )

    model.learn(
        total_timesteps=stage["timesteps"],
        reset_num_timesteps=True,
        callback=[progress_cb, qdiag_cb, promotion_cb],
    )

    if not promotion_cb.promoted:
        print(f"Completed full budget for {stage['name']} without early promotion.")

    final_stage_env = stage_train_env

model.save(f"{save_dir}/model_final")
print("Training complete!")
visualize_trained_policy(model, final_stage_env, delay=config["viz_delay"])
