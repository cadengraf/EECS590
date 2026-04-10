import os
import json
from collections import deque
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from envs.drone_env import DroneEnv
from classical_methods.utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions

ACTION_NAMES = ["up", "down", "left", "right"]


# ── Env helpers ───────────────────────────────────────────────────────────────

def build_bw_map(grid_size, loop_prob=0.25, seed=None):
    np_state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(seed)
        return PipeVisualizerBW(lanes=2, base=3).render(
            PipeGrid(grid_size[0], grid_size[1], loop_prob=loop_prob).to_pipe_ids(PipeOptions())
        )
    finally:
        np.random.set_state(np_state)


def shortest_path_distances(env, start):
    queue = deque([(start, 0)])
    distances = {start: 0}
    while queue:
        (r, c), dist = queue.popleft()
        for dr, dc in env.actions:
            nxt = (r + dr, c + dc)
            if nxt in env.lane_set and nxt not in distances:
                distances[nxt] = dist + 1
                queue.append((nxt, dist + 1))
    return distances


def pick_position_by_distance(candidates, min_dist, max_dist):
    valid = [pos for pos, dist in candidates if min_dist <= dist <= max_dist]
    if valid:
        return valid[-1]
    farther = [pos for pos, dist in candidates if dist >= min_dist]
    if farther:
        return farther[-1]
    return candidates[-1][0]


def configure_stage_positions(env, stage):
    if not stage.get("placement_strategy"):
        return

    start = min(env.lane_coords, key=lambda pos: (pos[0] + pos[1], pos[0], pos[1]))
    start_dists = shortest_path_distances(env, start)
    reachable_from_start = sorted(
        ((pos, dist) for pos, dist in start_dists.items() if pos != start),
        key=lambda item: (item[1], item[0][0], item[0][1]),
    )
    if not reachable_from_start:
        raise ValueError(f"No reachable lane cells found for stage {stage['name']}")

    pkg = pick_position_by_distance(
        reachable_from_start,
        stage.get("package_min_dist", 1),
        stage.get("package_max_dist", 9999),
    )

    pkg_dists = shortest_path_distances(env, pkg)
    delivery_candidates = sorted(
        (
            (pos, dist) for pos, dist in pkg_dists.items()
            if pos != start and pos != pkg
            and start_dists.get(pos, -1) >= stage.get("delivery_min_start_dist", 0)
        ),
        key=lambda item: (item[1], start_dists.get(item[0], 0), item[0][0], item[0][1]),
    )
    if not delivery_candidates:
        delivery_candidates = sorted(
            ((pos, dist) for pos, dist in pkg_dists.items() if pos != start and pos != pkg),
            key=lambda item: (item[1], start_dists.get(item[0], 0), item[0][0], item[0][1]),
        )

    delivery = pick_position_by_distance(
        delivery_candidates,
        stage.get("delivery_min_dist", 1),
        stage.get("delivery_max_dist", 9999),
    )

    env.fixed_start = start
    env.fixed_package = pkg
    env.fixed_delivery = delivery


def build_env(stage, print_freq=0):
    env = DroneEnv(
        build_bw_map(stage["grid_size"], stage.get("loop_prob", 0.25), seed=stage.get("map_seed")),
        max_steps=stage["max_steps"],
        print_freq=print_freq,
        progress_reward_scale=stage.get("progress_reward_scale", 0.3),
        randomize_package=stage.get("randomize_package", False),
        randomize_delivery=stage.get("randomize_delivery", False),
    )
    configure_stage_positions(env, stage)
    return env


def clone_env(src, print_freq=0, randomize_package=None, randomize_delivery=None):
    env = DroneEnv(
        src.bw_map.copy(),
        max_steps=src.max_steps,
        print_freq=print_freq,
        progress_reward_scale=src.progress_reward_scale,
        randomize_package=src.randomize_package if randomize_package is None else randomize_package,
        randomize_delivery=src.randomize_delivery if randomize_delivery is None else randomize_delivery,
    )
    env.fixed_start = src.fixed_start
    env.fixed_package = src.fixed_package
    env.fixed_delivery = src.fixed_delivery
    return env


def describe_env(tag, env):
    print(
        f"{tag}: obs={env.observation_space} grid={env.grid_shape} "
        f"start={env.fixed_start} pkg={env.fixed_package} "
        f"delivery={env.fixed_delivery} lanes={len(env.lane_coords)} "
        f"rand_pkg={env.randomize_package} rand_del={env.randomize_delivery}"
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_stage(model, src_env, n_episodes):
    eval_env = clone_env(src_env, randomize_package=src_env.randomize_package,
                         randomize_delivery=src_env.randomize_delivery)
    rewards, lengths, successes, pickups = [], [], 0, 0

    for i in range(n_episodes):
        obs, _ = eval_env.reset(seed=i)
        done = truncated = False
        ep_reward, steps, picked_up = 0.0, 0, False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = eval_env.step(action)
            ep_reward += reward
            steps += 1
            picked_up = picked_up or bool(eval_env.state[2])

        rewards.append(ep_reward)
        lengths.append(steps)
        pickups += int(picked_up)
        delivered = (
            done
            and tuple(eval_env.state[:2]) == tuple(eval_env.fixed_delivery)
            and bool(eval_env.state[2])
        )
        successes += int(delivered)

    return successes / n_episodes, pickups / n_episodes, float(np.mean(rewards)), float(np.mean(lengths))


# ── Callbacks ─────────────────────────────────────────────────────────────────

class ProgressCallback(BaseCallback):
    def __init__(self, print_freq=5000):
        super().__init__()
        self.print_freq = print_freq

    def _on_step(self):
        if self.num_timesteps % self.print_freq == 0:
            print(f"  Step {self.num_timesteps:>7} | Episodes: {self.model._episode_num:>5} | eps: {self.model.exploration_rate:.3f}")
        return True


class QValueDiagnosticCallback(BaseCallback):
    def __init__(self, env, check_freq=20_000):
        super().__init__()
        self.diag_env = env
        self.check_freq = check_freq

    def _on_step(self):
        if self.num_timesteps % self.check_freq == 0:
            obs, _ = self.diag_env.reset()
            obs_t = torch.as_tensor(np.array([obs], dtype=np.float32), device=self.model.device)
            with torch.no_grad():
                q = self.model.q_net(obs_t).cpu().numpy()[0]
            print(f"\n[Q-diag {self.num_timesteps}] Q={np.round(q, 3)}")
        return True


class StagePromotionCallback(BaseCallback):
    def __init__(self, stage_name, source_env, eval_every, n_eval_episodes,
                 promotion_threshold, min_timesteps_before_promotion, save_dir):
        super().__init__()
        self.stage_name = stage_name
        self.source_env = source_env
        self.eval_every = eval_every
        self.n_eval_episodes = n_eval_episodes
        self.promotion_threshold = promotion_threshold
        self.min_timesteps = min_timesteps_before_promotion
        self.save_dir = save_dir
        self.promoted = False

    def _on_step(self):
        if self.num_timesteps % self.eval_every != 0:
            return True

        sr, pr, mr, ml = evaluate_stage(self.model, self.source_env, self.n_eval_episodes)
        print(f"Eval @{self.num_timesteps:>6} | sr={sr:.2%} pr={pr:.2%} reward={mr:.2f} len={ml:.1f}")
        self.model.save(os.path.join(self.save_dir, f"model_{self.stage_name}_{self.num_timesteps}_steps"))

        if self.num_timesteps >= self.min_timesteps and sr >= self.promotion_threshold:
            print(f"Early promotion from {self.stage_name}: {sr:.2%} >= {self.promotion_threshold:.2%}")
            self.promoted = True
            return False
        return True


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_trained_policy(model, src_env, delay=0.12, save_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.animation import FuncAnimation

    env = clone_env(src_env, randomize_package=False, randomize_delivery=False)
    obs, _ = env.reset()
    trajectory = [(env.state[0], env.state[1])]
    total_reward, steps = 0.0, 0
    done = truncated = False

    cmap = ListedColormap(["#1a1a1a", "#f1efe8"])
    fig, (ax_g, ax_i) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.tight_layout(pad=2.0)

    def update(frame):
        nonlocal trajectory, total_reward, steps, done, truncated
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        r, c, has_pkg = env.state
        trajectory.append((r, c))

        ax_g.clear()
        ax_i.clear()
        ax_g.imshow(env.bw_map, cmap=cmap, origin="upper", vmin=0, vmax=1)
        ax_g.set(xticks=[], yticks=[], title=f"Policy Rollout | Step {steps}")

        if len(trajectory) > 1:
            ax_g.plot([p[1] for p in trajectory], [p[0] for p in trajectory],
                      color="#3f88c5", linewidth=2.0, alpha=0.8)

        sr, sc = env.fixed_start
        pr, pc = env.fixed_package
        dr, dc = env.fixed_delivery
        ax_g.scatter(sc, sr, s=80, color="#5b8c5a", zorder=4)
        if not has_pkg:
            ax_g.scatter(pc, pr, marker="s", s=120, color="#f2c14e", zorder=5)
        ax_g.scatter(dc, dr, marker="*", s=180, color="#ef476f", zorder=5)

        drone_color = "#06d6a0" if done else "#8338ec" if truncated else "#ff7f11" if has_pkg else "#118ab2"
        ax_g.scatter(c, r, s=140, color=drone_color, zorder=6)

        ax_i.set(xlim=(0, 1), ylim=(0, 1))
        ax_i.axis("off")
        ax_i.text(0.5, 0.96, "Rollout Info", ha="center", va="top", fontsize=14, fontweight="bold")
        goal = env.fixed_delivery if has_pkg else env.fixed_package
        lines = [
            ("action", ACTION_NAMES[int(action)]),
            ("step reward", f"{reward:+.2f}"),
            ("total reward", f"{total_reward:+.2f}"),
            ("position", f"({r}, {c})"),
            ("carrying", "yes" if has_pkg else "no"),
            ("goal", str(goal)),
            ("start", str(env.fixed_start)),
            ("package", str(env.fixed_package)),
            ("delivery", str(env.fixed_delivery)),
            ("status", "delivered" if done else "truncated" if truncated else "running"),
        ]
        for y, (k, v) in zip(np.linspace(0.86, 0.01, len(lines)), lines):
            ax_i.text(0.08, y, k, ha="left", va="top", fontsize=10, color="#666")
            ax_i.text(0.92, y, v, ha="right", va="top", fontsize=10, color="#111")

    if save_path is not None:
        anim = FuncAnimation(fig, update, frames=range(steps + 1), interval=delay * 1000, repeat=False)
        plt.ioff()
        anim.save(save_path, writer="ffmpeg")


# ── Config ────────────────────────────────────────────────────────────────────

run_name = "mlp_sparse_curriculum_v2"
save_dir = f"checkpoints/dqn/task1/{run_name}"
os.makedirs(save_dir, exist_ok=True)

config = {
    # Model defaults (overridable per stage)
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "batch_size": 128,
    "buffer_size": 100_000,
    "learning_starts": 20_000,
    "train_freq": 4,
    "target_update_interval": 2000,
    # Exploration defaults (overridable per stage)
    "stage_exploration_initial_eps": 1.0,
    "stage_exploration_final_eps": 0.10,
    "stage_exploration_fraction": 0.40,
    # Logging / eval
    "viz_delay": 0.12,
    "progress_print_freq": 5000,
    "qdiag_freq": 20_000,
    "env_print_freq": 20_000,
    "stage_eval_episodes": 20,
    "stage_eval_every": 25_000,
    "stage_promotion_success_rate": 0.70,
    "stage_min_timesteps_before_promotion": 50_000,
    "curriculum": [
        {
            "name": "s1",
            "grid_size": [4, 4],
            "max_steps": 250,
            "timesteps": 250_000,
            "map_seed": 101,
            "randomize_package": False,
            "randomize_delivery": False,
            "placement_strategy": "distance_bands",
            "package_min_dist": 5,  "package_max_dist": 10,
            "delivery_min_dist": 4, "delivery_max_dist": 9,
            "delivery_min_start_dist": 8,
            "exploration_initial_eps": 1.0,
            "exploration_fraction": 0.35,
        },
        {
            "name": "s2",
            "grid_size": [6, 6],
            "max_steps": 400,
            "timesteps": 400_000,
            "map_seed": 202,
            "randomize_package": False,
            "randomize_delivery": False,
            "placement_strategy": "distance_bands",
            "package_min_dist": 8,  "package_max_dist": 16,
            "delivery_min_dist": 7, "delivery_max_dist": 16,
            "delivery_min_start_dist": 14,
            "exploration_initial_eps": 0.8,
            "exploration_fraction": 0.40,
        },
        {
            "name": "s3",
            "grid_size": [6, 6],
            "max_steps": 500,
            "timesteps": 750_000,
            "map_seed": 202,
            "randomize_package": True,
            "randomize_delivery": False,
            "placement_strategy": "distance_bands",
            "package_min_dist": 8,  "package_max_dist": 16,
            "delivery_min_dist": 7, "delivery_max_dist": 16,
            "delivery_min_start_dist": 14,
            "exploration_initial_eps": 0.5,
            "exploration_fraction": 0.40,
            "min_timesteps_before_promotion": 100_000,
            "clear_buffer_on_start": True,
            "learning_starts": 20_000,
        },
        {
            "name": "s4",
            "grid_size": [8, 8],
            "max_steps": 1500,          
            "timesteps": 1_250_000,
            "map_seed": 303,
            "randomize_package": True,
            "randomize_delivery": False,
            "placement_strategy": "distance_bands",
            "package_min_dist": 10, "package_max_dist": 20,
            "delivery_min_dist": 8, "delivery_max_dist": 18,
            "delivery_min_start_dist": 16,
            "exploration_initial_eps": 0.30,
            "exploration_fraction": 0.30,  
            "exploration_final_eps": 0.05,
            "min_timesteps_before_promotion": 150_000,
            "clear_buffer_on_start": True,  
            "learning_starts": 30_000,     
        },
        {
            "name": "s5",
            "grid_size": [10, 10],
            "max_steps": 3000,
            "timesteps": 2_000_000,
            "map_seed": 404,
            "randomize_package": True,
            "randomize_delivery": True,
            "placement_strategy": "distance_bands",
            "package_min_dist": 14, "package_max_dist": 28,
            "delivery_min_dist": 12,"delivery_max_dist": 28,
            "delivery_min_start_dist": 22,
            "exploration_initial_eps": 0.40, # warm start; agent knows the task
            "exploration_fraction": 0.50,
            "exploration_final_eps": 0.05,
            "min_timesteps_before_promotion": 200_000,
            "clear_buffer_on_start": True,
            "learning_starts": 30_000,
        },
        {
            "name": "s6",
            "grid_size": [12, 12],
            "max_steps": 6000,
            "timesteps": 2_500_000,
            "map_seed": 505,
            "randomize_package": True,
            "randomize_delivery": True,
            "placement_strategy": "distance_bands",
            "package_min_dist": 18, "package_max_dist": 36,
            "delivery_min_dist": 16,"delivery_max_dist": 36,
            "delivery_min_start_dist": 28,
            "exploration_initial_eps": 0.30,
            "exploration_fraction": 0.40,
            "exploration_final_eps": 0.05,
            "min_timesteps_before_promotion": 250_000,
            "clear_buffer_on_start": True,
            "learning_starts": 30_000,
        },
    ],
}
config["timesteps"] = sum(s["timesteps"] for s in config["curriculum"])

with open(f"{save_dir}/config.json", "w") as f:
    json.dump(config, f, indent=4)


# ── Model init ────────────────────────────────────────────────────────────────

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
    exploration_fraction=config["stage_exploration_fraction"],
    exploration_initial_eps=config["stage_exploration_initial_eps"],
    exploration_final_eps=config["stage_exploration_final_eps"],
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=0,
)


# ── Curriculum loop ───────────────────────────────────────────────────────────

final_stage_env = train_env
for i, stage in enumerate(config["curriculum"], 1):
    if stage.get("randomize_package") and stage.get("randomize_delivery"):
        stage_mode = "random package+delivery"
    elif stage.get("randomize_package"):
        stage_mode = "random package"
    else:
        stage_mode = "fixed positions"

    print(f"\n=== Stage {i}/{len(config['curriculum'])}: {stage['name']} ({stage_mode}) ===")

    stage_env = build_env(stage, print_freq=config["env_print_freq"])
    describe_env("Stage", stage_env)
    model.set_env(stage_env)

    # Clear replay buffer when entering a new map to avoid stale transitions
    if stage.get("clear_buffer_on_start"):
        model.replay_buffer.reset()
        print(f"  Replay buffer cleared for {stage['name']}")

    # Per-stage learning_starts override
    if "learning_starts" in stage:
        model.learning_starts = stage["learning_starts"]
        print(f"  learning_starts set to {model.learning_starts}")

    # Exploration schedule
    for attr, key in [
        ("exploration_initial_eps", "stage_exploration_initial_eps"),
        ("exploration_final_eps",   "stage_exploration_final_eps"),
        ("exploration_fraction",    "stage_exploration_fraction"),
    ]:
        setattr(model, attr, stage.get(attr, config[key]))
    model.exploration_rate = model.exploration_initial_eps
    print(
        f"Exploration: init={model.exploration_initial_eps:.2f} "
        f"final={model.exploration_final_eps:.2f} "
        f"fraction={model.exploration_fraction:.2f}"
    )

    promotion_cb = StagePromotionCallback(
        stage_name=stage["name"],
        source_env=stage_env,
        eval_every=stage.get("eval_every", config["stage_eval_every"]),
        n_eval_episodes=stage.get("eval_episodes", config["stage_eval_episodes"]),
        promotion_threshold=stage.get("promotion_success_rate", config["stage_promotion_success_rate"]),
        min_timesteps_before_promotion=stage.get(
            "min_timesteps_before_promotion", config["stage_min_timesteps_before_promotion"]
        ),
        save_dir=save_dir,
    )

    model.learn(
        total_timesteps=stage["timesteps"],
        reset_num_timesteps=True,
        callback=[
            ProgressCallback(config["progress_print_freq"]),
            QValueDiagnosticCallback(stage_env, config["qdiag_freq"]),
            promotion_cb,
        ],
    )

    if not promotion_cb.promoted:
        print(f"Full budget used for {stage['name']}.")

    # Reset learning_starts to global default after each stage
    model.learning_starts = config["learning_starts"]
    final_stage_env = stage_env


# ── Final save + viz ──────────────────────────────────────────────────────────

model.save(f"{save_dir}/model_final")
print("Training complete!")
visualize_trained_policy(model, final_stage_env, delay=config["viz_delay"])
