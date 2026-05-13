import json
import os
import random
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

try:
    from envs.quadcopter_drone_env import QuadcopterDroneEnv
    from ppo_config import config
    from utils.pipes import PipeGrid, PipeOptions, PipeVisualizerBW
except ImportError:
    from quad_physics_ppo.envs.quadcopter_drone_env import QuadcopterDroneEnv
    from quad_physics_ppo.ppo_config import config
    from quad_physics_ppo.utils.pipes import PipeGrid, PipeOptions, PipeVisualizerBW



ACTION_NAMES = ["roll", "pitch"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DEFAULTS = {
    "randomize_package": False,
    "randomize_delivery": False,
    "step_penalty": -0.2,
    "invalid_move_penalty": 200.0,
    "revisit_penalty": 0.05,
    "revisit_penalty_cap": 1.5,
    "backtrack_penalty": 1.0,
    "pickup_reward": 300.0,
    "delivery_reward": 1000.0,
    "dt": 0.05,
    "physics_substeps": 8,
    "max_tilt": 0.55,
    "linear_drag": 0.22,
    "attitude_tau": 0.18,
    "angular_damping": 0.82,
    "target_altitude": 1.0,
}
ENV_ATTRS = ("max_steps", *ENV_DEFAULTS)
MODEL_KEYS = (
    "learning_rate",
    "n_steps",
    "batch_size",
    "gamma",
    "gae_lambda",
    "clip_range",
    "target_kl",
    "ent_coef",
    "vf_coef",
)


class ProgressCallback(BaseCallback):
    def __init__(self, print_freq=10_000):
        super().__init__()
        self.print_freq = print_freq
        self.episodes = 0

    def _on_step(self) -> bool:
        self.episodes += int(np.sum(self.locals.get("dones", [])))
        if self.num_timesteps % self.print_freq == 0:
            print(f"Step {self.num_timesteps:>7} | Episodes: {self.episodes:>5}")
        return True


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
        self.min_timesteps = min_timesteps_before_promotion
        self.save_dir = save_dir
        self.promoted = False
        self.best_success_rate = -1.0
        self.best_mean_reward = float("-inf")
        self.best_model_path = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every != 0:
            return True

        metrics = evaluate(self.model, self.source_env, self.n_eval_episodes)
        print(
            f"Eval @{self.num_timesteps:>7} | "
            f"sr={metrics['success_rate']:.2%} | "
            f"reward={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}"
        )
        checkpoint_path = os.path.join(
            self.save_dir,
            f"model_{self.stage_name}_{self.num_timesteps}_steps",
        )
        self.model.save(checkpoint_path)
        is_best = (
            metrics["success_rate"] > self.best_success_rate
            or (
                metrics["success_rate"] == self.best_success_rate
                and metrics["mean_reward"] > self.best_mean_reward
            )
        )
        if is_best:
            self.best_success_rate = metrics["success_rate"]
            self.best_mean_reward = metrics["mean_reward"]
            self.best_model_path = os.path.join(self.save_dir, f"model_{self.stage_name}_best")
            self.model.save(self.best_model_path)

        perfect_eval = metrics["success_rate"] >= 1.0
        can_promote = metrics["success_rate"] >= self.promotion_threshold and (
            perfect_eval or self.num_timesteps >= self.min_timesteps
        )
        if can_promote:
            print(
                f"Early promotion from {self.stage_name}: "
                f"{metrics['success_rate']:.2%} >= {self.promotion_threshold:.2%}"
            )
            self.promoted = True
            return False
        return True


def build_bw_map(grid_size, loop_prob=0.25, seed=None):
    np_state = np.random.get_state()
    random_state = random.getstate()
    try:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return PipeVisualizerBW(lanes=2, base=3).render(
            PipeGrid(grid_size[0], grid_size[1], loop_prob=loop_prob).to_pipe_ids(PipeOptions())
        )
    finally:
        np.random.set_state(np_state)
        random.setstate(random_state)


def build_straight_line_map(length, padding=1):
    height = 1 + 2 * padding
    width = length + 2 * padding
    bw_map = np.zeros((height, width), dtype=int)
    bw_map[padding, padding : padding + length] = 1
    return bw_map


def build_t_junction_map(stem_length=4, branch_left=2, branch_right=2, padding=1):
    height = stem_length + 2 * padding
    width = branch_left + 1 + branch_right + 2 * padding
    bw_map = np.zeros((height, width), dtype=int)
    center_col = padding + branch_left
    top_row = padding
    bw_map[padding : padding + stem_length, center_col] = 1
    bw_map[top_row, padding : padding + branch_left + 1 + branch_right] = 1
    return bw_map


def as_position(pos):
    return tuple(map(int, pos))


def shortest_path_distances(env, start):
    queue = deque([(start, 0)])
    distances = {start: 0}
    while queue:
        (row, col), dist = queue.popleft()
        for dr, dc in env.actions:
            nxt = (row + dr, col + dc)
            if nxt in env.lane_set and nxt not in distances:
                distances[nxt] = dist + 1
                queue.append((nxt, dist + 1))
    return distances


def pick_position_by_distance(candidates, min_dist, max_dist):
    valid = [pos for pos, dist in candidates if min_dist <= dist <= max_dist]
    if valid:
        return valid[-1]
    return candidates[-1][0]


def positions_in_distance_range(candidates, min_dist, max_dist):
    return [as_position(pos) for pos, dist in candidates if min_dist <= dist <= max_dist]


def configure_positions(env, stage_config):
    if "fixed_start" in stage_config and "fixed_package" in stage_config and "fixed_delivery" in stage_config:
        env.fixed_start = as_position(stage_config["fixed_start"])
        env.fixed_package = as_position(stage_config["fixed_package"])
        env.fixed_delivery = as_position(stage_config["fixed_delivery"])
        return

    start = as_position(min(env.lane_coords, key=lambda pos: (pos[0] + pos[1], pos[0], pos[1])))
    start_dists = shortest_path_distances(env, start)
    reachable = sorted(
        ((pos, dist) for pos, dist in start_dists.items() if pos != start),
        key=lambda item: (item[1], item[0][0], item[0][1]),
    )
    pkg = as_position(
        pick_position_by_distance(
            reachable,
            stage_config["package_min_dist"],
            stage_config["package_max_dist"],
        )
    )
    env.random_package_candidates = positions_in_distance_range(
        reachable,
        stage_config["package_min_dist"],
        stage_config["package_max_dist"],
    )

    pkg_dists = shortest_path_distances(env, pkg)
    delivery_candidates = sorted(
        (
            (pos, dist)
            for pos, dist in pkg_dists.items()
            if pos not in {start, pkg}
            and start_dists.get(pos, -1) >= stage_config["delivery_min_start_dist"]
        ),
        key=lambda item: (item[1], start_dists[item[0]], item[0][0], item[0][1]),
    )
    delivery = as_position(
        pick_position_by_distance(
            delivery_candidates,
            stage_config["delivery_min_dist"],
            stage_config["delivery_max_dist"],
        )
    )
    env.random_delivery_candidates = positions_in_distance_range(
        delivery_candidates,
        stage_config["delivery_min_dist"],
        stage_config["delivery_max_dist"],
    )

    env.fixed_start = start
    env.fixed_package = pkg
    env.fixed_delivery = delivery


def build_env(stage_config, print_freq=0):
    map_layout = stage_config.get("map_layout", "pipe")
    if map_layout == "straight_line":
        bw_map = build_straight_line_map(
            stage_config["line_length"],
            padding=stage_config.get("map_padding", 1),
        )
    elif map_layout == "t_junction":
        bw_map = build_t_junction_map(
            stem_length=stage_config["stem_length"],
            branch_left=stage_config.get("branch_left", 2),
            branch_right=stage_config.get("branch_right", 2),
            padding=stage_config.get("map_padding", 1),
        )
    else:
        bw_map = build_bw_map(
            stage_config["grid_size"],
            stage_config["loop_prob"],
            stage_config["map_seed"],
        )

    env_kwargs = {key: stage_config.get(key, default) for key, default in ENV_DEFAULTS.items()}
    env = QuadcopterDroneEnv(
        bw_map,
        max_steps=stage_config["max_steps"],
        print_freq=print_freq,
        **env_kwargs,
    )
    configure_positions(env, stage_config)
    return env


def describe_env(tag, env):
    print(
        f"{tag}: obs={env.observation_space} grid={env.grid_shape} "
        f"start={env.fixed_start} pkg={env.fixed_package} "
        f"delivery={env.fixed_delivery} lanes={len(env.lane_coords)} "
        f"rand_pkg={env.randomize_package} rand_del={env.randomize_delivery} "
        f"dt={env.dt} substeps={env.physics_substeps} max_tilt={env.max_tilt}"
    )


def clone_env(env):
    cloned = QuadcopterDroneEnv(env.bw_map.copy(), **{attr: getattr(env, attr) for attr in ENV_ATTRS})
    cloned.fixed_start = env.fixed_start
    cloned.fixed_package = env.fixed_package
    cloned.fixed_delivery = env.fixed_delivery
    cloned.random_package_candidates = env.random_package_candidates
    cloned.random_delivery_candidates = env.random_delivery_candidates
    return cloned


def is_delivery_success(env, done):
    return (
        done
        and tuple(env.state[:2]) == tuple(env.fixed_delivery)
        and bool(env.state[2])
    )


def evaluate(model, env, n_episodes=20):
    eval_env = clone_env(env)
    rewards = []
    successes = 0

    for seed in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed)
        done = truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        successes += int(is_delivery_success(eval_env, done))

    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


def rollout_once(model, src_env, seed=None):
    env = clone_env(src_env)

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    frames = []
    done = truncated = False

    while True:
        frames.append(
            {
                "step": env.current_step,
                "action": None if env.last_action is None else env.last_action.copy(),
                "reward": 0.0,
                "total_reward": float(total_reward),
                "state": tuple(env.state),
                "pos": env.pos.copy(),
                "vel": env.vel.copy(),
                "angles": env.angles.copy(),
                "done": bool(done),
                "truncated": bool(truncated),
                "delivered": bool(is_delivery_success(env, done)),
                "seed": seed,
            }
        )
        if done or truncated:
            break
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    return env, frames


def rollout_policy(model, src_env, n_trials=None):
    n_trials = config["stage_eval_episodes"] if n_trials is None else n_trials
    fallback = None

    for seed in range(n_trials):
        env, frames = rollout_once(model, src_env, seed=seed)
        if fallback is None:
            fallback = (env, frames)
        if frames[-1]["delivered"]:
            return env, frames

    return fallback


def draw_quad_top(ax, pos, angles, color):
    roll, pitch, _ = angles
    center = np.array([pos[1], pos[0]], dtype=np.float32)
    body = 0.32
    tilt = np.array([-roll, pitch], dtype=np.float32)
    tilt_norm = float(np.linalg.norm(tilt))
    if tilt_norm > 1e-6:
        tilt_dir = tilt / tilt_norm
    else:
        tilt_dir = np.array([0.0, 0.0], dtype=np.float32)
    tilt_shift = tilt_dir * min(tilt_norm, 0.65) * 0.18
    arms = (
        (np.array([-body, 0.0]), np.array([body, 0.0])),
        (np.array([0.0, -body]), np.array([0.0, body])),
    )
    for start, end in arms:
        ax.plot(
            [center[0] + start[0], center[0] + end[0]],
            [center[1] + start[1], center[1] + end[1]],
            color="#111111",
            linewidth=2.0,
            zorder=8,
        )
    for offset in (np.array([-body, 0.0]), np.array([body, 0.0]), np.array([0.0, -body]), np.array([0.0, body])):
        rotor = center + offset + tilt_shift
        frontness = float(np.dot(offset, tilt_dir)) if tilt_norm > 1e-6 else 0.0
        rotor_size = 34 - 10 * frontness
        ax.scatter(rotor[0], rotor[1], s=rotor_size, color=color, edgecolor="#111111", linewidth=0.8, zorder=9)
    if tilt_norm > 0.03:
        nose = center + tilt_dir * 0.48
        ax.annotate(
            "",
            xy=(nose[0], nose[1]),
            xytext=(center[0], center[1]),
            arrowprops={"arrowstyle": "-|>", "color": "#111111", "lw": 1.6},
            zorder=11,
        )
    ax.scatter(center[0], center[1], s=64, color=color, edgecolor="#111111", linewidth=1.0, zorder=10)


def format_action(action):
    if action is None:
        return "-"
    roll_cmd, pitch_cmd = np.asarray(action, dtype=np.float32)
    return f"roll={roll_cmd:+.2f}, pitch={pitch_cmd:+.2f}"


def draw_rollout_frame(ax_g, ax_i, env, frame):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#1a1a1a", "#f1efe8"])
    row, col, has_pkg = frame["state"]
    pos = frame["pos"]
    vel = frame["vel"]
    angles = frame["angles"]

    ax_g.clear()
    ax_i.clear()

    ax_g.imshow(env.bw_map, cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax_g.set(xticks=[], yticks=[], title=f"Quad PPO Rollout | Step {frame['step']}")
    ax_g.set_xlim(-0.5, env.grid_shape[1] - 0.5)
    ax_g.set_ylim(env.grid_shape[0] - 0.5, -0.5)

    sr, sc = env.fixed_start
    pr, pc = env.fixed_package
    dr, dc = env.fixed_delivery
    ax_g.scatter(sc, sr, s=80, color="#5b8c5a", zorder=4)
    if not has_pkg:
        ax_g.scatter(pc, pr, marker="s", s=120, color="#f2c14e", zorder=5)
    ax_g.scatter(dc, dr, marker="*", s=180, color="#ef476f", zorder=5)

    if frame["delivered"]:
        drone_color = "#06d6a0"
    elif frame["truncated"]:
        drone_color = "#8338ec"
    elif frame["done"]:
        drone_color = "#d62828"
    elif has_pkg:
        drone_color = "#ff7f11"
    else:
        drone_color = "#118ab2"

    draw_quad_top(ax_g, pos, angles, drone_color)
    ax_g.scatter(col, row, s=35, color="#ffffff", edgecolor="#111111", zorder=12)

    ax_i.set(xlim=(0, 1), ylim=(0, 1))
    ax_i.axis("off")
    action_name = format_action(frame["action"])
    roll, pitch, _ = angles
    lines = [
        ("action", action_name),
        ("total reward", f"{frame['total_reward']:+.2f}"),
        ("seed", "-" if frame["seed"] is None else str(frame["seed"])),
        ("cell", f"({row}, {col})"),
        ("position", f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"),
        ("velocity", f"({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})"),
        ("roll/pitch", f"({roll:.2f}, {pitch:.2f})"),
        ("carrying", "yes" if has_pkg else "no"),
        ("goal", str(env.fixed_delivery if has_pkg else env.fixed_package)),
        (
            "status",
            "delivered"
            if frame["delivered"]
            else "truncated"
            if frame["truncated"]
            else "ended"
            if frame["done"]
            else "running",
        ),
    ]
    ax_i.text(0.5, 0.96, "Quad State", ha="center", va="top", fontsize=14, fontweight="bold")
    for y, (key, value) in zip(np.linspace(0.86, 0.08, len(lines)), lines):
        ax_i.text(0.08, y, key, ha="left", va="top", fontsize=10, color="#666")
        ax_i.text(0.92, y, value, ha="right", va="top", fontsize=10, color="#111")
    plt.tight_layout(pad=1.4)


def save_rollout_images(fig, ax_g, ax_i, env, frames, image_dir):
    os.makedirs(image_dir, exist_ok=True)
    frame_indices = sorted(
        set(
            [
                0,
                len(frames) // 3,
                (2 * len(frames)) // 3,
                len(frames) - 1,
            ]
        )
    )
    for frame_idx in frame_indices:
        draw_rollout_frame(ax_g, ax_i, env, frames[frame_idx])
        path = os.path.join(image_dir, f"quad_rollout_step_{frames[frame_idx]['step']:04d}.png")
        fig.savefig(path, dpi=160)
    print(f"Saved rollout images -> {image_dir}")


def visualize_trained_policy(model, src_env, delay=0.12, save_path=None, image_dir=None):
    import matplotlib.pyplot as plt

    env, frames = rollout_policy(model, src_env)
    fig, (ax_g, ax_i) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.2, 1]})

    if image_dir is not None:
        save_rollout_images(fig, ax_g, ax_i, env, frames, image_dir)

    if save_path is not None:
        from matplotlib.animation import FuncAnimation

        def update(frame_idx):
            draw_rollout_frame(ax_g, ax_i, env, frames[frame_idx])

        anim = FuncAnimation(fig, update, frames=len(frames), interval=delay * 1000, repeat=False)
        writer = "pillow" if save_path.lower().endswith(".gif") else "ffmpeg"
        anim.save(save_path, writer=writer)
        print(f"Saved rollout animation -> {save_path}")
        plt.close(fig)
        return

    plt.ion()
    plt.show(block=False)
    for frame in frames:
        draw_rollout_frame(ax_g, ax_i, env, frame)
        fig.canvas.draw_idle()
        plt.pause(delay)
    plt.ioff()
    plt.show()


def make_model(env):
    return PPO(
        "MlpPolicy",
        env,
        **{key: config[key] for key in MODEL_KEYS},
        policy_kwargs={"net_arch": config["net_arch"]},
        verbose=0,
        device="cpu",
    )


def stage_mode(stage):
    if stage.get("randomize_package") and stage.get("randomize_delivery"):
        return "random package+delivery"
    if stage.get("randomize_package"):
        return "random package"
    return "fixed positions"


def make_promotion_callback(stage, stage_env, save_dir):
    return StagePromotionCallback(
        stage_name=stage["name"],
        source_env=stage_env,
        eval_every=stage.get("eval_every", config["stage_eval_every"]),
        n_eval_episodes=stage.get("eval_episodes", config["stage_eval_episodes"]),
        promotion_threshold=stage.get("promotion_success_rate", config["stage_promotion_success_rate"]),
        min_timesteps_before_promotion=stage.get(
            "min_timesteps_before_promotion",
            config["stage_min_timesteps_before_promotion"],
        ),
        save_dir=save_dir,
    )


def main():
    save_dir = os.path.join(BASE_DIR, "checkpoints", "ppo", "task1", config["run_name"])
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    first_stage = {**config, **config["curriculum"][0]}
    env = build_env(first_stage, print_freq=config["env_print_freq"])
    describe_env("Initial", env)

    model = make_model(env)
    final_stage_env = env
    total_stages = len(config["curriculum"])
    for stage_idx, stage in enumerate(config["curriculum"], start=1):
        stage_config = {**config, **stage}
        stage_env = build_env(stage_config, print_freq=config["env_print_freq"])
        final_stage_env = stage_env

        print(f"\n=== Stage {stage_idx}/{total_stages}: {stage['name']} ({stage_mode(stage)}) ===")
        describe_env("Stage", stage_env)
        model.set_env(stage_env)
        pre_stage_path = os.path.join(save_dir, f"model_{stage['name']}_pre_stage")
        model.save(pre_stage_path)
        promotion_cb = make_promotion_callback(stage, stage_env, save_dir)

        model.learn(
            total_timesteps=stage["timesteps"],
            reset_num_timesteps=True,
            callback=[
                ProgressCallback(config["progress_print_freq"]),
                promotion_cb,
            ],
        )

        if not promotion_cb.promoted:
            print(f"Full budget used for {stage['name']}.")
        if promotion_cb.best_model_path is not None and promotion_cb.best_success_rate > 0.0:
            model = PPO.load(promotion_cb.best_model_path, env=stage_env, device="cpu")
            print(
                f"Restored best for {stage['name']}: "
                f"sr={promotion_cb.best_success_rate:.2%} | "
                f"reward={promotion_cb.best_mean_reward:.2f}"
            )
        elif promotion_cb.best_model_path is not None:
            model = PPO.load(pre_stage_path, env=stage_env, device="cpu")
            print(f"No successful checkpoint found for {stage['name']}; restored pre-stage model.")

    metrics = evaluate(model, final_stage_env, config["stage_eval_episodes"])
    print(
        f"Final eval | success_rate={metrics['success_rate']:.2%} "
        f"| mean_reward={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}"
    )

    model.save(os.path.join(save_dir, "model_final"))

    if config["viz_enabled"]:
        viz_save_path = config["viz_save_path"]
        if viz_save_path is not None and not os.path.isabs(viz_save_path):
            viz_save_path = os.path.join(save_dir, viz_save_path)
        viz_image_dir = config.get("viz_image_dir") if config.get("viz_save_images") else None
        if viz_image_dir is not None and not os.path.isabs(viz_image_dir):
            viz_image_dir = os.path.join(save_dir, viz_image_dir)
        visualize_trained_policy(
            model,
            final_stage_env,
            delay=config["viz_delay"],
            save_path=viz_save_path,
            image_dir=viz_image_dir,
        )


if __name__ == "__main__":
    main()
