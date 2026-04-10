import json
import os
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.drone_env import DroneEnv
from classical_methods.utils.pipes import PipeGrid, PipeOptions, PipeVisualizerBW
from classical_methods.utils.saliency import run_saliency_suite as render_saliency_suite

ACTION_NAMES = ["up", "down", "left", "right"]


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
    return candidates[-1][0]


def configure_positions(env, config):
    start = min(env.lane_coords, key=lambda pos: (pos[0] + pos[1], pos[0], pos[1]))
    start_dists = shortest_path_distances(env, start)
    reachable = sorted(
        ((pos, dist) for pos, dist in start_dists.items() if pos != start),
        key=lambda item: (item[1], item[0][0], item[0][1]),
    )
    pkg = pick_position_by_distance(reachable, config["package_min_dist"], config["package_max_dist"])

    pkg_dists = shortest_path_distances(env, pkg)
    delivery_candidates = sorted(
        (
            (pos, dist) for pos, dist in pkg_dists.items()
            if pos not in {start, pkg} and start_dists.get(pos, -1) >= config["delivery_min_start_dist"]
        ),
        key=lambda item: (item[1], start_dists[item[0]], item[0][0], item[0][1]),
    )
    delivery = pick_position_by_distance(
        delivery_candidates,
        config["delivery_min_dist"],
        config["delivery_max_dist"],
    )

    env.fixed_start = start
    env.fixed_package = pkg
    env.fixed_delivery = delivery


def build_env(config, print_freq=0):
    env = DroneEnv(
        build_bw_map(config["grid_size"], config["loop_prob"], config["map_seed"]),
        max_steps=config["max_steps"],
        print_freq=print_freq,
        progress_reward_scale=config["progress_reward_scale"],
    )
    configure_positions(env, config)
    return env


def clone_env(env):
    cloned = DroneEnv(
        env.bw_map.copy(),
        max_steps=env.max_steps,
        progress_reward_scale=env.progress_reward_scale,
        randomize_package=env.randomize_package,
        randomize_delivery=env.randomize_delivery,
        step_penalty=env.step_penalty,
        invalid_move_penalty=env.invalid_move_penalty,
        revisit_penalty=env.revisit_penalty,
        revisit_penalty_cap=env.revisit_penalty_cap,
        backtrack_penalty=env.backtrack_penalty,
        pickup_reward=env.pickup_reward,
        delivery_reward=env.delivery_reward,
    )
    cloned.fixed_start = env.fixed_start
    cloned.fixed_package = env.fixed_package
    cloned.fixed_delivery = env.fixed_delivery
    return cloned


def _state_to_obs(env, state):
    prev_state = env.state
    env.state = (state[0], state[1], int(state[2]))
    try:
        return env._get_obs()
    finally:
        env.state = prev_state


def _collect_policy_path(model, src_env):
    env = clone_env(src_env)
    env.randomize_package = False
    env.randomize_delivery = False

    obs, _ = env.reset()
    path = [tuple(env.state)]
    done = truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        path.append(tuple(env.state))

    return env, path


def run_saliency_suite(model, src_env, show=True):
    import torch

    rollout_env, path = _collect_policy_path(model, src_env)

    class SaliencyPPOAdapter:
        def __init__(self, sb3_model, env):
            self.actions = env.actions
            self.lane_coords = env.lane_coords
            self.package_pos = env.fixed_package
            self.delivery_pos = env.fixed_delivery
            self.Q = {}

            with torch.no_grad():
                for r, c in env.lane_coords:
                    for has_pkg in (False, True):
                        state = (r, c, has_pkg)
                        obs = _state_to_obs(env, state)
                        obs_tensor = torch.as_tensor(
                            obs[None],
                            dtype=torch.float32,
                            device=sb3_model.device,
                        )
                        dist = sb3_model.policy.get_distribution(obs_tensor)
                        action_scores = dist.distribution.probs.cpu().numpy()[0]
                        self.Q[state] = action_scores.copy()

    render_saliency_suite(
        SaliencyPPOAdapter(model, rollout_env),
        path,
        rollout_env.bw_map,
        show=show,
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
        delivered = (
            done
            and tuple(eval_env.state[:2]) == tuple(eval_env.fixed_delivery)
            and bool(eval_env.state[2])
        )
        successes += int(delivered)

    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


def rollout_policy(model, src_env):
    env = clone_env(src_env)
    env.randomize_package = False
    env.randomize_delivery = False

    obs, _ = env.reset()
    total_reward = 0.0
    frames = [{
        "step": 0,
        "action": None,
        "reward": 0.0,
        "total_reward": 0.0,
        "state": tuple(env.state),
        "trajectory": [(env.state[0], env.state[1])],
        "done": False,
        "truncated": False,
    }]
    trajectory = [(env.state[0], env.state[1])]
    done = truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        trajectory.append((env.state[0], env.state[1]))
        frames.append({
            "step": len(trajectory) - 1,
            "action": int(action),
            "reward": float(reward),
            "total_reward": float(total_reward),
            "state": tuple(env.state),
            "trajectory": trajectory.copy(),
            "done": bool(done),
            "truncated": bool(truncated),
        })

    return env, frames


def draw_rollout_frame(ax_g, ax_i, env, frame):
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#1a1a1a", "#f1efe8"])
    action = frame["action"]
    reward = frame["reward"]
    total_reward = frame["total_reward"]
    r, c, has_pkg = frame["state"]

    ax_g.clear()
    ax_i.clear()
    ax_g.imshow(env.bw_map, cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax_g.set(xticks=[], yticks=[], title=f"Policy Rollout | Step {frame['step']}")

    trajectory = frame["trajectory"]
    if len(trajectory) > 1:
        ax_g.plot(
            [pos[1] for pos in trajectory],
            [pos[0] for pos in trajectory],
            color="#3f88c5",
            linewidth=2.0,
            alpha=0.8,
        )

    sr, sc = env.fixed_start
    pr, pc = env.fixed_package
    dr, dc = env.fixed_delivery
    ax_g.scatter(sc, sr, s=80, color="#5b8c5a", zorder=4)
    if not has_pkg:
        ax_g.scatter(pc, pr, marker="s", s=120, color="#f2c14e", zorder=5)
    ax_g.scatter(dc, dr, marker="*", s=180, color="#ef476f", zorder=5)

    drone_color = "#06d6a0" if frame["done"] else "#8338ec" if frame["truncated"] else "#ff7f11" if has_pkg else "#118ab2"
    ax_g.scatter(c, r, s=140, color=drone_color, zorder=6)

    ax_i.set(xlim=(0, 1), ylim=(0, 1))
    ax_i.axis("off")
    ax_i.text(0.5, 0.96, "Rollout Info", ha="center", va="top", fontsize=14, fontweight="bold")
    goal = env.fixed_delivery if has_pkg else env.fixed_package
    action_name = "-" if action is None else ACTION_NAMES[action]
    lines = [
        ("action", action_name),
        ("step reward", f"{reward:+.2f}"),
        ("total reward", f"{total_reward:+.2f}"),
        ("position", f"({r}, {c})"),
        ("carrying", "yes" if has_pkg else "no"),
        ("goal", str(goal)),
        ("start", str(env.fixed_start)),
        ("package", str(env.fixed_package)),
        ("delivery", str(env.fixed_delivery)),
        ("status", "delivered" if frame["done"] else "truncated" if frame["truncated"] else "running"),
    ]
    for y, (key, value) in zip(np.linspace(0.86, 0.01, len(lines)), lines):
        ax_i.text(0.08, y, key, ha="left", va="top", fontsize=10, color="#666")
        ax_i.text(0.92, y, value, ha="right", va="top", fontsize=10, color="#111")


def visualize_trained_policy(model, src_env, delay=0.12, save_path=None):
    import matplotlib.pyplot as plt

    env, frames = rollout_policy(model, src_env)
    fig, (ax_g, ax_i) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.tight_layout(pad=2.0)

    if save_path is not None:
        from matplotlib.animation import FuncAnimation

        def update(frame_idx):
            draw_rollout_frame(ax_g, ax_i, env, frames[frame_idx])

        anim = FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=delay * 1000,
            repeat=False,
        )
        anim.save(save_path, writer="ffmpeg")
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


config = {
    "run_name": "basic_fixed_v2",
    "grid_size": [4, 4],
    "loop_prob": 0.25,
    "map_seed": 101,
    "max_steps": 250,
    "progress_reward_scale": 0.5,
    "package_min_dist": 5,
    "package_max_dist": 10,
    "delivery_min_dist": 4,
    "delivery_max_dist": 9,
    "delivery_min_start_dist": 8,
    "total_timesteps": 150_000,
    "learning_rate": 1e-3,
    "n_steps": 1024,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "net_arch": [64, 64],
    "progress_print_freq": 10_000,
    "eval_episodes": 20,
    "viz_delay": 0.12,
    "viz_enabled": True,
    "viz_save_path": None,
}

def main():
    save_dir = os.path.join("checkpoints", "ppo", "task1", config["run_name"])
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = build_env(config)
    print(
        f"obs={env.observation_space} grid={env.grid_shape} "
        f"start={env.fixed_start} pkg={env.fixed_package} delivery={env.fixed_delivery}"
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        policy_kwargs={"net_arch": config["net_arch"]},
        verbose=0,
        device="cpu",
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=ProgressCallback(config["progress_print_freq"]),
    )

    metrics = evaluate(model, env, config["eval_episodes"])
    print(
        f"Final eval | success_rate={metrics['success_rate']:.2%} "
        f"| mean_reward={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}"
    )

    model.save(os.path.join(save_dir, "model_final"))
    run_saliency_suite(model, env, show=True)

    if config["viz_enabled"]:
        viz_save_path = config["viz_save_path"]
        if viz_save_path is not None and not os.path.isabs(viz_save_path):
            viz_save_path = os.path.join(save_dir, viz_save_path)
        visualize_trained_policy(
            model,
            env,
            delay=config["viz_delay"],
            save_path=viz_save_path,
        )


if __name__ == "__main__":
    main()
