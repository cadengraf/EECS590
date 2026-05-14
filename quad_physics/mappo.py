import copy
import json
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRONE_RL_DIR = os.path.join(os.path.dirname(BASE_DIR), "drone_rl")
if DRONE_RL_DIR not in sys.path:
    sys.path.insert(0, DRONE_RL_DIR)

try:
    from envs.multi_quadcopter_drone_env import ACTION_DIM, build_env, clone_env, describe_env, stage_mode
    from mappo_config import config
except ImportError:
    from quad_physics.envs.multi_quadcopter_drone_env import (
        ACTION_DIM,
        build_env,
        clone_env,
        describe_env,
        stage_mode,
    )
    from quad_physics.mappo_config import config

try:
    from classical_methods.utils.saliency import run_saliency_suite as render_saliency_suite
except ImportError:
    try:
        from drone_rl.classical_methods.utils.saliency import run_saliency_suite as render_saliency_suite
    except ImportError:
        render_saliency_suite = None


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, state_dim, action_dim, hidden_size):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )
        nn.init.zeros_(self.actor_mean[4].weight)
        nn.init.zeros_(self.actor_mean[4].bias)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -1.0))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def action_dist(self, obs):
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        return torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)

    def deterministic_action(self, obs):
        return torch.clamp(self.actor_mean(obs), -1.0, 1.0)

    def value(self, state):
        return self.critic(state).squeeze(-1)


def flatten_state(obs):
    return obs.reshape(-1)


def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for step in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[step]
        next_values = next_value if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    return advantages, advantages + values


def collect_rollout(env, model, cfg, device):
    obs, _ = env.reset()
    storage = {key: [] for key in ("obs", "states", "actions", "log_probs", "rewards", "dones", "values")}
    episode_rewards = []
    current_episode_reward = 0.0
    successes = 0
    rollout_pickups = 0
    rollout_deliveries = 0

    for _ in range(cfg["rollout_steps"]):
        state = flatten_state(obs)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist = model.action_dist(obs_tensor)
            action = torch.clamp(dist.sample(), -1.0, 1.0)
            log_prob = dist.log_prob(action)
            value = model.value(state_tensor).item()

        next_obs, rewards, done, truncated, info = env.step(action.cpu().numpy())
        terminal = done or truncated
        storage["obs"].append(obs.copy())
        storage["states"].append(state.copy())
        storage["actions"].append(action.cpu().numpy())
        storage["log_probs"].append(log_prob.cpu().numpy())
        storage["rewards"].append(float(np.mean(rewards)))
        storage["dones"].append(float(terminal))
        storage["values"].append(value)

        current_episode_reward += float(np.sum(rewards))
        obs = next_obs
        if terminal:
            episode_rewards.append(current_episode_reward)
            successes += int(info.get("success", False))
            rollout_pickups += int(info.get("episode_pickups", 0))
            rollout_deliveries += int(info.get("episode_deliveries", 0))
            current_episode_reward = 0.0
            obs, _ = env.reset()

    with torch.no_grad():
        next_value = model.value(
            torch.as_tensor(flatten_state(obs), dtype=torch.float32, device=device).unsqueeze(0)
        ).item()

    rewards = np.asarray(storage["rewards"], dtype=np.float32)
    dones = np.asarray(storage["dones"], dtype=np.float32)
    values = np.asarray(storage["values"], dtype=np.float32)
    storage["advantages"], storage["returns"] = compute_gae(
        rewards,
        dones,
        values,
        next_value,
        cfg["gamma"],
        cfg["gae_lambda"],
    )
    return storage, episode_rewards, successes, rollout_pickups, rollout_deliveries


def update_model(model, optimizer, rollout, cfg, device, num_drones):
    obs = torch.as_tensor(np.asarray(rollout["obs"]), dtype=torch.float32, device=device)
    states = torch.as_tensor(np.asarray(rollout["states"]), dtype=torch.float32, device=device)
    actions = torch.as_tensor(np.asarray(rollout["actions"]), dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(np.asarray(rollout["log_probs"]), dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    idxs = np.arange(obs.shape[0])
    stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
    updates = 0
    for _ in range(cfg["epochs"]):
        np.random.shuffle(idxs)
        for start in range(0, len(idxs), cfg["minibatch_size"]):
            mb = idxs[start : start + cfg["minibatch_size"]]
            dist = model.action_dist(obs[mb].reshape(-1, obs.shape[-1]))
            new_log_probs = dist.log_prob(actions[mb].reshape(-1, ACTION_DIM)).reshape(-1, num_drones)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[mb])
            mb_adv = advantages[mb].unsqueeze(-1)
            actor_loss = -torch.min(
                ratio * mb_adv,
                torch.clamp(ratio, 1.0 - cfg["clip_range"], 1.0 + cfg["clip_range"]) * mb_adv,
            ).mean()
            critic_loss = 0.5 * (returns[mb] - model.value(states[mb])).pow(2).mean()
            loss = actor_loss + cfg["vf_coef"] * critic_loss - cfg["ent_coef"] * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            if "min_actor_log_std" in cfg:
                with torch.no_grad():
                    model.actor_log_std.clamp_(min=float(cfg["min_actor_log_std"]))

            stats["actor_loss"] += float(actor_loss.item())
            stats["critic_loss"] += float(critic_loss.item())
            stats["entropy"] += float(entropy.item())
            updates += 1
    return {key: value / max(updates, 1) for key, value in stats.items()}


def evaluate(model, env, n_episodes, device):
    eval_env = clone_env(env)
    rewards = []
    successes = 0
    pickups = []
    deliveries = []
    delivered_agents = []
    for seed in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed)
        done = truncated = False
        total_reward = 0.0
        info = {}
        while not (done or truncated):
            with torch.no_grad():
                actions = model.deterministic_action(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
            obs, reward, done, truncated, info = eval_env.step(actions)
            total_reward += float(np.sum(reward))
        rewards.append(total_reward)
        successes += int(info.get("success", False))
        pickups.append(int(info.get("episode_pickups", 0)))
        deliveries.append(int(info.get("episode_deliveries", 0)))
        delivered_agents.append(float(np.sum(info.get("delivered", np.zeros(eval_env.num_drones)))))
    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_pickups": float(np.mean(pickups)),
        "mean_deliveries": float(np.mean(deliveries)),
        "mean_delivered_agents": float(np.mean(delivered_agents)),
    }


def save_checkpoint(model, cfg, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, os.path.join(save_dir, f"{name}.pt"))


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def checkpoint_step(checkpoint_path):
    match = re.search(r"_(\d+)_steps\.pt$", os.path.basename(checkpoint_path))
    return int(match.group(1)) if match else 0


def stage_checkpoint_path(save_dir, stage_name, suffix):
    return os.path.join(save_dir, f"model_{stage_name}_{suffix}.pt")


def latest_stage_step_checkpoint(save_dir, stage_name):
    prefix = f"model_{stage_name}_"
    best_path = None
    best_step = -1
    if not os.path.isdir(save_dir):
        return None
    for filename in os.listdir(save_dir):
        if not filename.startswith(prefix) or not filename.endswith("_steps.pt"):
            continue
        step = checkpoint_step(filename)
        if step > best_step:
            best_step = step
            best_path = os.path.join(save_dir, filename)
    return best_path


def failed_stage_records(save_dir):
    records = []
    if not os.path.isdir(save_dir):
        return records
    for filename in os.listdir(save_dir):
        match = re.match(r"failed_(.+)_(\d+)_steps\.gif$", filename)
        if not match:
            continue
        path = os.path.join(save_dir, filename)
        records.append(
            {
                "stage_name": match.group(1),
                "steps": int(match.group(2)),
                "path": path,
                "mtime": os.path.getmtime(path),
            }
        )
    return sorted(records, key=lambda record: record["mtime"], reverse=True)


def infer_resume_state(cfg, save_dir):
    resume_from = cfg.get("resume_from_checkpoint")
    if not resume_from:
        return None

    stages = cfg["curriculum"]
    stage_names = [stage["name"] for stage in stages]
    stage_name_to_idx = {name: idx for idx, name in enumerate(stage_names)}
    explicit_stage_name = cfg.get("resume_stage_name")
    final_path = os.path.join(save_dir, "model_final.pt")

    if resume_from == "auto":
        for failed in failed_stage_records(save_dir):
            failed_stage_name = failed["stage_name"]
            if failed_stage_name not in stage_name_to_idx or not os.path.exists(final_path):
                continue

            start_idx = stage_name_to_idx[failed_stage_name]
            global_steps = 0
            if start_idx > 0:
                previous_stage_name = stage_names[start_idx - 1]
                previous_checkpoint = latest_stage_step_checkpoint(save_dir, previous_stage_name)
                if previous_checkpoint is not None:
                    global_steps = checkpoint_step(previous_checkpoint)

            return {
                "path": final_path,
                "start_idx": start_idx,
                "global_steps": global_steps,
                "reason": f"latest failed stage {failed_stage_name}",
            }

        completed_idx = -1
        completed_path = None
        failed_stage_names = {record["stage_name"] for record in failed_stage_records(save_dir)}
        for idx, stage_name in enumerate(stage_names):
            if stage_name in failed_stage_names:
                break
            best_path = stage_checkpoint_path(save_dir, stage_name, "best")
            if not os.path.exists(best_path):
                break
            completed_idx = idx
            completed_path = best_path

        if completed_path is not None:
            latest_checkpoint = latest_stage_step_checkpoint(save_dir, stage_names[completed_idx])
            return {
                "path": completed_path,
                "start_idx": completed_idx + 1,
                "global_steps": checkpoint_step(latest_checkpoint or completed_path),
                "reason": f"completed stage {stage_names[completed_idx]}",
            }
        return None

    checkpoint_path = resume_from if os.path.isabs(resume_from) else os.path.join(save_dir, resume_from)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    if explicit_stage_name is not None:
        if explicit_stage_name not in stage_name_to_idx:
            raise ValueError(f"resume_stage_name={explicit_stage_name!r} is not in the curriculum")
        start_idx = stage_name_to_idx[explicit_stage_name]
    else:
        basename = os.path.basename(checkpoint_path)
        start_idx = 0
        for idx, stage_name in enumerate(stage_names):
            if basename.startswith(f"model_{stage_name}_"):
                start_idx = idx + 1
                break
        if basename == "model_final.pt":
            failed = next((record for record in failed_stage_records(save_dir) if record["stage_name"] in stage_name_to_idx), None)
            start_idx = stage_name_to_idx[failed["stage_name"]] if failed else len(stages)

    return {
        "path": checkpoint_path,
        "start_idx": start_idx,
        "global_steps": checkpoint_step(checkpoint_path),
        "reason": "explicit resume checkpoint",
    }


def apply_stage_exploration(model, stage_config):
    if "actor_log_std" not in stage_config:
        return
    with torch.no_grad():
        model.actor_log_std.fill_(float(stage_config["actor_log_std"]))
    print(f"Set actor_log_std={float(stage_config['actor_log_std']):+.2f} for stage exploration")


def select_device(cfg):
    requested = cfg.get("device", "cpu")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Using device: {device}")
    return device


def rollout_policy(model, src_env, device):
    env = clone_env(src_env)
    obs, _ = env.reset()
    trajectories = [[pos] for pos in env.positions]
    total_reward = 0.0
    frames = []
    done = truncated = False
    while True:
        frames.append(
            {
                "step": env.current_step,
                "actions": env.last_actions.copy(),
                "total_reward": total_reward,
                "positions": list(env.positions),
                "continuous_pos": env.pos.copy(),
                "angles": env.angles.copy(),
                "has_package": env.has_package.copy(),
                "delivered": env.delivered.copy(),
                "trajectories": [path.copy() for path in trajectories],
                "done": bool(done),
                "truncated": bool(truncated),
            }
        )
        if done or truncated:
            break
        with torch.no_grad():
            actions = model.deterministic_action(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
        obs, rewards, done, truncated, _ = env.step(actions)
        total_reward += float(np.sum(rewards))
        for idx, pos in enumerate(env.positions):
            trajectories[idx].append(pos)
    return env, frames


def draw_rollout_frame(ax_grid, ax_info, env, frame):
    from matplotlib.colors import ListedColormap

    colors = ["#118ab2", "#ef476f", "#06a77d", "#8338ec"]
    ax_grid.clear()
    ax_info.clear()
    ax_grid.imshow(env.bw_map, cmap=ListedColormap(["#1a1a1a", "#f1efe8"]), origin="upper", vmin=0, vmax=1)
    ax_grid.set(xticks=[], yticks=[], title=f"Quad MAPPO Rollout | Step {frame['step']}")
    pr, pc = env.fixed_package
    ax_grid.scatter(pc, pr, marker="s", s=140, color="#f2c14e", edgecolor="#111", zorder=5)
    for idx, (dr, dc) in enumerate(env.fixed_deliveries):
        ax_grid.scatter(dc, dr, marker="*", s=200, color=colors[idx % len(colors)], edgecolor="#111", zorder=5)
    for idx, trajectory in enumerate(frame["trajectories"]):
        if len(trajectory) > 1:
            ax_grid.plot([pos[1] for pos in trajectory], [pos[0] for pos in trajectory], color=colors[idx % len(colors)], linewidth=1.8, alpha=0.7)
    for idx, pos in enumerate(frame["continuous_pos"]):
        color = "#2ec4b6" if frame["delivered"][idx] else colors[idx % len(colors)]
        marker = "D" if frame["has_package"][idx] else "o"
        ax_grid.scatter(pos[1], pos[0], marker=marker, s=150, color=color, edgecolor="#111", linewidth=1.0, zorder=7)
        ax_grid.text(pos[1], pos[0], str(idx), ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=8)

    status = "delivered" if frame["done"] else "truncated" if frame["truncated"] else "running"
    ax_info.set(xlim=(0, 1), ylim=(0, 1))
    ax_info.axis("off")
    ax_info.text(0.5, 0.96, "Quad MAPPO", ha="center", va="top", fontsize=14, fontweight="bold")
    lines = [("status", status), ("total reward", f"{frame['total_reward']:+.2f}"), ("package", str(env.fixed_package))]
    y = 0.84
    for key, value in lines:
        ax_info.text(0.08, y, key, ha="left", va="top", fontsize=10, color="#666")
        ax_info.text(0.92, y, value, ha="right", va="top", fontsize=10, color="#111")
        y -= 0.08
    for idx in range(env.num_drones):
        action = frame["actions"][idx]
        roll, pitch, _ = frame["angles"][idx]
        ax_info.text(0.08, y, f"drone {idx}", ha="left", va="top", fontsize=10, color=colors[idx % len(colors)], fontweight="bold")
        y -= 0.055
        for key, value in [
            ("action", f"r={action[0]:+.2f}, p={action[1]:+.2f}"),
            ("cell", str(frame["positions"][idx])),
            ("roll/pitch", f"({roll:+.2f}, {pitch:+.2f})"),
            ("carrying", "yes" if frame["has_package"][idx] else "no"),
            ("delivered", "yes" if frame["delivered"][idx] else "no"),
            ("delivery", str(env.fixed_deliveries[idx])),
        ]:
            ax_info.text(0.12, y, key, ha="left", va="top", fontsize=9, color="#666")
            ax_info.text(0.92, y, value, ha="right", va="top", fontsize=9, color="#111")
            y -= 0.045
        y -= 0.03


class RolloutSaliencyAdapter:
    def __init__(self, package_pos, delivery_pos, actions):
        self.package_pos = tuple(package_pos)
        self.delivery_pos = tuple(delivery_pos)
        self.actions = list(actions)


def render_rollout_saliency(env, frames, agent_idx=0, show=False):
    if render_saliency_suite is None or not frames:
        return
    agent_idx = min(agent_idx, env.num_drones - 1)
    path = [
        (
            frame["positions"][agent_idx][0],
            frame["positions"][agent_idx][1],
            bool(frame["has_package"][agent_idx]),
        )
        for frame in frames
    ]
    adapter = RolloutSaliencyAdapter(env.fixed_package, env.fixed_deliveries[agent_idx], env.actions)
    try:
        render_saliency_suite(adapter, path, env.bw_map, show=show)
    except Exception as exc:
        print(f"[saliency] Skipped rollout saliency: {exc}")


def save_rollout_images(fig, ax_grid, ax_info, env, frames, image_dir):
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
        draw_rollout_frame(ax_grid, ax_info, env, frames[frame_idx])
        path = os.path.join(image_dir, f"mappo_quad_rollout_step_{frames[frame_idx]['step']:04d}.png")
        fig.savefig(path, dpi=160)
    print(f"Saved rollout images -> {image_dir}")


def visualize_trained_policy(model, src_env, device, delay=0.12, save_path=None, image_dir=None, show=False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    env, frames = rollout_policy(model, src_env, device)
    render_rollout_saliency(env, frames, agent_idx=0, show=False)
    fig, (ax_grid, ax_info) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.tight_layout(pad=2.0)
    if image_dir is not None:
        save_rollout_images(fig, ax_grid, ax_info, env, frames, image_dir)
    if save_path is not None:
        def update(frame_idx):
            draw_rollout_frame(ax_grid, ax_info, env, frames[frame_idx])

        writer = "pillow" if save_path.lower().endswith(".gif") else "ffmpeg"
        FuncAnimation(fig, update, frames=len(frames), interval=delay * 1000, repeat=False).save(save_path, writer=writer)
        plt.close(fig)
    elif show:
        plt.ion()
        plt.show(block=False)
        for frame in frames:
            draw_rollout_frame(ax_grid, ax_info, env, frame)
            fig.canvas.draw_idle()
            plt.pause(delay)
        plt.ioff()
        plt.show()


def main(run_config=None):
    config = run_config or globals()["config"]
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    save_dir = os.path.join(BASE_DIR, "checkpoints", "mappo", "task1", config["run_name"])
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    device = select_device(config)
    first_stage = {**config, **config["curriculum"][0]}
    env = build_env(first_stage)
    describe_env("Initial Quad MAPPO", env)

    model = ActorCritic(
        env.agent_observation_space.shape[0],
        env.global_state_size,
        ACTION_DIM,
        config["hidden_size"],
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    global_steps = 0
    episodes = 0
    final_stage_env = env
    total_stages = len(config["curriculum"])
    start_stage_idx = 0
    completed_stage_count = start_stage_idx
    stage_results = []
    resume_state = infer_resume_state(config, save_dir)
    if resume_state is not None:
        load_checkpoint(model, resume_state["path"], device)
        start_stage_idx = min(resume_state["start_idx"], total_stages)
        global_steps = resume_state["global_steps"]
        if start_stage_idx < total_stages:
            resume_stage = config["curriculum"][start_stage_idx]["name"]
            final_stage_env = build_env({**config, **config["curriculum"][start_stage_idx]})
            print(
                f"Resumed from {resume_state['path']} ({resume_state['reason']}); "
                f"starting at stage {start_stage_idx + 1}/{total_stages}: {resume_stage}"
            )
        else:
            final_stage_env = build_env({**config, **config["curriculum"][-1]})
            print(f"Resumed from {resume_state['path']} ({resume_state['reason']}); curriculum already complete.")

    completed_stage_count = start_stage_idx
    for stage_idx, stage in enumerate(config["curriculum"][start_stage_idx:], start=start_stage_idx + 1):
        stage_config = {**config, **stage}
        env = build_env(stage_config)
        eval_env = build_env(stage_config)
        for group in optimizer.param_groups:
            group["lr"] = stage_config.get("learning_rate", config["learning_rate"])
        apply_stage_exploration(model, stage_config)

        print(f"\n=== Stage {stage_idx}/{total_stages}: {stage['name']} ({stage_mode(stage)}) ===")
        describe_env("Stage", env)
        stage_steps = 0
        stage_best_success = -1.0
        stage_best_reward = float("-inf")
        pre_stage_state = copy.deepcopy(model.state_dict())
        stage_best_state = copy.deepcopy(model.state_dict())
        promoted = False

        while stage_steps < stage["timesteps"]:
            rollout, episode_rewards, rollout_successes, rollout_pickups, rollout_deliveries = collect_rollout(
                env,
                model,
                stage_config,
                device,
            )
            stats = update_model(model, optimizer, rollout, stage_config, device, env.num_drones)
            steps_this_rollout = stage_config["rollout_steps"] * env.num_drones
            stage_steps += steps_this_rollout
            global_steps += steps_this_rollout
            episodes += len(episode_rewards)

            if global_steps % config["progress_print_freq"] < steps_this_rollout:
                print(f"Step {global_steps:>7} | stage_step={stage_steps:>7} | Episodes: {episodes:>5}")

            eval_every = stage.get("eval_every", config["stage_eval_every"])
            should_eval = stage_steps % eval_every < steps_this_rollout or stage_steps >= stage["timesteps"]
            if not should_eval:
                continue

            metrics = evaluate(model, eval_env, stage.get("eval_episodes", config["stage_eval_episodes"]), device)
            print(
                f"Eval @{global_steps:>7} | "
                f"stage_step={stage_steps:>7} | "
                f"sr={metrics['success_rate']:.2%} | "
                f"reward={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f} | "
                f"pickups={metrics['mean_pickups']:.2f} "
                f"deliveries={metrics['mean_deliveries']:.2f} "
                f"delivered_agents={metrics['mean_delivered_agents']:.2f}/{env.num_drones}"
            )
            save_checkpoint(model, config, save_dir, f"model_{stage['name']}_{global_steps}_steps")
            if (
                metrics["success_rate"] > stage_best_success
                or metrics["success_rate"] == stage_best_success
                and metrics["mean_reward"] > stage_best_reward
            ):
                stage_best_success = metrics["success_rate"]
                stage_best_reward = metrics["mean_reward"]
                stage_best_state = copy.deepcopy(model.state_dict())
                save_checkpoint(model, config, save_dir, f"model_{stage['name']}_best")

            if (
                stage_steps >= stage.get("min_timesteps_before_promotion", config["stage_min_timesteps_before_promotion"])
                and metrics["success_rate"] >= stage.get("promotion_success_rate", config["stage_promotion_success_rate"])
            ):
                print(
                    f"Early promotion from {stage['name']}: "
                    f"{metrics['success_rate']:.2%} >= {stage.get('promotion_success_rate', config['stage_promotion_success_rate']):.2%}"
                )
                promoted = True
                break

        if not promoted:
            print(f"Full budget used for {stage['name']}. Best stage success={stage_best_success:.2%}.")
        if stage_best_success <= 0.0:
            stage_results.append(
                {
                    "name": stage["name"],
                    "status": "failed",
                    "best_success_rate": stage_best_success,
                    "best_reward": stage_best_reward,
                    "steps": stage_steps,
                }
            )
            model.load_state_dict(stage_best_state)
            if config["viz_enabled"]:
                failed_viz_path = os.path.join(save_dir, f"failed_{stage['name']}_{global_steps}_steps.gif")
                visualize_trained_policy(
                    model,
                    eval_env,
                    device,
                    delay=config["viz_delay"],
                    save_path=failed_viz_path,
                    image_dir=None,
                    show=False,
                )
                print(f"Saved failed-stage rollout visualization to {failed_viz_path}")
            model.load_state_dict(pre_stage_state)
            print(f"No successful checkpoint found for {stage['name']}; restored pre-stage model.")
            break

        continue_success_rate = stage.get(
            "continue_success_rate",
            stage.get("promotion_success_rate", config["stage_promotion_success_rate"]),
        )
        if not promoted and stage_best_success < continue_success_rate:
            model.load_state_dict(stage_best_state)
            final_stage_env = eval_env
            completed_stage_count = stage_idx
            stage_results.append(
                {
                    "name": stage["name"],
                    "status": "stopped",
                    "best_success_rate": stage_best_success,
                    "best_reward": stage_best_reward,
                    "steps": stage_steps,
                }
            )
            print(
                f"Stopping after {stage['name']}: best sr={stage_best_success:.2%} "
                f"< continue threshold {continue_success_rate:.2%}."
            )
            break

        model.load_state_dict(stage_best_state)
        final_stage_env = eval_env
        completed_stage_count = stage_idx
        stage_results.append(
            {
                "name": stage["name"],
                "status": "promoted" if promoted else "completed",
                "best_success_rate": stage_best_success,
                "best_reward": stage_best_reward,
                "steps": stage_steps,
            }
        )
        print(f"Restored best for {stage['name']}: sr={stage_best_success:.2%} | reward={stage_best_reward:.2f}")

    final_metrics = evaluate(model, final_stage_env, config["stage_eval_episodes"], device)
    print(
        f"Final eval | success_rate={final_metrics['success_rate']:.2%} "
        f"| mean_reward={final_metrics['mean_reward']:.2f} +/- {final_metrics['std_reward']:.2f} "
        f"| pickups={final_metrics['mean_pickups']:.2f} deliveries={final_metrics['mean_deliveries']:.2f} "
        f"| delivered_agents={final_metrics['mean_delivered_agents']:.2f}/{final_stage_env.num_drones} "
    )
    save_checkpoint(model, config, save_dir, "model_final")

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
            device,
            delay=config["viz_delay"],
            save_path=viz_save_path,
            image_dir=viz_image_dir,
            show=config["viz_show"],
        )
        if viz_save_path is not None:
            print(f"Saved Quad MAPPO rollout visualization to {viz_save_path}")

    return {
        "final_metrics": final_metrics,
        "global_steps": global_steps,
        "episodes": episodes,
        "completed_stage_count": completed_stage_count,
        "total_stages": total_stages,
        "final_num_drones": final_stage_env.num_drones,
        "stage_results": stage_results,
    }


if __name__ == "__main__":
    main()
