"""
Visualize a trained DQN policy running in the fixed pickup-and-delivery DroneEnv.

Examples:
    python drone_rl/viz.py
    python drone_rl/viz.py --model drone_rl/checkpoints/dqn/task1/mlp_sparse_v1/model_final.zip
    python drone_rl/viz.py --episodes 5 --delay 0.05
    python drone_rl/viz.py --no-render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from envs.drone_env import DroneEnv
from classical_methods.utils.pipes import PipeGrid, PipeOptions, PipeVisualizerBW


ACTION_NAMES = ["up", "down", "left", "right"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained DQN policy in DroneEnv."
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_DIR / "checkpoints/dqn/task1/mlp_sparse_v1/best/best_model.zip"),
        help="Path to a Stable-Baselines3 DQN checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.12,
        help="Seconds between rendered frames.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Override env max steps for visualization runs.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run episodes headlessly and print stats only.",
    )
    return parser.parse_args()



def build_env(max_steps: int) -> DroneEnv:
    grid_size = (12, 12)
    pipe_grid = PipeGrid(*grid_size)
    pipe_vis = PipeVisualizerBW(lanes=2, base=3)
    bw_map = pipe_vis.render(pipe_grid.to_pipe_ids(PipeOptions()))
    return DroneEnv(bw_map, max_steps=max_steps)



def draw_frame(
    ax_grid,
    ax_info,
    env: DroneEnv,
    trajectory,
    episode_idx: int,
    step_idx: int,
    action: int,
    reward: float,
    total_reward: float,
    done: bool,
    truncated: bool,
) -> None:
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#1a1a1a", "#f2f0e8"])

    ax_grid.clear()
    ax_info.clear()

    ax_grid.imshow(env.bw_map, cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.set_title(
        f"Episode {episode_idx + 1} | Step {step_idx}",
        fontsize=13,
        color="#202020",
        pad=10,
    )

    if len(trajectory) > 1:
        cols = [pos[1] for pos in trajectory]
        rows = [pos[0] for pos in trajectory]
        ax_grid.plot(cols, rows, color="#3f88c5", linewidth=2.0, alpha=0.75)

    start_r, start_c = env.fixed_start
    pkg_r, pkg_c = env.fixed_package
    dst_r, dst_c = env.fixed_delivery
    drone_r, drone_c, has_pkg = env.state

    ax_grid.scatter(start_c, start_r, marker="o", s=80, color="#5b8c5a", label="start", zorder=4)
    if not has_pkg:
        ax_grid.scatter(pkg_c, pkg_r, marker="s", s=110, color="#f2c14e", label="package", zorder=5)
    ax_grid.scatter(dst_c, dst_r, marker="*", s=180, color="#ef476f", label="delivery", zorder=5)

    drone_color = "#ff7f11" if has_pkg else "#118ab2"
    if done:
        drone_color = "#06d6a0"
    elif truncated:
        drone_color = "#8338ec"
    ax_grid.scatter(drone_c, drone_r, marker="o", s=140, color=drone_color, label="drone", zorder=6)

    legend_handles = [
        mpatches.Patch(color="#5b8c5a", label="start"),
        mpatches.Patch(color="#f2c14e", label="package"),
        mpatches.Patch(color="#ef476f", label="delivery"),
        mpatches.Patch(color="#118ab2", label="drone"),
    ]
    ax_grid.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    status = "delivered" if done else "truncated" if truncated else "running"
    goal = "delivery" if has_pkg else "package"
    lines = [
        ("action", ACTION_NAMES[int(action)]),
        ("reward", f"{reward:+.2f}"),
        ("total reward", f"{total_reward:+.2f}"),
        ("position", f"({drone_r}, {drone_c})"),
        ("carrying", "yes" if has_pkg else "no"),
        ("current goal", goal),
        ("start", str(env.fixed_start)),
        ("package", str(env.fixed_package)),
        ("delivery", str(env.fixed_delivery)),
        ("status", status),
    ]

    ax_info.text(0.5, 0.96, "Run Info", ha="center", va="top", fontsize=14, fontweight="bold")
    y = 0.86
    for key, value in lines:
        ax_info.text(0.08, y, key, ha="left", va="top", fontsize=10, color="#666666")
        ax_info.text(0.92, y, value, ha="right", va="top", fontsize=10, color="#111111")
        y -= 0.085



def run_episode(model: DQN, env: DroneEnv, episode_idx: int, render_ctx) -> dict:
    obs, _ = env.reset()
    trajectory = [(env.state[0], env.state[1])]
    total_reward = 0.0
    picked_up = False
    done = False
    truncated = False
    steps = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        r, c, has_pkg = env.state
        trajectory.append((r, c))
        picked_up = picked_up or bool(has_pkg)

        if render_ctx is not None:
            fig, ax_grid, ax_info, delay, plt = render_ctx
            draw_frame(
                ax_grid=ax_grid,
                ax_info=ax_info,
                env=env,
                trajectory=trajectory,
                episode_idx=episode_idx,
                step_idx=steps,
                action=int(action),
                reward=reward,
                total_reward=total_reward,
                done=done,
                truncated=truncated,
            )
            fig.canvas.draw_idle()
            plt.pause(delay)

    return {
        "episode": episode_idx + 1,
        "steps": steps,
        "total_reward": total_reward,
        "picked_up": picked_up,
        "delivered": done,
        "truncated": truncated,
        "final_position": (env.state[0], env.state[1]),
    }



def print_summary(results: list[dict]) -> None:
    delivered = sum(item["delivered"] for item in results)
    picked_up = sum(item["picked_up"] for item in results)
    truncated = sum(item["truncated"] for item in results)
    avg_reward = float(np.mean([item["total_reward"] for item in results]))
    avg_steps = float(np.mean([item["steps"] for item in results]))

    print("\nSummary")
    print("-" * 40)
    print(f"Episodes:       {len(results)}")
    print(f"Delivered:      {delivered}/{len(results)}")
    print(f"Picked up:      {picked_up}/{len(results)}")
    print(f"Truncated:      {truncated}/{len(results)}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps:  {avg_steps:.1f}")



def main() -> None:
    args = parse_args()

    env = build_env(max_steps=args.max_steps)
    model = DQN.load(args.model, env=env)

    print(f"Loaded model: {args.model}")
    print(f"Start:        {env.fixed_start}")
    print(f"Package:      {env.fixed_package}")
    print(f"Delivery:     {env.fixed_delivery}")
    print(f"Lane cells:   {len(env.lane_coords)}")

    render_ctx = None
    if not args.no_render:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, (ax_grid, ax_info) = plt.subplots(
            1,
            2,
            figsize=(13, 6),
            gridspec_kw={"width_ratios": [2.2, 1]},
        )
        fig.tight_layout(pad=2.0)
        render_ctx = (fig, ax_grid, ax_info, args.delay, plt)
        plt.show(block=False)

    results = []
    for episode_idx in range(args.episodes):
        stats = run_episode(model, env, episode_idx, render_ctx)
        results.append(stats)
        outcome = "DELIVERED" if stats["delivered"] else "TRUNCATED"
        print(
            f"Ep {stats['episode']:>2} | {outcome:<9} | steps={stats['steps']:>4} | "
            f"reward={stats['total_reward']:>8.2f} | final={stats['final_position']}"
        )

    print_summary(results)

    if render_ctx is not None:
        render_ctx[-1].ioff()
        render_ctx[-1].show()


if __name__ == "__main__":
    main()
