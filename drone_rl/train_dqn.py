import json as _json
import os as _os
import sys as _sys

import numpy as _np
import torch as _torch
from stable_baselines3 import DQN as _DQN
from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
from stable_baselines3.common.utils import LinearSchedule as _LinearSchedule

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _THIS_DIR not in _sys.path:
    _sys.path.insert(0, _THIS_DIR)

try:
    from classical_methods.utils.saliency import run_saliency_suite as _render_saliency_suite
    from ppo import (
        ACTION_NAMES as _ACTION_NAMES,
        build_env as _build_env,
        clone_env as _clone_env,
        describe_env as _describe_env,
        draw_rollout_frame as _draw_rollout_frame,
        is_delivery_success as _is_delivery_success,
        rollout_policy as _ppo_rollout_policy,
        _state_to_obs as _state_to_obs,
    )
except ImportError:
    from drone_rl.classical_methods.utils.saliency import run_saliency_suite as _render_saliency_suite
    from drone_rl.ppo import (
        ACTION_NAMES as _ACTION_NAMES,
        build_env as _build_env,
        clone_env as _clone_env,
        describe_env as _describe_env,
        draw_rollout_frame as _draw_rollout_frame,
        is_delivery_success as _is_delivery_success,
        rollout_policy as _ppo_rollout_policy,
        _state_to_obs as _state_to_obs,
    )


_DQN_MODEL_KEYS = (
    "learning_rate",
    "gamma",
    "batch_size",
    "buffer_size",
    "learning_starts",
    "train_freq",
    "target_update_interval",
    "exploration_fraction",
    "exploration_initial_eps",
    "exploration_final_eps",
)


_dqn_config = {
    "run_name": "dqn_curriculum_v3",
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "batch_size": 128,
    "buffer_size": 100_000,
    "learning_starts": 10_000,
    "train_freq": 4,
    "target_update_interval": 2_000,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "exploration_fraction": 0.40,
    "net_arch": [256, 256],
    "loop_prob": 0.25,
    "step_penalty": -0.2,
    "invalid_move_penalty": 200.0,
    "revisit_penalty": 0.05,
    "revisit_penalty_cap": 1.5,
    "backtrack_penalty": 1.0,
    "pickup_reward": 300.0,
    "delivery_reward": 1000.0,
    "progress_print_freq": 10_000,
    "qdiag_freq": 25_000,
    "env_print_freq": 20_000,
    "stage_eval_every": 25_000,
    "stage_eval_episodes": 20,
    "stage_promotion_success_rate": 0.70,
    "stage_min_timesteps_before_promotion": 50_000,
    "viz_delay": 0.12,
    "viz_enabled": True,
    "viz_save_path": "dqn_rollout.gif",
    "saliency_enabled": True,
    "curriculum": [
        {
            "name": "s1_line_two_rewards",
            "map_layout": "straight_line",
            "line_length": 4,
            "map_padding": 1,
            "max_steps": 8,
            "timesteps": 40_000,
            "fixed_start": [1, 1],
            "fixed_package": [1, 2],
            "fixed_delivery": [1, 3],
            "randomize_package": False,
            "randomize_delivery": False,
            "eval_every": 5_000,
            "min_timesteps_before_promotion": 10_000,
            "promotion_success_rate": 0.90,
            "exploration_fraction": 0.25,
            "revisit_penalty": 0.0,
            "revisit_penalty_cap": 0.0,
            "backtrack_penalty": 0.2,
        },
        {
            "name": "s2_line_longer",
            "map_layout": "straight_line",
            "line_length": 7,
            "map_padding": 1,
            "max_steps": 14,
            "timesteps": 80_000,
            "fixed_start": [1, 1],
            "fixed_package": [1, 3],
            "fixed_delivery": [1, 6],
            "randomize_package": False,
            "randomize_delivery": False,
            "eval_every": 10_000,
            "min_timesteps_before_promotion": 20_000,
            "promotion_success_rate": 0.85,
            "exploration_fraction": 0.30,
            "revisit_penalty": 0.01,
            "revisit_penalty_cap": 0.2,
            "backtrack_penalty": 0.3,
        },
        {
            "name": "s3_t_junction",
            "map_layout": "t_junction",
            "stem_length": 5,
            "branch_left": 2,
            "branch_right": 2,
            "map_padding": 1,
            "max_steps": 30,
            "timesteps": 150_000,
            "fixed_start": [5, 3],
            "fixed_package": [3, 3],
            "fixed_delivery": [1, 5],
            "randomize_package": True,
            "randomize_delivery": False,
            "eval_every": 15_000,
            "min_timesteps_before_promotion": 40_000,
            "promotion_success_rate": 0.75,
            "exploration_initial_eps": 0.8,
            "exploration_fraction": 0.35,
            "step_penalty": -0.1,
            "revisit_penalty": 0.02,
            "revisit_penalty_cap": 0.5,
            "backtrack_penalty": 0.5,
            "clear_buffer_on_start": True,
        },
        {
            "name": "s4_random_package_small_pipe",
            "grid_size": [3, 3],
            "max_steps": 180,
            "timesteps": 180_000,
            "map_seed": 101,
            "package_min_dist": 3,
            "package_max_dist": 7,
            "delivery_min_dist": 3,
            "delivery_max_dist": 7,
            "delivery_min_start_dist": 5,
            "randomize_package": True,
            "randomize_delivery": False,
            "eval_every": 20_000,
            "min_timesteps_before_promotion": 60_000,
            "promotion_success_rate": 0.75,
            "exploration_initial_eps": 0.6,
            "exploration_fraction": 0.40,
            "clear_buffer_on_start": True,
        },
        {
            "name": "s5_random_both_small_pipe",
            "grid_size": [4, 4],
            "max_steps": 300,
            "timesteps": 250_000,
            "map_seed": 101,
            "package_min_dist": 5,
            "package_max_dist": 10,
            "delivery_min_dist": 4,
            "delivery_max_dist": 9,
            "delivery_min_start_dist": 8,
            "randomize_package": True,
            "randomize_delivery": True,
            "eval_every": 25_000,
            "min_timesteps_before_promotion": 75_000,
            "promotion_success_rate": 0.75,
            "exploration_initial_eps": 0.5,
            "exploration_fraction": 0.45,
            "clear_buffer_on_start": True,
        },
        {
            "name": "s6_random_both_medium_pipe",
            "grid_size": [6, 6],
            "max_steps": 500,
            "timesteps": 350_000,
            "map_seed": 202,
            "package_min_dist": 8,
            "package_max_dist": 16,
            "delivery_min_dist": 7,
            "delivery_max_dist": 16,
            "delivery_min_start_dist": 14,
            "randomize_package": True,
            "randomize_delivery": True,
            "eval_every": 25_000,
            "min_timesteps_before_promotion": 100_000,
            "promotion_success_rate": 0.75,
            "exploration_initial_eps": 0.45,
            "exploration_fraction": 0.50,
            "clear_buffer_on_start": True,
            "learning_starts": 20_000,
        },
        {
            "name": "s7_random_both_large_pipe",
            "grid_size": [8, 8],
            "max_steps": 900,
            "timesteps": 500_000,
            "map_seed": 303,
            "package_min_dist": 10,
            "package_max_dist": 20,
            "delivery_min_dist": 8,
            "delivery_max_dist": 18,
            "delivery_min_start_dist": 16,
            "randomize_package": True,
            "randomize_delivery": True,
            "eval_every": 25_000,
            "min_timesteps_before_promotion": 125_000,
            "promotion_success_rate": 0.75,
            "exploration_initial_eps": 0.40,
            "exploration_fraction": 0.55,
            "clear_buffer_on_start": True,
            "learning_starts": 25_000,
        },
    ],
}
_dqn_config["total_timesteps"] = sum(stage["timesteps"] for stage in _dqn_config["curriculum"])


class _DQNProgressCallback(_BaseCallback):
    def __init__(self, print_freq=10_000):
        super().__init__()
        self.print_freq = print_freq
        self.episodes = 0

    def _on_step(self) -> bool:
        self.episodes += int(_np.sum(self.locals.get("dones", [])))
        if self.num_timesteps % self.print_freq == 0:
            print(
                f"Step {self.num_timesteps:>7} | Episodes: {self.episodes:>5} "
                f"| eps={self.model.exploration_rate:.3f}"
            )
        return True


class _DQNEvalCallback(_BaseCallback):
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

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every != 0:
            return True

        metrics = _dqn_evaluate(self.model, self.source_env, self.n_eval_episodes)
        print(
            f"Eval @{self.num_timesteps:>7} | "
            f"sr={metrics['success_rate']:.2%} | "
            f"pickups={metrics['pickup_rate']:.2%} | "
            f"reward={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}"
        )
        self.model.save(_os.path.join(self.save_dir, f"model_{self.stage_name}_{self.num_timesteps}_steps"))

        if self.num_timesteps >= self.min_timesteps and metrics["success_rate"] >= self.promotion_threshold:
            print(
                f"Early promotion from {self.stage_name}: "
                f"{metrics['success_rate']:.2%} >= {self.promotion_threshold:.2%}"
            )
            self.promoted = True
            return False
        return True


class _DQNQDiagnosticCallback(_BaseCallback):
    def __init__(self, env, check_freq=25_000):
        super().__init__()
        self.diag_env = env
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True
        obs, _ = self.diag_env.reset()
        obs_tensor = _torch.as_tensor(obs[None], dtype=_torch.float32, device=self.model.device)
        with _torch.no_grad():
            q_values = self.model.q_net(obs_tensor).cpu().numpy()[0]
        print(f"[Q-diag {self.num_timesteps}] Q={_np.round(q_values, 3)}")
        return True


def _dqn_evaluate(model, env, n_episodes=20):
    eval_env = _clone_env(env)
    rewards = []
    successes = 0
    pickups = 0
    for seed in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed)
        done = truncated = False
        total_reward = 0.0
        picked_up = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward
            picked_up = picked_up or bool(eval_env.state[2])
        rewards.append(total_reward)
        pickups += int(picked_up)
        successes += int(_is_delivery_success(eval_env, done))
    return {
        "success_rate": successes / n_episodes,
        "pickup_rate": pickups / n_episodes,
        "mean_reward": float(_np.mean(rewards)),
        "std_reward": float(_np.std(rewards)),
    }


def _dqn_run_saliency_suite(model, env, show=True):
    rollout_env, path = _ppo_rollout_policy(model, env)

    class DQNSaliencyAdapter:
        def __init__(self, sb3_model, rollout_env):
            self.actions = rollout_env.actions
            self.lane_coords = rollout_env.lane_coords
            self.package_pos = rollout_env.fixed_package
            self.delivery_pos = rollout_env.fixed_delivery
            self.Q = {}

            with _torch.no_grad():
                for r, c in rollout_env.lane_coords:
                    for has_pkg in (False, True):
                        state = (r, c, has_pkg)
                        obs = _state_to_obs(rollout_env, state)
                        obs_tensor = _torch.as_tensor(
                            obs[None],
                            dtype=_torch.float32,
                            device=sb3_model.device,
                        )
                        self.Q[state] = sb3_model.q_net(obs_tensor).cpu().numpy()[0].copy()

    _render_saliency_suite(
        DQNSaliencyAdapter(model, rollout_env),
        [tuple(frame["state"]) for frame in path],
        rollout_env.bw_map,
        show=show,
    )


def _dqn_visualize_trained_policy(model, env, delay=0.12, save_path=None):
    import matplotlib.pyplot as plt

    rollout_env, frames = _ppo_rollout_policy(model, env)
    fig, (ax_g, ax_i) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.tight_layout(pad=2.0)

    if save_path is not None:
        from matplotlib.animation import FuncAnimation

        def update(frame_idx):
            _draw_rollout_frame(ax_g, ax_i, rollout_env, frames[frame_idx])

        anim = FuncAnimation(fig, update, frames=len(frames), interval=delay * 1000, repeat=False)
        anim.save(save_path, writer="ffmpeg")
        plt.close(fig)
        return

    plt.ion()
    plt.show(block=False)
    for frame in frames:
        _draw_rollout_frame(ax_g, ax_i, rollout_env, frame)
        fig.canvas.draw_idle()
        plt.pause(delay)
    plt.ioff()
    plt.show()


def _make_dqn_model(env):
    return _DQN(
        "MlpPolicy",
        env,
        **{key: _dqn_config[key] for key in _DQN_MODEL_KEYS},
        policy_kwargs={"net_arch": _dqn_config["net_arch"]},
        verbose=0,
        device="cpu",
    )


def _configure_dqn_stage_model(model, stage):
    if stage.get("clear_buffer_on_start"):
        model.replay_buffer.reset()
        print(f"Replay buffer cleared for {stage['name']}")

    model.learning_starts = stage.get("learning_starts", _dqn_config["learning_starts"])
    initial_eps = stage.get("exploration_initial_eps", _dqn_config["exploration_initial_eps"])
    final_eps = stage.get("exploration_final_eps", _dqn_config["exploration_final_eps"])
    fraction = stage.get("exploration_fraction", _dqn_config["exploration_fraction"])
    model.exploration_initial_eps = initial_eps
    model.exploration_final_eps = final_eps
    model.exploration_fraction = fraction
    model.exploration_schedule = _LinearSchedule(initial_eps, final_eps, fraction)
    model.exploration_rate = initial_eps
    print(f"Exploration: init={initial_eps:.2f} final={final_eps:.2f} fraction={fraction:.2f}")


def _dqn_stage_mode(stage):
    if stage.get("randomize_package") and stage.get("randomize_delivery"):
        return "random package+delivery"
    if stage.get("randomize_package"):
        return "random package"
    return "fixed positions"


def _make_dqn_promotion_callback(stage, stage_env, save_dir):
    return _DQNEvalCallback(
        stage_name=stage["name"],
        source_env=stage_env,
        eval_every=stage.get("eval_every", _dqn_config["stage_eval_every"]),
        n_eval_episodes=stage.get("eval_episodes", _dqn_config["stage_eval_episodes"]),
        promotion_threshold=stage.get("promotion_success_rate", _dqn_config["stage_promotion_success_rate"]),
        min_timesteps_before_promotion=stage.get(
            "min_timesteps_before_promotion",
            _dqn_config["stage_min_timesteps_before_promotion"],
        ),
        save_dir=save_dir,
    )


def main():
    save_dir = _os.path.join("checkpoints", "dqn", "task1", _dqn_config["run_name"])
    _os.makedirs(save_dir, exist_ok=True)

    with open(_os.path.join(save_dir, "config.json"), "w") as f:
        _json.dump(_dqn_config, f, indent=4)

    first_stage = {**_dqn_config, **_dqn_config["curriculum"][0]}
    env = _build_env(first_stage, print_freq=_dqn_config["env_print_freq"])
    _describe_env("Initial", env)
    model = _make_dqn_model(env)

    final_stage_env = env
    total_stages = len(_dqn_config["curriculum"])
    for stage_idx, stage in enumerate(_dqn_config["curriculum"], start=1):
        stage_config = {**_dqn_config, **stage}
        stage_env = _build_env(stage_config, print_freq=_dqn_config["env_print_freq"])
        final_stage_env = stage_env

        print(f"\n=== Stage {stage_idx}/{total_stages}: {stage['name']} ({_dqn_stage_mode(stage)}) ===")
        _describe_env("Stage", stage_env)
        model.set_env(stage_env)
        _configure_dqn_stage_model(model, stage)
        promotion_cb = _make_dqn_promotion_callback(stage, stage_env, save_dir)

        model.learn(
            total_timesteps=stage["timesteps"],
            reset_num_timesteps=True,
            callback=[
                _DQNProgressCallback(_dqn_config["progress_print_freq"]),
                _DQNQDiagnosticCallback(stage_env, _dqn_config["qdiag_freq"]),
                promotion_cb,
            ],
        )

        if not promotion_cb.promoted:
            print(f"Full budget used for {stage['name']}.")

    metrics = _dqn_evaluate(model, final_stage_env, _dqn_config["stage_eval_episodes"])
    print(
        f"Final eval | success_rate={metrics['success_rate']:.2%} "
        f"| pickup_rate={metrics['pickup_rate']:.2%} "
        f"| mean_reward={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}"
    )

    model.save(_os.path.join(save_dir, "model_final"))
    if _dqn_config["saliency_enabled"]:
        _dqn_run_saliency_suite(model, final_stage_env, show=True)

    if _dqn_config["viz_enabled"]:
        viz_save_path = _dqn_config["viz_save_path"]
        if viz_save_path is not None and not _os.path.isabs(viz_save_path):
            viz_save_path = _os.path.join(save_dir, viz_save_path)
        _dqn_visualize_trained_policy(model, final_stage_env, delay=_dqn_config["viz_delay"], save_path=viz_save_path)


if __name__ == "__main__":
    main()
    raise SystemExit
