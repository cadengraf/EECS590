#!/usr/bin/env python3

import os
import sys
import json
import random

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


try:
    from ppo import build_env, describe_env, is_delivery_success, visualize_trained_policy
except ImportError:
    from quad_physics.ppo import build_env, describe_env, is_delivery_success, visualize_trained_policy


try:
    from sac_config import config
except ImportError:
    from quad_physics.sac_config import config


def make_env(stage_cfg):
    env = build_env(stage_cfg, print_freq=stage_cfg.get("env_print_freq", 0))

    # No action wrapper needed.
    # SAC supports continuous Box actions directly.
    env = Monitor(env)

    return env


def make_model(env, stage_cfg):
    policy_kwargs = {
        "net_arch": stage_cfg.get("net_arch", config.get("net_arch", [128, 128])),
    }

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=stage_cfg.get("learning_rate", config["learning_rate"]),
        buffer_size=stage_cfg.get("buffer_size", config["buffer_size"]),
        learning_starts=stage_cfg.get("learning_starts", config["learning_starts"]),
        batch_size=stage_cfg.get("batch_size", config["batch_size"]),
        gamma=stage_cfg.get("gamma", config["gamma"]),
        train_freq=stage_cfg.get("train_freq", config["train_freq"]),
        gradient_steps=stage_cfg.get("gradient_steps", config["gradient_steps"]),
        tau=stage_cfg.get("tau", config["tau"]),
        ent_coef=stage_cfg.get("ent_coef", config["ent_coef"]),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=config["seed"],
        device="auto",
    )

    return model


def get_has_package(base_env):
    """
    Tries to detect whether the drone has picked up the package.
    This is only for printing/debugging.
    """

    possible_names = [
        "has_package",
        "has_pkg",
        "picked_up",
        "package_picked",
        "package_picked_up",
    ]

    for name in possible_names:
        if hasattr(base_env, name):
            return bool(getattr(base_env, name))

    return False


def evaluate(model, stage_cfg, n_episodes=20):
    eval_cfg = {**stage_cfg, "env_print_freq": 0}
    env = make_env(eval_cfg)

    rewards = []
    successes = 0
    pickups = 0
    lengths = []

    pickup_reward = stage_cfg.get("pickup_reward", config["pickup_reward"])
    delivery_reward = stage_cfg.get("delivery_reward", config["delivery_reward"])

    print()
    print("Evaluation episodes:")
    print("-" * 100)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)

        done = False
        truncated = False
        total_reward = 0.0
        picked_up = False
        delivered = False
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)

            reward = float(reward)
            total_reward += reward
            steps += 1

            # Detect real pickup/delivery using reward spikes.
            # This avoids relying on state[2], which was misleading for DQN.
            if reward >= pickup_reward * 0.5:
                picked_up = True

            if reward >= delivery_reward * 0.5:
                delivered = True

            base_env = env.unwrapped
            picked_up = picked_up or get_has_package(base_env)

        base_env = env.unwrapped
        success = delivered or is_delivery_success(base_env, done)

        rewards.append(total_reward)
        successes += int(success)
        pickups += int(picked_up)

        if hasattr(base_env, "current_step"):
            lengths.append(base_env.current_step)
        else:
            lengths.append(steps)

        print(
            f"Ep {ep + 1:02d} | "
            f"pickup={'YES' if picked_up else 'NO '} | "
            f"delivery={'YES' if success else 'NO '} | "
            f"reward={total_reward:9.2f} | "
            f"steps={steps:4d}"
        )

    print("-" * 100)

    return {
        "success_rate": successes / n_episodes,
        "pickup_rate": pickups / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def print_stage_summary(stage_name, metrics):
    print()
    print("=" * 80)
    print(f"STAGE DONE: {stage_name}")
    print(f"Pickup rate:   {metrics['pickup_rate']:.2%}")
    print(f"Delivery rate: {metrics['success_rate']:.2%}")
    print(f"Mean reward:   {metrics['mean_reward']:.2f}")
    print(f"Mean length:   {metrics['mean_length']:.1f}")
    print("=" * 80)
    print()


def main():
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    save_dir = os.path.join(
        BASE_DIR,
        "checkpoints",
        "sac",
        "task1",
        config["run_name"],
    )

    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    model = None

    best_success = -1.0
    best_reward = float("-inf")
    best_path = None

    for stage_idx, stage in enumerate(config["curriculum"], start=1):
        stage_cfg = {**config, **stage}

        print()
        print("=" * 80)
        print(f"STARTING STAGE {stage_idx}/{len(config['curriculum'])}: {stage['name']}")
        print("=" * 80)

        raw_env = build_env(stage_cfg, print_freq=0)
        describe_env("SAC stage", raw_env)

        env = DummyVecEnv([lambda cfg=stage_cfg: make_env(cfg)])

        if model is None:
            model = make_model(env, stage_cfg)
        else:
            model.set_env(env)

            if config.get("reset_replay_each_stage", True):
                if model.replay_buffer is not None:
                    model.replay_buffer.reset()

        model.learn(
            total_timesteps=stage["timesteps"],
            reset_num_timesteps=True,
        )

        metrics = evaluate(
            model,
            stage_cfg,
            n_episodes=stage_cfg.get(
                "stage_eval_episodes",
                config["stage_eval_episodes"],
            ),
        )

        print_stage_summary(stage["name"], metrics)

        stage_path = os.path.join(save_dir, f"{stage['name']}.zip")
        model.save(stage_path)
        print(f"Saved stage model: {stage_path}")

        if (
            metrics["success_rate"] > best_success
            or (
                metrics["success_rate"] == best_success
                and metrics["mean_reward"] > best_reward
            )
        ):
            best_success = metrics["success_rate"]
            best_reward = metrics["mean_reward"]
            best_path = os.path.join(save_dir, "model_best.zip")
            model.save(best_path)
            print(f"Saved new best model: {best_path}")

    final_path = os.path.join(save_dir, "model_final.zip")
    model.save(final_path)

    print()
    print("=" * 80)
    print(f"Final model saved to: {final_path}")

    if best_path is not None:
        print(f"Best model saved to:  {best_path}")
        print(f"Best success rate:    {best_success:.2%}")
        print(f"Best mean reward:     {best_reward:.2f}")

    if config.get("viz_enabled", False) and model is not None:
        final_env = build_env({**config, **config["curriculum"][-1]}, print_freq=0)
        viz_save_path = config.get("viz_save_path", "sac_quad_rollout.gif")
        if viz_save_path is not None and not os.path.isabs(viz_save_path):
            viz_save_path = os.path.join(save_dir, viz_save_path)
        viz_image_dir = config.get("viz_image_dir") if config.get("viz_save_images") else None
        if viz_image_dir is not None and not os.path.isabs(viz_image_dir):
            viz_image_dir = os.path.join(save_dir, viz_image_dir)
        visualize_trained_policy(
            model,
            final_env,
            delay=config.get("viz_delay", 0.12),
            save_path=viz_save_path,
            image_dir=viz_image_dir,
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
