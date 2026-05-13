import json
import os
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

try:
    from classical_methods.utils.pipes import PipeGrid, PipeOptions, PipeVisualizerBW
except ImportError:
    from drone_rl.classical_methods.utils.pipes import PipeGrid, PipeOptions, PipeVisualizerBW


ACTION_NAMES = ["up", "down", "left", "right"]
ENV_DEFAULTS = {
    "num_drones": 2,
    "max_steps": 300,
    "randomize_package": False,
    "randomize_deliveries": False,
    "step_penalty": -0.5,
    "invalid_move_penalty": 20.0,
    "collision_penalty": 20.0,
    "revisit_penalty": 0.08,
    "revisit_penalty_cap": 2.0,
    "pickup_reward": 150.0,
    "delivery_reward": 800.0,
    "team_delivery_bonus": 1500.0,
}


config = {
    **ENV_DEFAULTS,
    "run_name": "mappo_curriculum_two_drones_v5_ppo_style",
    "seed": 7,
    "num_drones": 2,
    "rollout_steps": 512,
    "epochs": 6,
    "minibatch_size": 256,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "hidden_size": 128,
    "eval_every": 25_000,
    "eval_episodes": 20,
    "progress_print_freq": 10_000,
    "viz_enabled": True,
    "viz_delay": 0.18,
    "viz_save_path": "mappo_rollout.gif",
    "viz_show": False,
    "stage_eval_every": 25_000,
    "stage_eval_episodes": 20,
    "stage_promotion_success_rate": 0.70,
    "stage_min_timesteps_before_promotion": 50_000,
    "curriculum": [
        {
            "name": "s1_short_t_fixed",
            "map_layout": "t_junction",
            "stem_length": 4,
            "branch_left": 1,
            "branch_right": 1,
            "map_padding": 1,
            "max_steps": 24,
            "timesteps": 180_000,
            "fixed_starts": [[4, 2], [4, 2]],
            "fixed_package": [2, 2],
            "fixed_deliveries": [[1, 1], [1, 3]],
            "randomize_package": False,
            "randomize_deliveries": False,
            "eval_every": 20_000,
            "min_timesteps_before_promotion": 80_000,
            "promotion_success_rate": 0.90,
            "collision_penalty": 8.0,
            "revisit_penalty": 0.0,
            "revisit_penalty_cap": 0.0,
        },
        {
            "name": "s2_wider_t_fixed",
            "map_layout": "t_junction",
            "stem_length": 6,
            "branch_left": 3,
            "branch_right": 3,
            "map_padding": 1,
            "max_steps": 60,
            "timesteps": 240_000,
            "fixed_starts": [[6, 4], [6, 4]],
            "fixed_package": [3, 4],
            "fixed_deliveries": [[1, 1], [1, 7]],
            "randomize_package": False,
            "randomize_deliveries": False,
            "eval_every": 25_000,
            "min_timesteps_before_promotion": 100_000,
            "promotion_success_rate": 0.80,
        },
        {
            "name": "s3_wider_t_random_package",
            "map_layout": "t_junction",
            "stem_length": 7,
            "branch_left": 4,
            "branch_right": 4,
            "map_padding": 1,
            "max_steps": 90,
            "timesteps": 300_000,
            "package_min_dist": 2,
            "package_max_dist": 5,
            "delivery_min_dist": 3,
            "delivery_max_dist": 8,
            "randomize_package": True,
            "randomize_deliveries": False,
            "eval_every": 30_000,
            "min_timesteps_before_promotion": 120_000,
            "promotion_success_rate": 0.75,
        },
        {
            "name": "s4_small_pipe_fixed",
            "grid_size": [3, 3],
            "map_seed": 101,
            "loop_prob": 0.25,
            "max_steps": 130,
            "timesteps": 220_000,
            "package_min_dist": 3,
            "package_max_dist": 7,
            "delivery_min_dist": 3,
            "delivery_max_dist": 9,
            "randomize_package": False,
            "randomize_deliveries": False,
            "eval_every": 25_000,
            "min_timesteps_before_promotion": 90_000,
            "promotion_success_rate": 0.75,
        },
        {
            "name": "s5_small_pipe_random_package",
            "grid_size": [3, 3],
            "map_seed": 101,
            "loop_prob": 0.25,
            "max_steps": 150,
            "timesteps": 300_000,
            "package_min_dist": 3,
            "package_max_dist": 7,
            "delivery_min_dist": 3,
            "delivery_max_dist": 9,
            "randomize_package": True,
            "randomize_deliveries": False,
            "eval_every": 30_000,
            "min_timesteps_before_promotion": 120_000,
            "promotion_success_rate": 0.70,
        },
        {
            "name": "s6_small_pipe_random_package_delivery",
            "grid_size": [3, 3],
            "map_seed": 101,
            "loop_prob": 0.25,
            "max_steps": 180,
            "timesteps": 450_000,
            "package_min_dist": 3,
            "package_max_dist": 7,
            "delivery_min_dist": 3,
            "delivery_max_dist": 10,
            "randomize_package": True,
            "randomize_deliveries": True,
            "eval_every": 45_000,
            "min_timesteps_before_promotion": 180_000,
            "promotion_success_rate": 0.65,
        },
        {
            "name": "s7_medium_pipe_fixed",
            "grid_size": [4, 4],
            "map_seed": 202,
            "loop_prob": 0.30,
            "max_steps": 210,
            "timesteps": 320_000,
            "package_min_dist": 5,
            "package_max_dist": 12,
            "delivery_min_dist": 5,
            "delivery_max_dist": 16,
            "randomize_package": False,
            "randomize_deliveries": False,
            "eval_every": 40_000,
            "min_timesteps_before_promotion": 140_000,
            "promotion_success_rate": 0.70,
        },
        {
            "name": "s8_medium_pipe_random_package",
            "grid_size": [4, 4],
            "map_seed": 202,
            "loop_prob": 0.30,
            "max_steps": 240,
            "timesteps": 450_000,
            "package_min_dist": 5,
            "package_max_dist": 12,
            "delivery_min_dist": 5,
            "delivery_max_dist": 16,
            "randomize_package": True,
            "randomize_deliveries": False,
            "eval_every": 45_000,
            "min_timesteps_before_promotion": 180_000,
            "promotion_success_rate": 0.65,
        },
        {
            "name": "s9_medium_pipe_random_package_delivery",
            "grid_size": [4, 4],
            "map_seed": 202,
            "loop_prob": 0.30,
            "max_steps": 280,
            "timesteps": 700_000,
            "package_min_dist": 5,
            "package_max_dist": 12,
            "delivery_min_dist": 5,
            "delivery_max_dist": 16,
            "randomize_package": True,
            "randomize_deliveries": True,
            "eval_every": 50_000,
            "min_timesteps_before_promotion": 260_000,
            "promotion_success_rate": 0.60,
        },
        {
            "name": "s10_large_pipe_fixed",
            "grid_size": [6, 6],
            "map_seed": 303,
            "loop_prob": 0.30,
            "max_steps": 420,
            "timesteps": 600_000,
            "package_min_dist": 8,
            "package_max_dist": 16,
            "delivery_min_dist": 7,
            "delivery_max_dist": 18,
            "randomize_package": False,
            "randomize_deliveries": False,
            "eval_every": 60_000,
            "min_timesteps_before_promotion": 240_000,
            "promotion_success_rate": 0.60,
        },
        {
            "name": "s11_large_pipe_random_package",
            "grid_size": [6, 6],
            "map_seed": 303,
            "loop_prob": 0.30,
            "max_steps": 500,
            "timesteps": 800_000,
            "package_min_dist": 8,
            "package_max_dist": 18,
            "delivery_min_dist": 7,
            "delivery_max_dist": 20,
            "randomize_package": True,
            "randomize_deliveries": False,
            "eval_every": 75_000,
            "min_timesteps_before_promotion": 320_000,
            "promotion_success_rate": 0.55,
        },
        {
            "name": "s12_large_pipe_random_package_delivery",
            "grid_size": [6, 6],
            "map_seed": 303,
            "loop_prob": 0.30,
            "max_steps": 620,
            "timesteps": 1_000_000,
            "package_min_dist": 8,
            "package_max_dist": 20,
            "delivery_min_dist": 8,
            "delivery_max_dist": 24,
            "randomize_package": True,
            "randomize_deliveries": True,
            "eval_every": 100_000,
            "min_timesteps_before_promotion": 420_000,
            "promotion_success_rate": 0.50,
        },
        {
            "name": "s13_xlarge_pipe_random_package_delivery",
            "grid_size": [8, 8],
            "map_seed": 404,
            "loop_prob": 0.35,
            "max_steps": 900,
            "timesteps": 1_200_000,
            "package_min_dist": 10,
            "package_max_dist": 26,
            "delivery_min_dist": 10,
            "delivery_max_dist": 32,
            "randomize_package": True,
            "randomize_deliveries": True,
            "eval_every": 120_000,
            "min_timesteps_before_promotion": 500_000,
            "promotion_success_rate": 0.45,
        },
    ],
}
config["total_timesteps"] = sum(stage["timesteps"] for stage in config["curriculum"])


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


def shortest_path_distances(lane_set, actions, start):
    queue = deque([(start, 0)])
    distances = {start: 0}
    while queue:
        (r, c), dist = queue.popleft()
        for dr, dc in actions:
            nxt = (r + dr, c + dc)
            if nxt in lane_set and nxt not in distances:
                distances[nxt] = dist + 1
                queue.append((nxt, dist + 1))
    return distances


def positions_in_distance_range(candidates, min_dist, max_dist):
    return [as_position(pos) for pos, dist in candidates if min_dist <= dist <= max_dist]


def pick_position_by_distance(candidates, min_dist, max_dist, fallback_index=-1):
    valid = [pos for pos, dist in candidates if min_dist <= dist <= max_dist]
    if valid:
        return as_position(valid[fallback_index])
    return as_position(candidates[fallback_index][0])


def configure_positions(env, cfg):
    has_fixed = all(key in cfg for key in ("fixed_starts", "fixed_package", "fixed_deliveries"))
    if has_fixed:
        env.fixed_starts = [as_position(pos) for pos in cfg["fixed_starts"]]
        env.fixed_package = as_position(cfg["fixed_package"])
        env.fixed_deliveries = [as_position(pos) for pos in cfg["fixed_deliveries"]]
        env.random_package_candidates = None
        env.random_delivery_candidates = None
        return

    start = as_position(min(env.lane_coords, key=lambda pos: (pos[0] + pos[1], pos[0], pos[1])))
    start_dists = shortest_path_distances(env.lane_set, env.actions, start)
    reachable = sorted(
        ((pos, dist) for pos, dist in start_dists.items() if pos != start),
        key=lambda item: (item[1], item[0][0], item[0][1]),
    )
    package = pick_position_by_distance(
        reachable,
        cfg.get("package_min_dist", 1),
        cfg.get("package_max_dist", 9999),
    )
    env.random_package_candidates = positions_in_distance_range(
        reachable,
        cfg.get("package_min_dist", 1),
        cfg.get("package_max_dist", 9999),
    )

    package_dists = shortest_path_distances(env.lane_set, env.actions, package)
    delivery_candidates = sorted(
        (
            (pos, dist) for pos, dist in package_dists.items()
            if pos not in {start, package}
        ),
        key=lambda item: (item[1], start_dists.get(item[0], 0), item[0][0], item[0][1]),
    )
    if len(delivery_candidates) < env.num_drones:
        raise ValueError("Map does not have enough reachable cells for all delivery zones.")

    min_delivery_dist = cfg.get("delivery_min_dist", 1)
    max_delivery_dist = cfg.get("delivery_max_dist", 9999)
    delivery_pool = positions_in_distance_range(
        delivery_candidates,
        min_delivery_dist,
        max_delivery_dist,
    )
    if len(delivery_pool) < env.num_drones:
        delivery_pool = [as_position(pos) for pos, _ in delivery_candidates]
    delivery_pool = sorted(
        delivery_pool,
        key=lambda pos: (
            -package_dists.get(pos, 0),
            -start_dists.get(pos, 0),
            pos[0],
            pos[1],
        ),
    )

    start_pool = [start]
    start_pool.extend(
        pos for pos, _ in reachable
        if pos != package and pos not in start_pool
    )
    if len(start_pool) < env.num_drones:
        raise ValueError("Map does not have enough reachable cells for all drone starts.")

    env.fixed_starts = start_pool[: env.num_drones]
    env.fixed_package = package
    env.fixed_deliveries = delivery_pool[: env.num_drones]
    env.random_delivery_candidates = delivery_pool


class MultiDroneDeliveryEnv(gym.Env):
    """Cooperative multi-drone delivery with one package zone and N delivery zones.

    Each drone can independently pick up from the shared package zone. A drone is
    finished once it reaches its assigned delivery zone while carrying a package.
    The episode succeeds when every drone has delivered once.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bw_map,
        num_drones=2,
        max_steps=300,
        fixed_starts=None,
        fixed_package=None,
        fixed_deliveries=None,
        randomize_package=False,
        randomize_deliveries=False,
        step_penalty=-0.2,
        invalid_move_penalty=20.0,
        collision_penalty=20.0,
        revisit_penalty=0.02,
        revisit_penalty_cap=0.5,
        pickup_reward=150.0,
        delivery_reward=500.0,
        team_delivery_bonus=1000.0,
    ):
        super().__init__()
        self.bw_map = np.asarray(bw_map, dtype=int)
        self.grid_shape = self.bw_map.shape
        self.lane_coords = list(map(tuple, np.argwhere(self.bw_map == 1)))
        self.lane_set = set(self.lane_coords)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.num_drones = int(num_drones)
        self.max_steps = int(max_steps)
        self.randomize_package = randomize_package
        self.randomize_deliveries = randomize_deliveries
        self.step_penalty = step_penalty
        self.invalid_move_penalty = invalid_move_penalty
        self.collision_penalty = collision_penalty
        self.revisit_penalty = revisit_penalty
        self.revisit_penalty_cap = revisit_penalty_cap
        self.pickup_reward = pickup_reward
        self.delivery_reward = delivery_reward
        self.team_delivery_bonus = team_delivery_bonus

        self.action_space = spaces.MultiDiscrete([4] * self.num_drones)
        self.agent_observation_space = spaces.Box(low=-1.0, high=1.0, shape=(25,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_drones, self.agent_observation_space.shape[0]),
            dtype=np.float32,
        )

        start = self.lane_coords[0]
        package = self.lane_coords[len(self.lane_coords) // 2]
        deliveries = self._default_delivery_positions(package)
        self.fixed_starts = [as_position(pos) for pos in (fixed_starts or [start] * self.num_drones)]
        self.fixed_package = as_position(fixed_package or package)
        self.fixed_deliveries = [
            as_position(pos) for pos in (fixed_deliveries or deliveries[: self.num_drones])
        ]
        if len(self.fixed_starts) != self.num_drones:
            raise ValueError("fixed_starts must have one start position per drone.")
        if len(self.fixed_deliveries) != self.num_drones:
            raise ValueError("fixed_deliveries must have one delivery zone per drone.")

        self.current_step = 0
        self.positions = None
        self.has_package = None
        self.delivered = None
        self.prev_positions = None
        self.last_actions = None
        self.last_invalid = None
        self.visit_counts = None
        self.random_package_candidates = None
        self.random_delivery_candidates = None

    @property
    def global_state_size(self):
        return int(np.prod(self.observation_space.shape))

    def _default_delivery_positions(self, package):
        dists = shortest_path_distances(self.lane_set, self.actions, as_position(package))
        ranked = sorted(dists, key=lambda pos: (-dists[pos], pos[0], pos[1]))
        return [pos for pos in ranked if pos != package]

    def _sample_position(self, excluded):
        candidates = [pos for pos in self.lane_coords if pos not in excluded]
        idx = int(self.np_random.integers(len(candidates)))
        return as_position(candidates[idx])

    def _sample_from_candidates(self, candidates, excluded):
        candidates = [pos for pos in candidates if pos not in excluded]
        if not candidates:
            candidates = [pos for pos in self.lane_coords if pos not in excluded]
        idx = int(self.np_random.integers(len(candidates)))
        return as_position(candidates[idx])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.positions = [as_position(pos) for pos in self.fixed_starts]
        self.has_package = np.zeros(self.num_drones, dtype=np.float32)
        self.delivered = np.zeros(self.num_drones, dtype=np.float32)
        self.prev_positions = [None] * self.num_drones
        self.last_actions = [None] * self.num_drones
        self.last_invalid = np.zeros(self.num_drones, dtype=np.float32)
        self.visit_counts = [dict() for _ in range(self.num_drones)]

        excluded = set(self.positions)
        if self.randomize_package:
            package_candidates = self.random_package_candidates or self.lane_coords
            self.fixed_package = self._sample_from_candidates(package_candidates, excluded)
        excluded.add(self.fixed_package)

        if self.randomize_deliveries:
            self.fixed_deliveries = []
            for _ in range(self.num_drones):
                delivery_candidates = self.random_delivery_candidates or self.lane_coords
                delivery = self._sample_from_candidates(delivery_candidates, excluded)
                self.fixed_deliveries.append(delivery)
                excluded.add(delivery)

        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        occupied = {pos: idx for idx, pos in enumerate(self.positions)}
        for i, (r, c) in enumerate(self.positions):
            target = self.fixed_deliveries[i] if self.has_package[i] else self.fixed_package
            tr, tc = target
            own_delivery_r, own_delivery_c = self.fixed_deliveries[i]
            surroundings = [
                1.0 if (r + dr, c + dc) in self.lane_set else 0.0
                for dr, dc in self.actions
            ]
            neighbor_flags = [
                1.0 if occupied.get((r + dr, c + dc), i) != i else 0.0
                for dr, dc in self.actions
            ]
            last_action_one_hot = [0.0, 0.0, 0.0, 0.0]
            if self.last_actions[i] is not None:
                last_action_one_hot[int(self.last_actions[i])] = 1.0
            current_visits = self.visit_counts[i].get((r, c), 0)
            obs.append([
                r / self.grid_shape[0],
                c / self.grid_shape[1],
                (tr - r) / self.grid_shape[0],
                (tc - c) / self.grid_shape[1],
                (self.fixed_package[0] - r) / self.grid_shape[0],
                (self.fixed_package[1] - c) / self.grid_shape[1],
                (own_delivery_r - r) / self.grid_shape[0],
                (own_delivery_c - c) / self.grid_shape[1],
                *surroundings,
                *neighbor_flags,
                float(self.has_package[i]),
                float(self.delivered[i]),
                *last_action_one_hot,
                float(self.last_invalid[i]),
                min(current_visits / 5.0, 1.0),
                i / max(self.num_drones - 1, 1),
            ])
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions):
        actions = np.asarray(actions, dtype=int).reshape(self.num_drones)
        self.current_step += 1
        rewards = np.full(self.num_drones, self.step_penalty, dtype=np.float32)
        proposed = []
        invalid = np.zeros(self.num_drones, dtype=bool)

        for i, action in enumerate(actions):
            r, c = self.positions[i]
            self.visit_counts[i][(r, c)] = self.visit_counts[i].get((r, c), 0) + 1
            if self.delivered[i]:
                proposed.append((r, c))
                continue
            dr, dc = self.actions[int(action)]
            nxt = (r + dr, c + dc)
            if nxt not in self.lane_set:
                nxt = (r, c)
                invalid[i] = True
            proposed.append(nxt)

        counts = {}
        for pos in proposed:
            counts[pos] = counts.get(pos, 0) + 1
        original_positions = list(self.positions)
        for i, pos in enumerate(proposed):
            if self.delivered[i]:
                continue
            swapped = any(
                i != j
                and pos == original_positions[j]
                and proposed[j] == original_positions[i]
                for j in range(self.num_drones)
            )
            if counts[pos] > 1 or swapped:
                proposed[i] = original_positions[i]
                rewards[i] -= self.collision_penalty

        for i, pos in enumerate(proposed):
            if self.delivered[i]:
                rewards[i] = 0.0
                continue
            rewards[i] -= self.invalid_move_penalty if invalid[i] else 0.0
            rewards[i] -= min(
                self.revisit_penalty * self.visit_counts[i].get(pos, 0),
                self.revisit_penalty_cap,
            )
            if self.has_package[i] == 0.0 and pos == self.fixed_package:
                self.has_package[i] = 1.0
                rewards[i] += self.pickup_reward
            elif self.has_package[i] == 1.0 and pos == self.fixed_deliveries[i]:
                self.delivered[i] = 1.0
                rewards[i] += self.delivery_reward

        done = bool(np.all(self.delivered))
        if done:
            rewards += self.team_delivery_bonus / self.num_drones
        truncated = self.current_step >= self.max_steps

        self.prev_positions = original_positions
        self.positions = proposed
        self.last_actions = [int(action) for action in actions]
        self.last_invalid = invalid.astype(np.float32)
        info = {
            "team_reward": float(np.sum(rewards)),
            "delivered": self.delivered.copy(),
            "success": done,
        }
        return self._get_obs(), rewards, done, truncated, info


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, state_dim, action_dim, hidden_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def action_dist(self, obs):
        return torch.distributions.Categorical(logits=self.actor(obs))

    def value(self, state):
        return self.critic(state).squeeze(-1)


def build_env(cfg):
    if cfg.get("map_layout") == "t_junction":
        bw_map = build_t_junction_map(
            stem_length=cfg["stem_length"],
            branch_left=cfg.get("branch_left", 2),
            branch_right=cfg.get("branch_right", 2),
            padding=cfg.get("map_padding", 1),
        )
    else:
        bw_map = build_bw_map(
            cfg.get("grid_size", [4, 4]),
            loop_prob=cfg.get("loop_prob", 0.25),
            seed=cfg.get("map_seed"),
        )
    env_kwargs = {key: cfg[key] for key in ENV_DEFAULTS if key in cfg}
    for key in ("fixed_starts", "fixed_package", "fixed_deliveries"):
        if key in cfg:
            env_kwargs[key] = cfg[key]
    env = MultiDroneDeliveryEnv(bw_map, **env_kwargs)
    configure_positions(env, cfg)
    return env


def describe_env(tag, env):
    print(
        f"{tag}: agents={env.num_drones} obs={env.agent_observation_space.shape} "
        f"grid={env.grid_shape} starts={env.fixed_starts} package={env.fixed_package} "
        f"deliveries={env.fixed_deliveries} lanes={len(env.lane_coords)}"
    )


def stage_mode(stage):
    if stage.get("randomize_package") and stage.get("randomize_deliveries"):
        return "random package+deliveries"
    if stage.get("randomize_package"):
        return "random package"
    if stage.get("randomize_deliveries"):
        return "random deliveries"
    return "fixed positions"


def flatten_state(obs):
    return obs.reshape(-1)


def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        next_values = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def collect_rollout(env, model, cfg, device):
    obs, _ = env.reset()
    storage = {key: [] for key in ("obs", "states", "actions", "log_probs", "rewards", "dones", "values")}
    episode_rewards = []
    current_episode_reward = 0.0
    successes = 0

    for _ in range(cfg["rollout_steps"]):
        state = flatten_state(obs)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist = model.action_dist(obs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = model.value(state_tensor).item()

        next_obs, rewards, done, truncated, info = env.step(action.cpu().numpy())
        team_reward = float(np.mean(rewards))
        terminal = done or truncated

        storage["obs"].append(obs.copy())
        storage["states"].append(state.copy())
        storage["actions"].append(action.cpu().numpy())
        storage["log_probs"].append(log_prob.cpu().numpy())
        storage["rewards"].append(team_reward)
        storage["dones"].append(float(terminal))
        storage["values"].append(value)

        current_episode_reward += float(np.sum(rewards))
        obs = next_obs

        if terminal:
            episode_rewards.append(current_episode_reward)
            successes += int(info.get("success", False))
            current_episode_reward = 0.0
            obs, _ = env.reset()

    with torch.no_grad():
        next_value = model.value(
            torch.as_tensor(flatten_state(obs), dtype=torch.float32, device=device).unsqueeze(0)
        ).item()

    rewards = np.asarray(storage["rewards"], dtype=np.float32)
    dones = np.asarray(storage["dones"], dtype=np.float32)
    values = np.asarray(storage["values"], dtype=np.float32)
    advantages, returns = compute_gae(
        rewards,
        dones,
        values,
        next_value,
        cfg["gamma"],
        cfg["gae_lambda"],
    )
    storage["advantages"] = advantages
    storage["returns"] = returns
    return storage, episode_rewards, successes


def update_model(model, optimizer, rollout, cfg, device, num_drones):
    obs = torch.as_tensor(np.asarray(rollout["obs"]), dtype=torch.float32, device=device)
    states = torch.as_tensor(np.asarray(rollout["states"]), dtype=torch.float32, device=device)
    actions = torch.as_tensor(np.asarray(rollout["actions"]), dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(np.asarray(rollout["log_probs"]), dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch_size = obs.shape[0]
    idxs = np.arange(batch_size)
    stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(cfg["epochs"]):
        np.random.shuffle(idxs)
        for start in range(0, batch_size, cfg["minibatch_size"]):
            mb = idxs[start : start + cfg["minibatch_size"]]
            mb_obs = obs[mb].reshape(-1, obs.shape[-1])
            mb_actions = actions[mb].reshape(-1)
            dist = model.action_dist(mb_obs)
            new_log_probs = dist.log_prob(mb_actions).reshape(-1, num_drones)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[mb])
            mb_adv = advantages[mb].unsqueeze(-1)
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - cfg["clip_range"], 1.0 + cfg["clip_range"]) * mb_adv
            actor_loss = -torch.min(unclipped, clipped).mean()

            values = model.value(states[mb])
            critic_loss = 0.5 * (returns[mb] - values).pow(2).mean()
            loss = actor_loss + cfg["vf_coef"] * critic_loss - cfg["ent_coef"] * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()

            stats["actor_loss"] += float(actor_loss.item())
            stats["critic_loss"] += float(critic_loss.item())
            stats["entropy"] += float(entropy.item())
            updates += 1

    return {key: value / max(updates, 1) for key, value in stats.items()}


def evaluate(model, env, n_episodes, device):
    rewards = []
    successes = 0
    for seed in range(n_episodes):
        obs, _ = env.reset(seed=seed)
        done = truncated = False
        total_reward = 0.0
        while not (done or truncated):
            with torch.no_grad():
                dist = model.action_dist(torch.as_tensor(obs, dtype=torch.float32, device=device))
                actions = torch.argmax(dist.logits, dim=-1).cpu().numpy()
            obs, reward, done, truncated, info = env.step(actions)
            total_reward += float(np.sum(reward))
        rewards.append(total_reward)
        successes += int(info.get("success", False))
    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


def rollout_policy(model, src_env, device):
    env = MultiDroneDeliveryEnv(
        src_env.bw_map.copy(),
        num_drones=src_env.num_drones,
        max_steps=src_env.max_steps,
        fixed_starts=src_env.fixed_starts,
        fixed_package=src_env.fixed_package,
        fixed_deliveries=src_env.fixed_deliveries,
        randomize_package=False,
        randomize_deliveries=False,
        step_penalty=src_env.step_penalty,
        invalid_move_penalty=src_env.invalid_move_penalty,
        collision_penalty=src_env.collision_penalty,
        revisit_penalty=src_env.revisit_penalty,
        revisit_penalty_cap=src_env.revisit_penalty_cap,
        pickup_reward=src_env.pickup_reward,
        delivery_reward=src_env.delivery_reward,
        team_delivery_bonus=src_env.team_delivery_bonus,
    )
    obs, _ = env.reset()
    trajectories = [[pos] for pos in env.positions]
    total_reward = 0.0
    frames = [{
        "step": 0,
        "actions": None,
        "rewards": np.zeros(env.num_drones, dtype=np.float32),
        "total_reward": 0.0,
        "positions": list(env.positions),
        "has_package": env.has_package.copy(),
        "delivered": env.delivered.copy(),
        "trajectories": [path.copy() for path in trajectories],
        "done": False,
        "truncated": False,
    }]

    done = truncated = False
    while not (done or truncated):
        with torch.no_grad():
            dist = model.action_dist(torch.as_tensor(obs, dtype=torch.float32, device=device))
            actions = torch.argmax(dist.logits, dim=-1).cpu().numpy()
        obs, rewards, done, truncated, _ = env.step(actions)
        total_reward += float(np.sum(rewards))
        for idx, pos in enumerate(env.positions):
            trajectories[idx].append(pos)
        frames.append({
            "step": len(frames),
            "actions": actions.copy(),
            "rewards": rewards.copy(),
            "total_reward": total_reward,
            "positions": list(env.positions),
            "has_package": env.has_package.copy(),
            "delivered": env.delivered.copy(),
            "trajectories": [path.copy() for path in trajectories],
            "done": bool(done),
            "truncated": bool(truncated),
        })

    return env, frames


def draw_rollout_frame(ax_grid, ax_info, env, frame):
    from matplotlib.colors import ListedColormap

    colors = ["#118ab2", "#ef476f", "#06a77d", "#8338ec", "#ff7f11", "#3a86ff"]
    cmap = ListedColormap(["#1a1a1a", "#f1efe8"])
    ax_grid.clear()
    ax_info.clear()
    ax_grid.imshow(env.bw_map, cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax_grid.set(xticks=[], yticks=[], title=f"MAPPO Multi-Drone Rollout | Step {frame['step']}")

    package_r, package_c = env.fixed_package
    ax_grid.scatter(package_c, package_r, marker="s", s=170, color="#f2c14e", edgecolor="#111", zorder=5)
    ax_grid.text(package_c, package_r, "P", ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)

    for idx, (start_r, start_c) in enumerate(env.fixed_starts):
        color = colors[idx % len(colors)]
        ax_grid.scatter(
            start_c,
            start_r,
            marker="o",
            s=130,
            facecolors="none",
            edgecolors=color,
            linewidths=2.0,
            zorder=4,
        )

    for idx, (delivery_r, delivery_c) in enumerate(env.fixed_deliveries):
        color = colors[idx % len(colors)]
        ax_grid.scatter(delivery_c, delivery_r, marker="*", s=220, color=color, edgecolor="#111", zorder=5)
        ax_grid.text(
            delivery_c,
            delivery_r,
            str(idx),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            zorder=6,
        )

    for idx, trajectory in enumerate(frame["trajectories"]):
        color = colors[idx % len(colors)]
        if len(trajectory) > 1:
            ax_grid.plot(
                [pos[1] for pos in trajectory],
                [pos[0] for pos in trajectory],
                color=color,
                linewidth=2.0,
                alpha=0.75,
                zorder=3,
            )

    for idx, (r, c) in enumerate(frame["positions"]):
        color = "#2ec4b6" if frame["delivered"][idx] else colors[idx % len(colors)]
        marker = "D" if frame["has_package"][idx] else "o"
        ax_grid.scatter(c, r, marker=marker, s=180, color=color, edgecolor="#111", linewidth=1.0, zorder=7)
        ax_grid.text(
            c,
            r,
            str(idx),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
            zorder=8,
        )

    ax_info.set(xlim=(0, 1), ylim=(0, 1))
    ax_info.axis("off")
    ax_info.text(0.5, 0.96, "Rollout Info", ha="center", va="top", fontsize=14, fontweight="bold")
    status = "delivered" if frame["done"] else "truncated" if frame["truncated"] else "running"
    action_names = ["-"] * env.num_drones
    if frame["actions"] is not None:
        action_names = [ACTION_NAMES[int(action)] for action in frame["actions"]]
    lines = [
        ("status", status),
        ("total reward", f"{frame['total_reward']:+.2f}"),
        ("package", str(env.fixed_package)),
    ]
    y = 0.84
    for key, value in lines:
        ax_info.text(0.08, y, key, ha="left", va="top", fontsize=10, color="#666")
        ax_info.text(0.92, y, value, ha="right", va="top", fontsize=10, color="#111")
        y -= 0.08

    for idx in range(env.num_drones):
        pos = frame["positions"][idx]
        carrying = "yes" if frame["has_package"][idx] else "no"
        delivered = "yes" if frame["delivered"][idx] else "no"
        reward = frame["rewards"][idx]
        ax_info.text(
            0.08,
            y,
            f"drone {idx}",
            ha="left",
            va="top",
            fontsize=10,
            color=colors[idx % len(colors)],
            fontweight="bold",
        )
        y -= 0.055
        for key, value in [
            ("action", action_names[idx]),
            ("reward", f"{reward:+.2f}"),
            ("position", str(pos)),
            ("carrying", carrying),
            ("delivered", delivered),
            ("delivery", str(env.fixed_deliveries[idx])),
        ]:
            ax_info.text(0.12, y, key, ha="left", va="top", fontsize=9, color="#666")
            ax_info.text(0.92, y, value, ha="right", va="top", fontsize=9, color="#111")
            y -= 0.045
        y -= 0.03


def visualize_trained_policy(model, src_env, device, delay=0.18, save_path=None, show=False):
    import matplotlib.pyplot as plt

    env, frames = rollout_policy(model, src_env, device)
    fig, (ax_grid, ax_info) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    fig.tight_layout(pad=2.0)

    if save_path is not None:
        from matplotlib.animation import FuncAnimation

        def update(frame_idx):
            draw_rollout_frame(ax_grid, ax_info, env, frames[frame_idx])

        anim = FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=delay * 1000,
            repeat=False,
        )
        writer = "pillow" if save_path.lower().endswith(".gif") else "ffmpeg"
        anim.save(save_path, writer=writer)
        plt.close(fig)

    if show:
        plt.ion()
        plt.show(block=False)
        for frame in frames:
            draw_rollout_frame(ax_grid, ax_info, env, frame)
            fig.canvas.draw_idle()
            plt.pause(delay)
        plt.ioff()
        plt.show()
    elif save_path is None:
        draw_rollout_frame(ax_grid, ax_info, env, frames[-1])
        plt.show()


def save_checkpoint(model, cfg, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        os.path.join(save_dir, f"{name}.pt"),
    )


def main():
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    save_dir = os.path.join("checkpoints", "mappo", "task1", config["run_name"])
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    device = torch.device("cpu")
    first_stage = {**config, **config["curriculum"][0]}
    env = build_env(first_stage)
    eval_env = build_env(first_stage)
    describe_env("Initial MAPPO", env)

    obs_dim = env.agent_observation_space.shape[0]
    state_dim = env.global_state_size
    model = ActorCritic(obs_dim, state_dim, env.action_space.nvec[0], config["hidden_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    global_steps = 0
    best_success = -1.0
    final_stage_env = eval_env
    total_stages = len(config["curriculum"])

    for stage_idx, stage in enumerate(config["curriculum"], start=1):
        stage_config = {**config, **stage}
        env = build_env(stage_config)
        eval_env = build_env(stage_config)
        final_stage_env = eval_env
        if env.num_drones != config["num_drones"]:
            raise ValueError("Changing num_drones across MAPPO stages requires a new actor-critic model.")
        if env.agent_observation_space.shape[0] != obs_dim or env.global_state_size != state_dim:
            raise ValueError("Changing observation size across MAPPO stages requires a new actor-critic model.")

        print(f"\n=== Stage {stage_idx}/{total_stages}: {stage['name']} ({stage_mode(stage)}) ===")
        describe_env("Stage", env)
        stage_steps = 0
        stage_best = -1.0
        promoted = False
        eval_every = stage.get("eval_every", config["stage_eval_every"])
        eval_episodes = stage.get("eval_episodes", config["stage_eval_episodes"])
        promotion_success_rate = stage.get(
            "promotion_success_rate",
            config["stage_promotion_success_rate"],
        )
        min_timesteps_before_promotion = stage.get(
            "min_timesteps_before_promotion",
            config["stage_min_timesteps_before_promotion"],
        )

        while stage_steps < stage["timesteps"]:
            rollout, episode_rewards, rollout_successes = collect_rollout(env, model, stage_config, device)
            stats = update_model(model, optimizer, rollout, stage_config, device, env.num_drones)
            steps_this_rollout = stage_config["rollout_steps"] * env.num_drones
            stage_steps += steps_this_rollout
            global_steps += steps_this_rollout

            if global_steps % config["progress_print_freq"] < steps_this_rollout:
                mean_ep_reward = np.mean(episode_rewards) if episode_rewards else 0.0
                print(
                    f"Step {global_steps:>8} | stage_step={stage_steps:>7} "
                    f"| ep_reward={mean_ep_reward:>8.2f} "
                    f"| rollout_successes={rollout_successes:>3} "
                    f"| actor={stats['actor_loss']:+.4f} critic={stats['critic_loss']:.4f} "
                    f"entropy={stats['entropy']:.4f}"
                )

            should_eval = stage_steps % eval_every < steps_this_rollout
            should_eval = should_eval or stage_steps >= stage["timesteps"]
            if should_eval:
                metrics = evaluate(model, eval_env, eval_episodes, device)
                print(
                    f"Eval @{global_steps:>8} | stage={stage['name']} "
                    f"| success_rate={metrics['success_rate']:.2%} "
                    f"| reward={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}"
                )
                save_checkpoint(model, config, save_dir, f"model_{stage['name']}_{global_steps}_steps")
                if metrics["success_rate"] > stage_best:
                    stage_best = metrics["success_rate"]
                if metrics["success_rate"] > best_success:
                    best_success = metrics["success_rate"]
                    save_checkpoint(model, config, save_dir, "model_best")
                if (
                    stage_steps >= min_timesteps_before_promotion
                    and metrics["success_rate"] >= promotion_success_rate
                ):
                    print(
                        f"Early promotion from {stage['name']}: "
                        f"{metrics['success_rate']:.2%} >= {promotion_success_rate:.2%}"
                    )
                    promoted = True
                    break

        if not promoted:
            print(f"Full budget used for {stage['name']}. Best stage success={stage_best:.2%}.")

    final_metrics = evaluate(model, final_stage_env, config["stage_eval_episodes"], device)
    print(
        f"Final eval | success_rate={final_metrics['success_rate']:.2%} "
        f"| mean_reward={final_metrics['mean_reward']:.2f} +/- {final_metrics['std_reward']:.2f}"
    )
    save_checkpoint(model, config, save_dir, "model_final")

    if config["viz_enabled"]:
        viz_save_path = config["viz_save_path"]
        if viz_save_path is not None and not os.path.isabs(viz_save_path):
            viz_save_path = os.path.join(save_dir, viz_save_path)
        visualize_trained_policy(
            model,
            final_stage_env,
            device,
            delay=config["viz_delay"],
            save_path=viz_save_path,
            show=config["viz_show"],
        )
        if viz_save_path is not None:
            print(f"Saved MAPPO rollout visualization to {viz_save_path}")


if __name__ == "__main__":
    main()
