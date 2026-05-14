import random
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from utils.pipes import PipeGrid, PipeOptions, PipeVisualizerBW
except ImportError:
    from quad_physics.utils.pipes import PipeGrid, PipeOptions, PipeVisualizerBW


ACTION_DIM = 2
ENV_DEFAULTS = {
    "num_drones": 2,
    "max_steps": 300,
    "print_freq": 20_000,
    "randomize_package": False,
    "randomize_deliveries": False,
    "step_penalty": -0.2,
    "invalid_move_penalty": 20.0,
    "collision_penalty": 20.0,
    "revisit_penalty": 0.05,
    "revisit_penalty_cap": 1.5,
    "pickup_reward": 150.0,
    "delivery_reward": 500.0,
    "team_delivery_bonus": 1000.0,
    "undelivered_package_penalty": 300.0,
    "undelivered_agent_penalty": 0.0,
    "wrong_delivery_penalty": 50.0,
    "dt": 0.05,
    "physics_substeps": 8,
    "max_tilt": 0.50,
    "linear_drag": 0.55,
    "attitude_tau": 0.25,
    "angular_damping": 0.90,
    "target_altitude": 1.0,
}


def as_position(pos):
    return tuple(map(int, pos))


def shortest_path_distances(lane_set, actions, start):
    queue = deque([(start, 0)])
    distances = {start: 0}
    while queue:
        (row, col), dist = queue.popleft()
        for dr, dc in actions:
            nxt = (row + dr, col + dc)
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


class MultiQuadDeliveryEnv(gym.Env):
    """Cooperative multi-drone delivery with simplified quadcopter physics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bw_map,
        num_drones=2,
        max_steps=300,
        print_freq=20_000,
        fixed_starts=None,
        fixed_package=None,
        fixed_deliveries=None,
        randomize_package=False,
        randomize_deliveries=False,
        step_penalty=-0.2,
        invalid_move_penalty=20.0,
        collision_penalty=20.0,
        revisit_penalty=0.05,
        revisit_penalty_cap=1.5,
        pickup_reward=150.0,
        delivery_reward=500.0,
        team_delivery_bonus=1000.0,
        undelivered_package_penalty=300.0,
        undelivered_agent_penalty=0.0,
        wrong_delivery_penalty=50.0,
        dt=0.05,
        physics_substeps=8,
        mass=1.0,
        gravity=9.81,
        max_tilt=0.50,
        attitude_tau=0.18,
        angular_damping=0.82,
        linear_drag=0.55,
        altitude_kp=14.0,
        altitude_kd=5.5,
        target_altitude=1.0,
        crash_altitude=0.2,
        crash_tilt=0.85,
    ):
        super().__init__()
        self.bw_map = np.asarray(bw_map, dtype=int)
        self.grid_shape = self.bw_map.shape
        self.lane_coords = list(map(tuple, np.argwhere(self.bw_map == 1)))
        self.lane_set = set(self.lane_coords)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.num_drones = int(num_drones)
        self.max_steps = int(max_steps)
        self.print_freq = int(print_freq)
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
        self.undelivered_package_penalty = undelivered_package_penalty
        self.undelivered_agent_penalty = undelivered_agent_penalty
        self.wrong_delivery_penalty = wrong_delivery_penalty
        self.dt = dt
        self.physics_substeps = physics_substeps
        self.mass = mass
        self.gravity = gravity
        self.max_tilt = max_tilt
        self.attitude_tau = attitude_tau
        self.angular_damping = angular_damping
        self.linear_drag = linear_drag
        self.altitude_kp = altitude_kp
        self.altitude_kd = altitude_kd
        self.target_altitude = target_altitude
        self.crash_altitude = crash_altitude
        self.crash_tilt = crash_tilt

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_drones, ACTION_DIM), dtype=np.float32)
        self.agent_observation_space = spaces.Box(low=-1.0, high=1.0, shape=(36,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_drones, self.agent_observation_space.shape[0]),
            dtype=np.float32,
        )

        start = as_position(self.lane_coords[0])
        package = as_position(self.lane_coords[len(self.lane_coords) // 2])
        deliveries = self._default_delivery_positions(package)
        self.fixed_starts = [as_position(pos) for pos in (fixed_starts or [start] * self.num_drones)]
        self.fixed_package = as_position(fixed_package or package)
        self.fixed_deliveries = [as_position(pos) for pos in (fixed_deliveries or deliveries[: self.num_drones])]
        self.random_package_candidates = None
        self.random_delivery_candidates = None
        self.package_min_dist = 1
        self.package_max_dist = 9999
        self.delivery_min_dist = 1
        self.delivery_max_dist = 9999

        self.current_step = 0
        self.total_steps = 0
        self.pickups_since_report = 0
        self.deliveries_since_report = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.episode_pickups = 0
        self.episode_deliveries = 0
        self.episode_wrong_deliveries = 0
        self.positions = None
        self.shared_start_cells = None
        self.has_package = None
        self.delivered = None
        self.prev_positions = None
        self.last_actions = None
        self.last_invalid = None
        self.wrong_delivery_visits = None
        self.visit_counts = None
        self.pos = None
        self.vel = None
        self.angles = None
        self.rates = None

    @property
    def global_state_size(self):
        return int(np.prod(self.observation_space.shape))

    def _default_delivery_positions(self, package):
        dists = shortest_path_distances(self.lane_set, self.actions, as_position(package))
        ranked = sorted(dists, key=lambda pos: (-dists[pos], pos[0], pos[1]))
        return [pos for pos in ranked if pos != package]

    def _sample_from_candidates(self, candidates, excluded):
        candidates = [as_position(pos) for pos in candidates if as_position(pos) not in excluded]
        if not candidates:
            candidates = [as_position(pos) for pos in self.lane_coords if as_position(pos) not in excluded]
        return candidates[int(self.np_random.integers(len(candidates)))]

    def _delivery_pool_for_package(self, package, excluded):
        package_dists = shortest_path_distances(self.lane_set, self.actions, package)
        start_dists = shortest_path_distances(self.lane_set, self.actions, self.fixed_starts[0])
        candidates = sorted(
            ((pos, dist) for pos, dist in package_dists.items() if pos not in excluded | {package}),
            key=lambda item: (item[1], start_dists.get(item[0], 0), item[0]),
        )
        pool = positions_in_distance_range(candidates, self.delivery_min_dist, self.delivery_max_dist)
        if len(pool) < self.num_drones:
            pool = [as_position(pos) for pos, _ in candidates]
        return sorted(pool, key=lambda pos: (-package_dists.get(pos, 0), -start_dists.get(pos, 0), pos))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_pickups = 0
        self.episode_deliveries = 0
        self.episode_wrong_deliveries = 0
        self.positions = [as_position(pos) for pos in self.fixed_starts]
        start_counts = {}
        for pos in self.positions:
            start_counts[pos] = start_counts.get(pos, 0) + 1
        self.shared_start_cells = {pos for pos, count in start_counts.items() if count > 1}
        self.has_package = np.zeros(self.num_drones, dtype=np.float32)
        self.delivered = np.zeros(self.num_drones, dtype=np.float32)
        self.prev_positions = [None] * self.num_drones
        self.last_actions = np.zeros((self.num_drones, ACTION_DIM), dtype=np.float32)
        self.last_invalid = np.zeros(self.num_drones, dtype=np.float32)
        self.wrong_delivery_visits = [set() for _ in range(self.num_drones)]
        self.visit_counts = [dict() for _ in range(self.num_drones)]
        self.pos = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.vel = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.angles = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.rates = np.zeros((self.num_drones, 3), dtype=np.float32)

        excluded = set(self.positions)
        if self.randomize_package:
            self.fixed_package = self._sample_from_candidates(self.random_package_candidates or self.lane_coords, excluded)
        excluded.add(self.fixed_package)
        if self.randomize_deliveries:
            self.fixed_deliveries = []
            self.random_delivery_candidates = self._delivery_pool_for_package(self.fixed_package, excluded)
            for _ in range(self.num_drones):
                delivery = self._sample_from_candidates(self.random_delivery_candidates or self.lane_coords, excluded)
                self.fixed_deliveries.append(delivery)
                excluded.add(delivery)

        for idx, start in enumerate(self.positions):
            self.pos[idx] = (float(start[0]), float(start[1]), self.target_altitude)
        return self._get_obs(), {}

    def _current_cell(self, idx):
        return int(np.rint(self.pos[idx, 0])), int(np.rint(self.pos[idx, 1]))

    def _integrate_physics(self, idx, action):
        roll_cmd, pitch_cmd = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target_angles = np.array([roll_cmd * self.max_tilt, pitch_cmd * self.max_tilt, 0.0], dtype=np.float32)
        for _ in range(self.physics_substeps):
            angle_error = target_angles - self.angles[idx]
            desired_rates = angle_error / max(self.attitude_tau, 1e-6)
            self.rates[idx] += (desired_rates - self.rates[idx]) * (1.0 - self.angular_damping)
            self.angles[idx] += self.rates[idx] * self.dt
            self.angles[idx, 0:2] = np.clip(self.angles[idx, 0:2], -self.max_tilt, self.max_tilt)
            roll, pitch, _ = self.angles[idx]
            horizontal_accel = np.array([self.gravity * np.sin(pitch), -self.gravity * np.sin(roll)], dtype=np.float32)
            altitude_error = self.target_altitude - self.pos[idx, 2]
            vertical_accel = self.altitude_kp * altitude_error - self.altitude_kd * self.vel[idx, 2]
            accel = np.array([horizontal_accel[0], horizontal_accel[1], vertical_accel], dtype=np.float32)
            accel -= self.linear_drag * self.vel[idx]
            self.vel[idx] += accel * self.dt
            self.pos[idx] += self.vel[idx] * self.dt

    def _get_obs(self):
        obs = []
        occupied = {pos: idx for idx, pos in enumerate(self.positions) if not self.delivered[idx]}
        for idx, (row, col) in enumerate(self.positions):
            target = self.fixed_deliveries[idx] if self.has_package[idx] else self.fixed_package
            tr, tc = target
            own_dr, own_dc = self.fixed_deliveries[idx]
            surroundings = [1.0 if (row + dr, col + dc) in self.lane_set else 0.0 for dr, dc in self.actions]
            neighbor_flags = [1.0 if occupied.get((row + dr, col + dc), idx) != idx else 0.0 for dr, dc in self.actions]
            current_visits = self.visit_counts[idx].get((row, col), 0)
            obs.append(
                [
                    row / self.grid_shape[0],
                    col / self.grid_shape[1],
                    (tr - row) / self.grid_shape[0],
                    (tc - col) / self.grid_shape[1],
                    (self.fixed_package[0] - row) / self.grid_shape[0],
                    (self.fixed_package[1] - col) / self.grid_shape[1],
                    (own_dr - row) / self.grid_shape[0],
                    (own_dc - col) / self.grid_shape[1],
                    self.pos[idx, 0] - row,
                    self.pos[idx, 1] - col,
                    self.pos[idx, 2] - self.target_altitude,
                    *np.clip(self.vel[idx] / 6.0, -1.0, 1.0),
                    *np.clip(self.angles[idx] / self.max_tilt, -1.0, 1.0),
                    *np.clip(self.rates[idx] / 8.0, -1.0, 1.0),
                    *surroundings,
                    *neighbor_flags,
                    float(self.has_package[idx]),
                    float(self.delivered[idx]),
                    *self.last_actions[idx].tolist(),
                    float(self.last_invalid[idx]),
                    min(current_visits / 5.0, 1.0),
                    1.0 if idx == 0 else 0.0,
                    idx / max(self.num_drones - 1, 1),
                ]
            )
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions):
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0).reshape(self.num_drones, ACTION_DIM)
        self.current_step += 1
        self.total_steps += 1
        rewards = np.full(self.num_drones, self.step_penalty, dtype=np.float32)
        original_positions = list(self.positions)
        proposed = []
        invalid = np.zeros(self.num_drones, dtype=bool)

        for idx, action in enumerate(actions):
            row, col = self.positions[idx]
            self.visit_counts[idx][(row, col)] = self.visit_counts[idx].get((row, col), 0) + 1
            if self.delivered[idx]:
                proposed.append((row, col))
                continue
            self._integrate_physics(idx, action)
            nr, nc = self._current_cell(idx)
            crashed = (
                self.pos[idx, 2] < self.crash_altitude
                or abs(float(self.angles[idx, 0])) > self.crash_tilt
                or abs(float(self.angles[idx, 1])) > self.crash_tilt
            )
            if (nr, nc) not in self.lane_set or crashed:
                invalid[idx] = True
                nr, nc = row, col
                self.pos[idx, 0:2] = (float(row), float(col))
                self.vel[idx, 0:2] = 0.0
            proposed.append((nr, nc))

        counts = {}
        for idx, pos in enumerate(proposed):
            if not self.delivered[idx]:
                counts[pos] = counts.get(pos, 0) + 1
        for idx, pos in enumerate(proposed):
            if self.delivered[idx]:
                continue
            allowed_shared_start = (
                pos in self.shared_start_cells
                and pos == original_positions[idx]
                and all(
                    proposed[other] == pos
                    for other in range(self.num_drones)
                    if original_positions[other] == pos and not self.delivered[other]
                )
            )
            swapped = any(
                idx != other
                and not self.delivered[other]
                and original_positions[idx] != original_positions[other]
                and pos == original_positions[other]
                and proposed[other] == original_positions[idx]
                for other in range(self.num_drones)
            )
            if (counts[pos] > 1 and not allowed_shared_start) or swapped:
                proposed[idx] = original_positions[idx]
                self.pos[idx, 0:2] = (float(original_positions[idx][0]), float(original_positions[idx][1]))
                self.vel[idx, 0:2] = 0.0
                rewards[idx] -= self.collision_penalty

        for idx, pos in enumerate(proposed):
            if self.delivered[idx]:
                rewards[idx] = 0.0
                continue
            rewards[idx] -= self.invalid_move_penalty if invalid[idx] else 0.0
            rewards[idx] -= min(self.revisit_penalty * self.visit_counts[idx].get(pos, 0), self.revisit_penalty_cap)
            if self.has_package[idx] == 0.0 and pos == self.fixed_package:
                self.has_package[idx] = 1.0
                rewards[idx] += self.pickup_reward
                self.pickups_since_report += 1
                self.total_pickups += 1
                self.episode_pickups += 1
            elif self.has_package[idx] == 1.0 and pos == self.fixed_deliveries[idx]:
                self.delivered[idx] = 1.0
                rewards[idx] += self.delivery_reward
                self.deliveries_since_report += 1
                self.total_deliveries += 1
                self.episode_deliveries += 1
            elif self.has_package[idx] == 1.0 and pos in self.fixed_deliveries:
                if pos not in self.wrong_delivery_visits[idx]:
                    rewards[idx] -= self.wrong_delivery_penalty
                    self.wrong_delivery_visits[idx].add(pos)
                    self.episode_wrong_deliveries += 1

        done = bool(np.all(self.delivered))
        failed = bool(np.any(invalid))
        truncated = self.current_step >= self.max_steps
        if done:
            for idx in range(self.num_drones):
                if self.delivered[idx]:  # only reward drones that actually delivered
                    rewards[idx] += self.team_delivery_bonus / self.num_drones
        else:
            for idx, pos in enumerate(proposed):
                if self.has_package[idx] == 0.0 and pos == self.fixed_deliveries[idx]:
                    rewards[idx] -= 2.0  # small nudge away from camping delivery zone
        
        if truncated and not done and not failed:
            for idx in range(self.num_drones):
                if self.has_package[idx] and not self.delivered[idx]:
                    rewards[idx] -= self.undelivered_package_penalty
                elif not self.delivered[idx]:
                    rewards[idx] -= self.undelivered_agent_penalty

        self.prev_positions = original_positions
        self.positions = proposed
        self.last_actions = actions.copy()
        self.last_invalid = invalid.astype(np.float32)
        if self.print_freq and self.total_steps % self.print_freq == 0:
            print(
                f"[MultiQuadEnv] Steps {self.total_steps - self.print_freq + 1}-{self.total_steps} | "
                f"pickups={self.pickups_since_report} | deliveries={self.deliveries_since_report} | "
                f"total_pickups={self.total_pickups} | total_deliveries={self.total_deliveries}"
            )
            self.pickups_since_report = 0
            self.deliveries_since_report = 0
        return self._get_obs(), rewards, done, truncated or failed, {
            "team_reward": float(np.sum(rewards)),
            "delivered": self.delivered.copy(),
            "episode_pickups": self.episode_pickups,
            "episode_deliveries": self.episode_deliveries,
            "episode_wrong_deliveries": self.episode_wrong_deliveries,
            "success": done,
            "failed": failed,
        }


def configure_positions(env, cfg):
    env.package_min_dist = cfg.get("package_min_dist", 1)
    env.package_max_dist = cfg.get("package_max_dist", 9999)
    env.delivery_min_dist = cfg.get("delivery_min_dist", 1)
    env.delivery_max_dist = cfg.get("delivery_max_dist", 9999)

    start = as_position(cfg["fixed_starts"][0]) if "fixed_starts" in cfg else as_position(
        min(env.lane_coords, key=lambda pos: (pos[0] + pos[1], pos[0], pos[1]))
    )
    start_dists = shortest_path_distances(env.lane_set, env.actions, start)
    reachable = sorted(((pos, dist) for pos, dist in start_dists.items() if pos != start), key=lambda item: (item[1], item[0]))
    starts = [as_position(pos) for pos in cfg["fixed_starts"]] if "fixed_starts" in cfg else [start]
    if "fixed_starts" not in cfg:
        starts.extend(as_position(pos) for pos, _ in reachable[: max(env.num_drones - 1, 0)])
    starts = starts[: env.num_drones]
    excluded_starts = set(starts)
    package_candidates = [(pos, dist) for pos, dist in reachable if as_position(pos) not in excluded_starts]
    env.random_package_candidates = positions_in_distance_range(
        package_candidates,
        env.package_min_dist,
        env.package_max_dist,
    )
    if "fixed_package" in cfg and not cfg.get("randomize_package", False):
        package = as_position(cfg["fixed_package"])
    else:
        package = pick_position_by_distance(package_candidates, env.package_min_dist, env.package_max_dist)
    if "fixed_deliveries" in cfg and not cfg.get("randomize_deliveries", False):
        delivery_pool = [as_position(pos) for pos in cfg["fixed_deliveries"]]
    else:
        delivery_pool = env._delivery_pool_for_package(package, excluded_starts)
    env.fixed_starts = starts
    env.fixed_package = package
    env.fixed_deliveries = delivery_pool[: env.num_drones]
    env.random_delivery_candidates = delivery_pool


def build_env(cfg):
    if cfg.get("map_layout") == "straight_line":
        bw_map = build_straight_line_map(cfg["line_length"], padding=cfg.get("map_padding", 1))
    elif cfg.get("map_layout") == "t_junction":
        bw_map = build_t_junction_map(
            stem_length=cfg["stem_length"],
            branch_left=cfg.get("branch_left", 2),
            branch_right=cfg.get("branch_right", 2),
            padding=cfg.get("map_padding", 1),
        )
    else:
        bw_map = build_bw_map(cfg.get("grid_size", [4, 4]), loop_prob=cfg.get("loop_prob", 0.25), seed=cfg.get("map_seed"))
    env_kwargs = {key: cfg[key] for key in ENV_DEFAULTS if key in cfg}
    for key in ("fixed_starts", "fixed_package", "fixed_deliveries"):
        if key in cfg:
            env_kwargs[key] = cfg[key]
    env = MultiQuadDeliveryEnv(bw_map, **env_kwargs)
    configure_positions(env, cfg)
    return env


def clone_env(src_env):
    attrs = {key: getattr(src_env, key) for key in ENV_DEFAULTS}
    cloned = MultiQuadDeliveryEnv(
        src_env.bw_map.copy(),
        fixed_starts=src_env.fixed_starts,
        fixed_package=src_env.fixed_package,
        fixed_deliveries=src_env.fixed_deliveries,
        **attrs,
    )
    cloned.random_package_candidates = src_env.random_package_candidates
    cloned.random_delivery_candidates = src_env.random_delivery_candidates
    cloned.package_min_dist = src_env.package_min_dist
    cloned.package_max_dist = src_env.package_max_dist
    cloned.delivery_min_dist = src_env.delivery_min_dist
    cloned.delivery_max_dist = src_env.delivery_max_dist
    return cloned


def describe_env(tag, env):
    print(
        f"{tag}: agents={env.num_drones} obs={env.agent_observation_space.shape} "
        f"grid={env.grid_shape} starts={env.fixed_starts} package={env.fixed_package} "
        f"deliveries={env.fixed_deliveries} lanes={len(env.lane_coords)} "
        f"rand_pkg={env.randomize_package} rand_del={env.randomize_deliveries} "
        f"dt={env.dt} substeps={env.physics_substeps} max_tilt={env.max_tilt}"
    )


def stage_mode(stage):
    if stage.get("randomize_package") and stage.get("randomize_deliveries"):
        return "random package+deliveries"
    if stage.get("randomize_package"):
        return "random package"
    if stage.get("randomize_deliveries"):
        return "random deliveries"
    return "fixed positions"
