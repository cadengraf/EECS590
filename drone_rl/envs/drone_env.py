import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneEnv(gym.Env):
    def __init__(self, bw_map, max_steps=2000, print_freq=0):
        super().__init__()
        self.bw_map = bw_map
        self.grid_shape = bw_map.shape
        self.lane_coords = list(map(tuple, np.argwhere(bw_map == 1)))
        self.lane_set = set(self.lane_coords)

        # Directions: Up, Down, Left, Right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation: [r, c, rel_r, rel_c, wall_u, wall_d, wall_l, wall_r, N, S, W, E, has_pkg]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

        self.fixed_start = self.lane_coords[0]
        self.fixed_package = self.lane_coords[len(self.lane_coords) // 4]
        self.fixed_delivery = self.lane_coords[len(self.lane_coords) // 2]

        self.max_steps = max_steps
        self.print_freq = print_freq
        self.current_step = 0
        self.total_steps = 0
        self.pickups_since_report = 0
        self.deliveries_since_report = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.state = None  # (r, c, has_pkg)
        self.prev_pos = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_pos = None

        # Always start at the same spot
        r, c = self.fixed_start
        self.state = (r, c, 0)  # 0 = no package

        return self._get_obs(), {}

    def _get_obs(self):
        r, c, has_pkg = self.state
        # Dynamic target based on current objective
        tr, tc = self.fixed_delivery if has_pkg else self.fixed_package

        # 1. Normalized Relative Vector
        rel_r = (tr - r) / self.grid_shape[0]
        rel_c = (tc - c) / self.grid_shape[1]

        # 2. Local Wall Radar (is adjacent cell a lane?)
        surroundings = []
        for dr, dc in self.actions:
            nr, nc = r + dr, c + dc
            surroundings.append(1.0 if (nr, nc) in self.lane_set else 0.0)

        # 3. Binary Quadrant Flags
        is_north = float(tr < r)
        is_south = float(tr > r)
        is_west = float(tc < c)
        is_east = float(tc > c)

        return np.array([
            r / self.grid_shape[0],
            c / self.grid_shape[1],
            rel_r, rel_c,
            *surroundings,
            is_north, is_south, is_west, is_east,
            float(has_pkg)
        ], dtype=np.float32)

    def _maybe_print_report(self):
        if self.print_freq and self.total_steps % self.print_freq == 0:
            print(
                f"[Env] Steps {self.total_steps - self.print_freq + 1}-{self.total_steps} | "
                f"pickups={self.pickups_since_report} | "
                f"deliveries={self.deliveries_since_report} | "
                f"total_pickups={self.total_pickups} | "
                f"total_deliveries={self.total_deliveries}"
            )
            self.pickups_since_report = 0
            self.deliveries_since_report = 0

    def step(self, action):
        self.current_step += 1
        self.total_steps += 1
        r, c, has_pkg = self.state
        dr, dc = self.actions[int(action)]
        nr, nc = r + dr, c + dc

        reward = -0.1  # Constant step penalty to encourage speed
        done = False
        truncated = self.current_step >= self.max_steps

        # Wall Collision Logic
        if (nr, nc) not in self.lane_set:
            # Stay put and take a heavy penalty
            nr, nc = r, c
            reward -= 2.0

        # Anti-Oscillation (Penalty for moving back to where you just were)
        if self.prev_pos == (nr, nc):
            reward -= 0.5

        self.prev_pos = (r, c)

        # Objective Logic
        if not has_pkg and (nr, nc) == self.fixed_package:
            has_pkg = 1
            reward += 100.0  # Huge bonus for pickup
            self.pickups_since_report += 1
            self.total_pickups += 1

        elif has_pkg and (nr, nc) == self.fixed_delivery:
            reward += 200.0  # Jackpot for delivery
            done = True
            self.deliveries_since_report += 1
            self.total_deliveries += 1

        self.state = (nr, nc, has_pkg)
        self._maybe_print_report()
        return self._get_obs(), reward, done, truncated, {}
