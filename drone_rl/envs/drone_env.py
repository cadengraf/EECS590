import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneEnv(gym.Env):
    def __init__(
        self,
        bw_map,
        max_steps=2000,
        print_freq=0,
        progress_reward_scale=0.1,
        randomize_package=False,
        randomize_delivery=False,
        step_penalty=-0.2,
        invalid_move_penalty=2.0,
        revisit_penalty=0.05,
        revisit_penalty_cap=1.5,
        backtrack_penalty=1.0,
        pickup_reward=300.0,
        delivery_reward=1000.0,
    ):
        super().__init__()
        self.bw_map = bw_map
        self.grid_shape = bw_map.shape
        self.lane_coords = list(map(tuple, np.argwhere(bw_map == 1)))
        self.lane_set = set(self.lane_coords)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32,
        )

        self.max_steps = max_steps
        self.print_freq = print_freq
        self.progress_reward_scale = progress_reward_scale
        self.randomize_package = randomize_package
        self.randomize_delivery = randomize_delivery
        self.step_penalty = step_penalty
        self.invalid_move_penalty = invalid_move_penalty
        self.revisit_penalty = revisit_penalty
        self.revisit_penalty_cap = revisit_penalty_cap
        self.backtrack_penalty = backtrack_penalty
        self.pickup_reward = pickup_reward
        self.delivery_reward = delivery_reward
        self.current_step = 0
        self.total_steps = 0
        self.pickups_since_report = 0
        self.deliveries_since_report = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.state = None
        self.prev_state = None
        self.visit_counts = {}

        n = len(self.lane_coords)
        self.fixed_start = self.lane_coords[0]
        self.fixed_package = self.lane_coords[n // 4]
        self.fixed_delivery = self.lane_coords[n // 2]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_state = None
        self.visit_counts = {}

        if self.randomize_package:
            package_candidates = [
                pos for pos in self.lane_coords
                if pos != self.fixed_start and pos != self.fixed_delivery
            ]
            pkg_idx = int(self.np_random.integers(len(package_candidates)))
            self.fixed_package = package_candidates[pkg_idx]

        if self.randomize_delivery:
            delivery_candidates = [
                pos for pos in self.lane_coords
                if pos != self.fixed_start and pos != self.fixed_package
            ]
            del_idx = int(self.np_random.integers(len(delivery_candidates)))
            self.fixed_delivery = delivery_candidates[del_idx]

        self.state = (*self.fixed_start, 0)
        return self._get_obs(), {}

    def _get_obs(self):
        r, c, has_pkg = self.state
        tr, tc = self.fixed_delivery if has_pkg else self.fixed_package
        surroundings = [
            1.0 if (r + dr, c + dc) in self.lane_set else 0.0
            for dr, dc in self.actions
        ]
        return np.array([
            r / self.grid_shape[0],
            c / self.grid_shape[1],
            (tr - r) / self.grid_shape[0],
            (tc - c) / self.grid_shape[1],
            *surroundings,
            float(tr < r),
            float(tr > r),
            float(tc < c),
            float(tc > c),
            float(has_pkg),
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        self.total_steps += 1
        r, c, has_pkg = self.state
        dr, dc = self.actions[int(action)]
        nr, nc = r + dr, c + dc

        self.visit_counts[(r, c)] = self.visit_counts.get((r, c), 0) + 1

        reward = self.step_penalty
        done = False
        truncated = self.current_step >= self.max_steps
        invalid_move = False

        if (nr, nc) not in self.lane_set:
            nr, nc = r, c
            invalid_move = True

        if invalid_move:
            reward -= self.invalid_move_penalty
            done = True

        visits = self.visit_counts.get((nr, nc), 0)
        reward -= min(self.revisit_penalty * visits, self.revisit_penalty_cap)

        if self.prev_state is not None and (nr, nc) == self.prev_state[:2]:
            reward -= self.backtrack_penalty

        if not has_pkg and (nr, nc) == self.fixed_package:
            has_pkg = 1
            reward += self.pickup_reward
            self.pickups_since_report += 1
            self.total_pickups += 1
        elif has_pkg and (nr, nc) == self.fixed_delivery:
            reward += self.delivery_reward
            done = True
            self.deliveries_since_report += 1
            self.total_deliveries += 1

        self.state = (nr, nc, has_pkg)
        self.prev_state = (r, c, has_pkg)

        if self.print_freq and self.total_steps % self.print_freq == 0:
            print(
                f"[Env] Steps {self.total_steps - self.print_freq + 1}-{self.total_steps} | "
                f"pickups={self.pickups_since_report} | deliveries={self.deliveries_since_report} | "
                f"total_pickups={self.total_pickups} | total_deliveries={self.total_deliveries}"
            )
            self.pickups_since_report = 0
            self.deliveries_since_report = 0

        return self._get_obs(), reward, done, truncated, {}
