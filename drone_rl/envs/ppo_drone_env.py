import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneEnv(gym.Env):
    def __init__(
        self,
        bw_map,
        max_steps=2000,
        print_freq=0,
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
            shape=(19,),
            dtype=np.float32,
        )

        self.max_steps = max_steps
        self.print_freq = print_freq
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
        self.random_package_candidates = None
        self.random_delivery_candidates = None
        self.last_action = None
        self.last_move_invalid = 0.0

        n = len(self.lane_coords)
        self.fixed_start = tuple(map(int, self.lane_coords[0]))
        self.fixed_package = tuple(map(int, self.lane_coords[n // 4]))
        self.fixed_delivery = tuple(map(int, self.lane_coords[n // 2]))

    def _sample_position(self, candidates, excluded):
        candidates = [pos for pos in candidates if pos not in excluded]
        if not candidates:
            candidates = [pos for pos in self.lane_coords if pos not in excluded]
        idx = int(self.np_random.integers(len(candidates)))
        return tuple(map(int, candidates[idx]))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_state = None
        self.visit_counts = {}
        self.last_action = None
        self.last_move_invalid = 0.0

        if self.randomize_package:
            package_candidates = self.random_package_candidates or self.lane_coords
            self.fixed_package = self._sample_position(
                package_candidates,
                {self.fixed_start, self.fixed_delivery},
            )

        if self.randomize_delivery:
            delivery_candidates = self.random_delivery_candidates or self.lane_coords
            self.fixed_delivery = self._sample_position(
                delivery_candidates,
                {self.fixed_start, self.fixed_package},
            )

        self.state = (*self.fixed_start, 0)
        return self._get_obs(), {}

    def _get_obs(self):
        r, c, has_pkg = self.state
        tr, tc = self.fixed_delivery if has_pkg else self.fixed_package
        surroundings = [
            1.0 if (r + dr, c + dc) in self.lane_set else 0.0
            for dr, dc in self.actions
        ]
        last_action_one_hot = [0.0, 0.0, 0.0, 0.0]
        if self.last_action is not None:
            last_action_one_hot[int(self.last_action)] = 1.0
        current_visits = self.visit_counts.get((r, c), 0)
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
            *last_action_one_hot,
            self.last_move_invalid,
            min(current_visits / 5.0, 1.0),
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
        self.last_action = int(action)
        self.last_move_invalid = float(invalid_move)

        if self.print_freq and self.total_steps % self.print_freq == 0:
            print(
                f"[Env] Steps {self.total_steps - self.print_freq + 1}-{self.total_steps} | "
                f"pickups={self.pickups_since_report} | "
                f"deliveries={self.deliveries_since_report} | "
                f"total_pickups={self.total_pickups} | total_deliveries={self.total_deliveries}"
            )
            self.pickups_since_report = 0
            self.deliveries_since_report = 0

        return self._get_obs(), reward, done, truncated, {}
