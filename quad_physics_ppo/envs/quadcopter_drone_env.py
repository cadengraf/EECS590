import gymnasium as gym
from gymnasium import spaces
import numpy as np


class QuadcopterDroneEnv(gym.Env):
    """Grid delivery task with simplified quadcopter rigid-body dynamics."""

    def __init__(
        self,
        bw_map,
        max_steps=2000,
        print_freq=0,
        randomize_package=False,
        randomize_delivery=False,
        step_penalty=-0.2,
        invalid_move_penalty=200.0,
        revisit_penalty=0.05,
        revisit_penalty_cap=1.5,
        backtrack_penalty=1.0,
        pickup_reward=300.0,
        delivery_reward=1000.0,
        undelivered_package_penalty=300.0,
        dt=0.05,
        physics_substeps=8,
        mass=1.0,
        gravity=9.81,
        max_tilt=0.55,
        attitude_tau=0.18,
        angular_damping=0.82,
        linear_drag=0.22,
        altitude_kp=14.0,
        altitude_kd=5.5,
        target_altitude=1.0,
        crash_altitude=0.2,
        crash_tilt=0.85,
    ):
        super().__init__()
        self.bw_map = bw_map
        self.grid_shape = bw_map.shape
        self.lane_coords = list(map(tuple, np.argwhere(bw_map == 1)))
        self.lane_set = set(self.lane_coords)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(31,),
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
        self.undelivered_package_penalty = undelivered_package_penalty

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
        self.last_action_cmd = np.zeros(2, dtype=np.float32)
        self.last_move_invalid = 0.0
        self.last_cell = None
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.angles = np.zeros(3, dtype=np.float32)
        self.rates = np.zeros(3, dtype=np.float32)

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
        self.last_action_cmd[:] = 0.0
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

        self.pos[:] = (float(self.fixed_start[0]), float(self.fixed_start[1]), self.target_altitude)
        self.vel[:] = 0.0
        self.angles[:] = 0.0
        self.rates[:] = 0.0
        self.last_cell = self.fixed_start
        self.state = (*self.fixed_start, 0)
        return self._get_obs(), {}

    def _current_cell(self):
        row = int(np.rint(self.pos[0]))
        col = int(np.rint(self.pos[1]))
        return row, col

    def _target_attitude(self, action):
        roll_cmd, pitch_cmd = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # Positive pitch accelerates south/down rows; positive roll accelerates west/left columns.
        target_roll = roll_cmd * self.max_tilt
        target_pitch = pitch_cmd * self.max_tilt
        return target_roll, target_pitch, 0.0

    def _integrate_physics(self, action):
        target_angles = np.array(self._target_attitude(action), dtype=np.float32)
        for _ in range(self.physics_substeps):
            angle_error = target_angles - self.angles
            desired_rates = angle_error / max(self.attitude_tau, 1e-6)
            self.rates += (desired_rates - self.rates) * (1.0 - self.angular_damping)
            self.angles += self.rates * self.dt
            self.angles[0:2] = np.clip(self.angles[0:2], -self.max_tilt, self.max_tilt)

            roll, pitch, _ = self.angles
            horizontal_accel = np.array(
                [
                    self.gravity * np.sin(pitch),
                    -self.gravity * np.sin(roll),
                ],
                dtype=np.float32,
            )
            altitude_error = self.target_altitude - self.pos[2]
            vertical_accel = self.altitude_kp * altitude_error - self.altitude_kd * self.vel[2]
            accel = np.array([horizontal_accel[0], horizontal_accel[1], vertical_accel])
            accel -= self.linear_drag * self.vel

            self.vel += accel * self.dt
            self.pos += self.vel * self.dt

    def _get_obs(self):
        r, c, has_pkg = self.state
        tr, tc = self.fixed_delivery if has_pkg else self.fixed_package
        surroundings = [
            1.0 if (r + dr, c + dc) in self.lane_set else 0.0
            for dr, dc in self.actions
        ]
        last_action_one_hot = [0.0, 0.0, 0.0, 0.0]
        last_action_one_hot[0:2] = self.last_action_cmd.tolist()
        current_visits = self.visit_counts.get((r, c), 0)

        vel_scale = 6.0
        rate_scale = 8.0
        return np.array(
            [
                r / self.grid_shape[0],
                c / self.grid_shape[1],
                (tr - r) / self.grid_shape[0],
                (tc - c) / self.grid_shape[1],
                (self.pos[0] - r),
                (self.pos[1] - c),
                (self.pos[2] - self.target_altitude),
                *(np.clip(self.vel / vel_scale, -1.0, 1.0)),
                *(np.clip(self.angles / self.max_tilt, -1.0, 1.0)),
                *(np.clip(self.rates / rate_scale, -1.0, 1.0)),
                *surroundings,
                float(tr < r),
                float(tr > r),
                float(tc < c),
                float(tc > c),
                float(has_pkg),
                *last_action_one_hot,
                self.last_move_invalid,
                min(current_visits / 5.0, 1.0),
            ],
            dtype=np.float32,
        )

    def step(self, action):
        self.current_step += 1
        self.total_steps += 1
        r, c, has_pkg = self.state
        self.visit_counts[(r, c)] = self.visit_counts.get((r, c), 0) + 1

        reward = self.step_penalty
        done = False
        truncated = self.current_step >= self.max_steps
        invalid_move = False

        self._integrate_physics(action)
        nr, nc = self._current_cell()
        if (nr, nc) not in self.lane_set:
            invalid_move = True
            nr, nc = r, c
            self.pos[0] = float(r)
            self.pos[1] = float(c)
            self.vel[0:2] = 0.0

        crashed = (
            self.pos[2] < self.crash_altitude
            or abs(float(self.angles[0])) > self.crash_tilt
            or abs(float(self.angles[1])) > self.crash_tilt
        )
        if invalid_move or crashed:
            reward -= self.invalid_move_penalty
            done = True

        moved_cells = (nr, nc) != (r, c)
        if moved_cells:
            visits = self.visit_counts.get((nr, nc), 0)
            reward -= min(self.revisit_penalty * visits, self.revisit_penalty_cap)

        if moved_cells and self.prev_state is not None and (nr, nc) == self.prev_state[:2]:
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

        if truncated and has_pkg and not done:
            reward -= self.undelivered_package_penalty

        self.state = (nr, nc, has_pkg)
        self.prev_state = (r, c, has_pkg)
        self.last_cell = (nr, nc)
        self.last_action_cmd = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.last_action = self.last_action_cmd.copy()
        self.last_move_invalid = float(invalid_move or crashed)

        if self.print_freq and self.total_steps % self.print_freq == 0:
            print(
                f"[QuadEnv] Steps {self.total_steps - self.print_freq + 1}-{self.total_steps} | "
                f"pickups={self.pickups_since_report} | "
                f"deliveries={self.deliveries_since_report} | "
                f"total_pickups={self.total_pickups} | total_deliveries={self.total_deliveries}"
            )
            self.pickups_since_report = 0
            self.deliveries_since_report = 0

        return self._get_obs(), reward, done, truncated, {}
