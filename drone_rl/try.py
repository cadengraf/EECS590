import argparse
import numpy as np
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions

PATCH = 15  # local view size (must be odd)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_map(grid_size=(12, 12)):
    pg = PipeGrid(*grid_size)
    return PipeVisualizerBW(lanes=2, base=3).render(pg.to_pipe_ids(PipeOptions()))


def bfs(bw_map, goal):
    H, W = bw_map.shape
    dist = np.full((H, W), -1)
    dist[goal] = 0
    q = deque([goal])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W and bw_map[nr,nc]==1 and dist[nr,nc]<0:
                dist[nr,nc] = dist[r,c]+1
                q.append((nr,nc))
    return dist


def pick_positions(bw_map):
    """Spread start, package, goal as far apart as possible."""
    cells = list(map(tuple, np.argwhere(bw_map == 1)))
    start = min(cells, key=lambda p: p[0]+p[1])
    d_s   = bfs(bw_map, start)
    goal  = max(cells, key=lambda p: d_s[p])
    d_g   = bfs(bw_map, goal)
    pkg   = max(cells, key=lambda p: min(d_s[p], d_g[p]))
    return start, pkg, goal


def make_obs(bw_map, pos, pkg, goal, has_pkg, H, W):
    """3-channel local patch centred on the drone."""
    r, c  = pos
    half  = PATCH // 2
    patch = np.zeros((3, PATCH, PATCH), dtype=np.float32)
    for pr in range(PATCH):
        for pc in range(PATCH):
            mr, mc = r-half+pr, c-half+pc
            if 0<=mr<H and 0<=mc<W:
                patch[0,pr,pc] = bw_map[mr,mc]
                if not has_pkg and (mr,mc)==pkg:  patch[1,pr,pc] = 1.
                if (mr,mc)==goal:                 patch[2,pr,pc] = 1.
    # border hints when target is outside the local patch
    for ch, (tr, tc) in [(1, pkg), (2, goal)]:
        if ch == 1 and has_pkg: continue
        dr = np.clip((tr-r)/(H+W), -1, 1)
        dc = np.clip((tc-c)/(H+W), -1, 1)
        patch[ch, 0,  :] += max( dr,0)*.5
        patch[ch,-1,  :] += max(-dr,0)*.5
        patch[ch, :,  0] += max( dc,0)*.5
        patch[ch, :, -1] += max(-dc,0)*.5
    return patch


# ── Gym environment ───────────────────────────────────────────────────────────

class DroneEnv(gym.Env):
    def __init__(self, grid_size=(12,12), max_steps=500):
        super().__init__()
        self.max_steps         = max_steps
        self.action_space      = spaces.Discrete(4)
        self.observation_space = spaces.Box(0., 1., (3, PATCH, PATCH), np.float32)
        self._moves            = [(-1,0),(1,0),(0,-1),(0,1)]
        self._init_map(grid_size)

    def _init_map(self, grid_size):
        self.bw            = build_map(grid_size)
        self.H, self.W     = self.bw.shape
        self.lanes         = set(map(tuple, np.argwhere(self.bw == 1)))
        self.start, self.pkg, self.goal = pick_positions(self.bw)
        self.d_pkg         = bfs(self.bw, self.pkg)
        self.d_goal        = bfs(self.bw, self.goal)

    def reset(self, seed=None, options=None):
        options = options or {}
        if "new_map_size" in options:
            self._init_map(options["new_map_size"])
        elif options.get("randomize_positions"):
            _, self.pkg, self.goal = pick_positions(self.bw)
            self.d_pkg  = bfs(self.bw, self.pkg)
            self.d_goal = bfs(self.bw, self.goal)

        # eval=True skips curriculum so we always test the full task
        if options.get("eval"):
            self.pos, self.has_pkg = self.start, False
        else:
            r = np.random.random()
            if r < 0.20:
                pool = [c for c in self.lanes if 0 <= self.d_goal[c] <= 20]
                self.pos, self.has_pkg = (pool[np.random.randint(len(pool))] if pool else self.start), True
            elif r < 0.40:
                pool = [c for c in self.lanes if 0 <= self.d_pkg[c] <= 20]
                self.pos, self.has_pkg = (pool[np.random.randint(len(pool))] if pool else self.start), False
            else:
                self.pos, self.has_pkg = self.start, False  # full task

        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        r, c   = self.pos
        dr, dc = self._moves[action]
        nr, nc = r+dr, c+dc
        valid  = (nr,nc) in self.lanes
        nr, nc = (nr,nc) if valid else (r,c)

        d_map  = self.d_goal if self.has_pkg else self.d_pkg
        M      = self.H + self.W
        old_d  = d_map[r,  c ] if d_map[r,  c ] >= 0 else M
        new_d  = d_map[nr, nc] if d_map[nr, nc] >= 0 else M
        reward = -0.1 + 2.0 * float(old_d - new_d)  # scaled shaping
        reward += 0. if valid else -2.

        if not self.has_pkg and (nr,nc)==self.pkg:
            self.has_pkg = True;  reward += 20.
        done = self.has_pkg and (nr,nc)==self.goal
        if done: reward += 100.

        self.pos = (nr,nc);  self.steps += 1
        return self._obs(), reward, done, self.steps>=self.max_steps, {}

    def _obs(self):
        return make_obs(self.bw, self.pos, self.pkg, self.goal,
                        self.has_pkg, self.H, self.W)


# ── CNN feature extractor for SB3 ─────────────────────────────────────────────

class PatchCNN(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=256):
        super().__init__(obs_space, features_dim)
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*PATCH*PATCH, features_dim), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)


# ── Train / test ──────────────────────────────────────────────────────────────

def train(grid_size=(12,12), timesteps=2_000_000, path="drone_dqn"):
    model = DQN("CnnPolicy", DroneEnv(grid_size),
                policy_kwargs=dict(features_extractor_class=PatchCNN,
                                   features_extractor_kwargs=dict(features_dim=256)),
                learning_rate=5e-4, buffer_size=100_000, batch_size=64,
                gamma=0.99, train_freq=4, target_update_interval=1000,
                learning_starts=5_000,        # fill buffer before training
                exploration_fraction=0.5,     # explore for longer
                exploration_final_eps=0.05, verbose=1)
    model.learn(timesteps)
    model.save(path)
    print(f"Saved → {path}.zip")


def evaluate(model, env, n=10, options=None):
    wins = 0
    eval_options = {**(options or {}), "eval": True}  # always full task during eval
    for _ in range(n):
        obs, _ = env.reset(options=eval_options)
        for _ in range(800):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = env.step(int(action))
            if done or trunc: wins += done; break
    print(f"  {wins}/{n} successes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--test",  action="store_true")
    ap.add_argument("--mode",  choices=["random_positions","new_map"], default="random_positions")
    ap.add_argument("--timesteps", type=int,   default=2_000_000)
    ap.add_argument("--map-size",  type=int,   nargs=2, default=[12,12])
    args = ap.parse_args()

    if args.train:
        train(tuple(args.map_size), args.timesteps)

    if args.test:
        model = DQN.load("drone_dqn")
        if args.mode == "random_positions":
            print("Same map, random positions:")
            evaluate(model, DroneEnv(tuple(args.map_size)), options={"randomize_positions": True})
        else:
            for size in [(8,8),(16,16),(20,20)]:
                print(f"Map size {size}:")
                evaluate(model, DroneEnv(), options={"new_map_size": size})

                