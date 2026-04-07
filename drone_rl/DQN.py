import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from classical_methods.utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions


#  PATHFINDING UTILITIES

def bfs_distance_map(bw_map, goal):
    H, W = bw_map.shape
    dist = np.full((H, W), -1, dtype=np.int32)
    dist[goal] = 0
    q = deque([goal])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and bw_map[nr,nc] == 1 and dist[nr,nc] < 0:
                dist[nr,nc] = dist[r,c] + 1
                q.append((nr,nc))
    return dist


def astar(bw_map, start, goal):
    lane = (bw_map == 1)
    H, W = bw_map.shape
    def h(r, c): return abs(r-goal[0]) + abs(c-goal[1])
    heap = [(h(*start), 0, start, [start])]
    seen = set()
    while heap:
        _, g, cur, path = heapq.heappop(heap)
        if cur in seen: continue
        seen.add(cur)
        if cur == goal: return path
        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and lane[nr,nc] and (nr,nc) not in seen:
                heapq.heappush(heap, (g+1+h(nr,nc), g+1, (nr,nc), path+[(nr,nc)]))
    return []


def get_largest_component(bw_map):
    unvisited = set(map(tuple, np.argwhere(bw_map == 1)))
    best = set()
    while unvisited:
        seed = next(iter(unvisited))
        comp, q = set(), deque([seed])
        while q:
            cell = q.popleft()
            if cell in comp: continue
            comp.add(cell)
            r, c = cell
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                n = (r+dr, c+dc)
                if n in unvisited and n not in comp:
                    q.append(n)
        unvisited -= comp
        if len(comp) > len(best):
            best = comp
    return best


def pick_positions(bw_map):
    comp  = get_largest_component(bw_map)
    cells = list(comp)
    tmp   = np.zeros_like(bw_map)
    for r, c in comp: tmp[r, c] = 1
    start   = min(cells, key=lambda p: p[0]+p[1])
    dist_s  = bfs_distance_map(tmp, start)
    goal    = max(cells, key=lambda p: dist_s[p])
    dist_g  = bfs_distance_map(tmp, goal)
    package = max(cells, key=lambda p: min(dist_s[p], dist_g[p]))
    return start, package, goal


# ─────────────────────────────────────────────────────────────────────────────
#  STATE: LOCAL CNN PATCH  (what the drone actually "sees")
#
#  Rather than passing raw coordinates, we give the CNN a 3-channel local
#  crop around the drone:
#    Channel 0: maze walls/lanes  (0=wall, 1=lane)
#    Channel 1: package location  (1 at package cell, 0 elsewhere)
#    Channel 2: goal location     (1 at goal cell, 0 elsewhere)
#
#  The crop is PATCH_SIZE × PATCH_SIZE, centred on the drone, padded with
#  walls where the crop goes out-of-bounds.  This means:
#    - The network sees local topology (which directions are open)
#    - It sees where the package/goal are RELATIVE to the drone
#    - It generalises across positions because the input is translation-invariant
# ─────────────────────────────────────────────────────────────────────────────

PATCH_SIZE = 15   # must be odd


def make_state(bw_map, pos, package_pos, delivery_pos, has_pkg, H, W):
    """
    Returns a (3, PATCH_SIZE, PATCH_SIZE) float32 numpy array.
    Channel 0: local maze crop
    Channel 1: package marker (or zeros if already picked up)
    Channel 2: goal marker
    """
    r, c  = pos
    half  = PATCH_SIZE // 2
    patch = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    for pr in range(PATCH_SIZE):
        for pc in range(PATCH_SIZE):
            mr = r - half + pr   # maze row
            mc = c - half + pc   # maze col
            if 0 <= mr < H and 0 <= mc < W:
                patch[0, pr, pc] = bw_map[mr, mc]   # 1=lane, 0=wall
                if not has_pkg and (mr, mc) == package_pos:
                    patch[1, pr, pc] = 1.0           # package visible
                if (mr, mc) == delivery_pos:
                    patch[2, pr, pc] = 1.0           # goal visible

    # If package/goal are outside the local patch, add a global direction hint
    # in the border pixels so the agent still knows which way to go.
    # We encode direction as a faint gradient on the border row/col.
    def add_border_hint(channel, target_r, target_c):
        dr = np.clip((target_r - r) / (H + W), -1, 1)
        dc = np.clip((target_c - c) / (H + W), -1, 1)
        patch[channel,  0,  :] += max( dr, 0) * 0.5   # top row    → target above
        patch[channel, -1,  :] += max(-dr, 0) * 0.5   # bottom row → target below
        patch[channel,  :,  0] += max( dc, 0) * 0.5   # left col   → target left
        patch[channel,  :, -1] += max(-dc, 0) * 0.5   # right col  → target right

    if not has_pkg:
        add_border_hint(1, package_pos[0], package_pos[1])
    add_border_hint(2, delivery_pos[0], delivery_pos[1])

    return patch


# ─────────────────────────────────────────────────────────────────────────────
#  NETWORK:  CNN  +  Dueling head
# ─────────────────────────────────────────────────────────────────────────────

class CNNDQN(nn.Module):
    """
    Processes a (3, PATCH_SIZE, PATCH_SIZE) local view.
    Uses small conv layers (the patch is only 15×15) then a Dueling head.
    """
    def __init__(self, patch=PATCH_SIZE):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 64 * patch * patch
        self.fc_shared = nn.Sequential(
            nn.Linear(conv_out + 1, 512), nn.ReLU(),   # +1 for has_pkg scalar
            nn.Linear(512, 256),          nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv   = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))

    def forward(self, patch_t, has_pkg_t):
        # patch_t:   (B, 3, P, P)
        # has_pkg_t: (B, 1)
        x = self.conv(patch_t)
        x = torch.cat([x, has_pkg_t], dim=1)
        x = self.fc_shared(x)
        v = self.value(x)
        a = self.adv(x)
        return v + a - a.mean(-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT
# ─────────────────────────────────────────────────────────────────────────────

class CNNDQNDrone:
    def __init__(self, bw_map, package_pos, delivery_pos, start_pos):
        self.bw_map       = bw_map
        self.package_pos  = tuple(int(x) for x in package_pos)
        self.delivery_pos = tuple(int(x) for x in delivery_pos)
        self.start_pos    = tuple(int(x) for x in start_pos)
        self.H, self.W    = bw_map.shape

        self.actions  = [(-1,0),(1,0),(0,-1),(0,1)]
        self.lane_set = set(map(tuple, np.argwhere(bw_map == 1)))
        self.lane_list = list(self.lane_set)

        print("Pre-computing BFS distance maps...")
        self.dist_to_pkg  = bfs_distance_map(bw_map, self.package_pos)
        self.dist_to_goal = bfs_distance_map(bw_map, self.delivery_pos)
        d_sp = self.dist_to_pkg [self.start_pos]
        d_pg = self.dist_to_goal[self.package_pos]
        print(f"  start->pkg={d_sp}  pkg->goal={d_pg}")
        assert d_sp >= 0 and d_pg >= 0, "Positions are disconnected!"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model  = CNNDQN().to(self.device)
        self.target = CNNDQN().to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.memory    = deque(maxlen=100_000)

        self.gamma       = 0.99
        self.batch_size  = 64    # smaller batch — CNN is heavier per sample
        self.train_freq  = 4
        self.target_freq = 1000

        # Curriculum pools
        self.near_goal = [(r,c) for (r,c) in self.lane_list
                          if 0 <= self.dist_to_goal[r,c] <= 25]
        self.near_pkg  = [(r,c) for (r,c) in self.lane_list
                          if 0 <= self.dist_to_pkg[r,c]  <= 25]
        print(f"  near_goal pool={len(self.near_goal)}  near_pkg pool={len(self.near_pkg)}")

    # ── Build state tensor from (pos, has_pkg) ────────────────────────────
    def _state(self, pos, has_pkg):
        return make_state(self.bw_map, pos, self.package_pos,
                          self.delivery_pos, has_pkg, self.H, self.W)

    def _to_tensors(self, states):
        """states: list of (patch_array, has_pkg_bool)"""
        patches  = np.stack([s[0] for s in states])
        has_pkgs = np.array([[float(s[1])] for s in states], dtype=np.float32)
        return (torch.FloatTensor(patches).to(self.device),
                torch.FloatTensor(has_pkgs).to(self.device))

    # ── Boltzmann action selection ──────────────────────────────
    def choose_action(self, pos, has_pkg, tau):
        patch = self._state(pos, has_pkg)
        pt    = torch.FloatTensor(patch).unsqueeze(0).to(self.device)
        hp_t  = torch.FloatTensor([[float(has_pkg)]]).to(self.device)
        with torch.no_grad():
            q = self.model(pt, hp_t).squeeze(0)
        if tau < 0.01:
            return q.argmax().item()
        probs = F.softmax(q / tau, dim=0).cpu().numpy()
        return np.random.choice(4, p=probs)

    def step_env(self, pos, has_pkg, a):
        r, c   = pos
        dr, dc = self.actions[a]
        nr, nc = r+dr, c+dc

        if (nr,nc) not in self.lane_set:
            nr, nc = r, c
            wall_pen = -2.0
        else:
            wall_pen = 0.0

        if has_pkg:
            old_d = self.dist_to_goal[r,  c ]
            new_d = self.dist_to_goal[nr, nc]
        else:
            old_d = self.dist_to_pkg[r,  c ]
            new_d = self.dist_to_pkg[nr, nc]

        max_d = self.H + self.W
        if old_d < 0: old_d = max_d
        if new_d < 0: new_d = max_d

        reward      = -0.1 + float(old_d - new_d) + wall_pen
        new_has_pkg = has_pkg
        done        = False

        if not has_pkg and (nr,nc) == self.package_pos:
            new_has_pkg = True
            reward += 20.0

        if has_pkg and (nr,nc) == self.delivery_pos:
            reward += 100.0
            done = True

        return (nr,nc), new_has_pkg, reward, done

    def store(self, state, a, r, next_state, done):
        self.memory.append((state, a, r, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)

        states     = [x[0] for x in batch]
        actions    = torch.LongTensor( [x[1] for x in batch]).to(self.device)
        rewards    = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        nxt_states = [x[3] for x in batch]
        dones      = torch.FloatTensor([x[4] for x in batch]).to(self.device)

        pt,  hp  = self._to_tensors(states)
        npt, nhp = self._to_tensors(nxt_states)

        q = self.model(pt, hp).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_a = self.model(npt, nhp).argmax(1, keepdim=True)
            nq     = self.target(npt, nhp).gather(1, best_a).squeeze(1)
            tq     = rewards + self.gamma * nq * (1 - dones)

        loss = nn.SmoothL1Loss()(q, tq)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

    def sample_start(self):
        rv = random.random()
        if rv < 0.35:
            pos = random.choice(self.near_goal)
            return pos, True          # near goal, carrying package
        elif rv < 0.65:
            pos = random.choice(self.near_pkg)
            return pos, False         # near package, not carrying
        else:
            return self.start_pos, False   # full task

    def train(self, episodes=2000):
        print("Training CNN-DQN with Boltzmann exploration...")
        successes   = 0
        global_step = 0
        best_eval   = 0

        # Boltzmann temperature schedule: starts high, anneals to near-zero
        tau_start = 5.0
        tau_end   = 0.05
        tau_decay = episodes * 0.8   

        for ep in range(episodes):
            tau = max(tau_end, tau_start - (tau_start - tau_end) * ep / tau_decay)

            pos, has_pkg = self.sample_start()
            total_reward = 0

            for _ in range(500):
                a                       = self.choose_action(pos, has_pkg, tau)
                s_now                   = (self._state(pos, has_pkg), has_pkg)
                new_pos, new_hp, r, done = self.step_env(pos, has_pkg, a)
                s_next                  = (self._state(new_pos, new_hp), new_hp)

                self.store(s_now, a, r, s_next, float(done))
                global_step  += 1
                total_reward += r
                pos, has_pkg  = new_pos, new_hp

                if global_step % self.train_freq  == 0: self.train_step()
                if global_step % self.target_freq == 0:
                    self.target.load_state_dict(self.model.state_dict())

                if done:
                    successes += 1
                    break

            if ep % 50 == 0:
                eval_ok = sum(self._eval() for _ in range(5))
                print(f"  Ep {ep:4d} | tau={tau:.3f} | "
                      f"reward={total_reward:7.1f} | "
                      f"train_ok={successes} | eval={eval_ok}/5")
                if eval_ok > best_eval:
                    best_eval = eval_ok
                    torch.save(self.model.state_dict(), "cnn_dqn_best.pt")

        print(f"\nDone. Best eval: {best_eval}/5")
        try:
            self.model.load_state_dict(
                torch.load("cnn_dqn_best.pt", map_location=self.device, weights_only=True))
            print("Loaded best checkpoint.")
        except Exception:
            pass

    def _eval(self, max_steps=600):
        pos, has_pkg = self.start_pos, False
        for _ in range(max_steps):
            a = self.choose_action(pos, has_pkg, tau=0.0)
            pos, has_pkg, _, done = self.step_env(pos, has_pkg, a)
            if done: return 1
        return 0

    def run(self):
        pos, has_pkg = self.start_pos, False
        path = [(pos[0], pos[1], has_pkg)]
        done = False
        for _ in range(800):
            a = self.choose_action(pos, has_pkg, tau=0.0)
            pos, has_pkg, _, done = self.step_env(pos, has_pkg, a)
            path.append((pos[0], pos[1], has_pkg))
            if done: break
        return path, done


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    grid_size = (12, 12)
    pg        = PipeGrid(*grid_size)
    vis       = PipeVisualizerBW(lanes=2, base=3)
    bw_map    = vis.render(pg.to_pipe_ids(PipeOptions()))

    print(f"Map: {bw_map.shape}")
    start, package, goal = pick_positions(bw_map)
    print(f"Start={start}  Package={package}  Goal={goal}")

    agent = CNNDQNDrone(bw_map, package, goal, start)
    agent.train(episodes=3000)

    # A* reference
    ref1 = astar(bw_map, start,   package)
    ref2 = astar(bw_map, package, goal)
    print(f"A* optimal: {len(ref1)+len(ref2)-1} steps")

    path, success = agent.run()
    print(f"Agent: {len(path)} steps | success={success}")

    # ── Visualise ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9,9))
    ax.imshow(bw_map, cmap="gray")
    ax.set_xlim(-0.5, bw_map.shape[1]-0.5)
    ax.set_ylim(bw_map.shape[0]-0.5, -0.5)

    p_mark, = ax.plot(package[1], package[0], 'ys', markersize=14, zorder=5, label="Package")
    g_mark, = ax.plot(goal[1],    goal[0],    'g^', markersize=14, zorder=5, label="Goal")
    d_mark, = ax.plot([], [],                 'ro', markersize=8,  zorder=6, label="Drone")
    title   = ax.set_title("", fontsize=11)

    def update(i):
        r, c, has_pkg = path[i]
        d_mark.set_data([c], [r])
        if has_pkg:
            d_mark.set_color('orange'); p_mark.set_visible(False)
            title.set_text(f"Step {i}/{len(path)-1} — carrying → goal")
        else:
            d_mark.set_color('red'); p_mark.set_visible(True)
            title.set_text(f"Step {i}/{len(path)-1} — searching for package")
        return d_mark, p_mark, title

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=40, blit=True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    ani.save("dqn_fixed.gif", writer="pillow")
    print("Saved dqn_fixed.gif")
    plt.show()