import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from tqdm import tqdm
from utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions

class DeliveryDroneTDLambdaForward:
    def __init__(self, bw_map, start_pos, package_pos, delivery_pos,
                 alpha=0.1, gamma=0.95, epsilon=0.4, lmbda=0.9):
        
        self.bw_map = bw_map
        self.start_pos = start_pos
        self.package_pos = package_pos
        self.delivery_pos = delivery_pos
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lmbda = lmbda
        
        self.actions = [(-1,0),(1,0),(0,-1),(0,1)]
        self.lane_coords = [tuple(p) for p in np.argwhere(bw_map==1)]
        
        # Value function
        self.V = {}
        for s in self.lane_coords:
            self.V[(s[0], s[1], False)] = 50.0
            self.V[(s[0], s[1], True)] = 50.0
        
        # Force delivery to be high value
        self.V[(delivery_pos[0], delivery_pos[1], True)] = 10000.0

        self.train(episodes=300)

    def get_valid_actions(self, state):
        r, c, _ = state
        return [(dr,dc) for dr,dc in self.actions if (r+dr, c+dc) in self.lane_coords]

    def reward(self, state, next_state):
        r, c, has_pkg = state
        nr, nc, n_has_pkg = next_state
        if not has_pkg and n_has_pkg:
            return 500
        if n_has_pkg and (nr, nc) == self.delivery_pos:
            return 1000
        return -1

    def step(self, state, action):
        r, c, has_pkg = state
        dr, dc = action
        nr, nc = r+dr, c+dc
        if (nr, nc) not in self.lane_coords:
            return (r, c, has_pkg)
        return (nr, nc, has_pkg)

    def epsilon_greedy(self, state):
        valid = self.get_valid_actions(state)
        if not valid:
            return self.actions[0]
        if np.random.rand() < self.epsilon:
            return valid[np.random.randint(len(valid))]
        
        vals = []
        for a in valid:
            ns = self.step(state, a)
            if not state[2] and ns[:2] == self.package_pos:
                ns = (ns[0], ns[1], True)
            vals.append(self.V.get(ns, 0))
        
        return valid[np.argmax(vals)]

    def train(self, episodes=300):
        max_steps = 400
        decay_rate = 0.99
        success_count = 0
        N = 20  # truncation horizon

        for ep in tqdm(range(episodes), desc="Training Progress"):
            self.epsilon = max(0.01, self.epsilon * decay_rate)
            state = (self.start_pos[0], self.start_pos[1], False)

            trajectory = [state]
            rewards = []

            for _ in range(max_steps):
                action = self.epsilon_greedy(state)
                next_state = self.step(state, action)

                if not state[2] and next_state[:2] == self.package_pos:
                    next_state = (next_state[0], next_state[1], True)

                r = self.reward(state, next_state)

                trajectory.append(next_state)
                rewards.append(r)

                state = next_state

                if state[2] and state[:2] == self.delivery_pos:
                    success_count += 1
                    break

            # --- FAST FORWARD TD(λ) ---
            T = len(trajectory)

            for t in range(T-1):
                G_lambda = 0.0

                max_n = min(N, T - t - 1)

                for n in range(1, max_n + 1):
                    G_n = 0.0
                    for k in range(n):
                        G_n += (self.gamma ** k) * rewards[t + k]

                    if t + n < T:
                        G_n += (self.gamma ** n) * self.V.get(trajectory[t + n], 0)

                    G_lambda += (self.lmbda ** (n - 1)) * G_n

                G_lambda *= (1 - self.lmbda)

                self.V[trajectory[t]] += self.alpha * (G_lambda - self.V[trajectory[t]])

            if success_count > 200:
                print(f"\nEarly stopping at episode {ep}")
                break

    def bfs_distances(self, target):
        dist = {pos: float('inf') for pos in self.lane_coords}
        dist[target] = 0
        q = deque([target])

        while q:
            r, c = q.popleft()
            for dr, dc in self.actions:
                nr, nc = r+dr, c+dc
                if (nr, nc) in self.lane_coords and dist[(nr,nc)] > dist[(r,c)] + 1:
                    dist[(nr,nc)] = dist[(r,c)] + 1
                    q.append((nr,nc))
        return dist

    def get_action(self, state, visited=None):
        r, c, has_pkg = state

        if not has_pkg and (r, c) == self.package_pos:
            return "PICKUP"
        if has_pkg and (r, c) == self.delivery_pos:
            return "DONE"

        valid = self.get_valid_actions(state)
        if not valid:
            return self.actions[0]

        target = self.package_pos if not has_pkg else self.delivery_pos
        dist_map = self.bfs_distances(target)

        # Direct delivery move
        if has_pkg:
            for a in valid:
                nr, nc = r+a[0], c+a[1]
                if (nr, nc) == self.delivery_pos:
                    return a

        # Before pickup
        if not has_pkg:
            return min(valid, key=lambda a: dist_map[(r+a[0], c+a[1])])

        # After pickup: value + distance
        def score(a):
            ns = self.step(state, a)
            v = self.V.get(ns, 0)
            d = dist_map[(ns[0], ns[1])]
            if visited and ns in visited:
                v -= 0.5
            return v - d

        return max(valid, key=score)


# --- MAIN ---
if __name__=="__main__":
    grid_size = (12,12)
    pg = PipeGrid(*grid_size)
    vis = PipeVisualizerBW(lanes=2, base=3)
    bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))
    
    lane_coords = [tuple(p) for p in np.argwhere(bw_map==1)]
    start_pos = lane_coords[0]
    package_pos = lane_coords[len(lane_coords)//2]
    delivery_pos = lane_coords[-1]

    print("Training Forward TD(λ)...")
    drone = DeliveryDroneTDLambdaForward(bw_map, start_pos, package_pos, delivery_pos)

    # --- EVALUATION ---
    state = (start_pos[0], start_pos[1], False)
    path = [state]
    visited = {state}

    for _ in range(500):
        action = drone.get_action(state, visited)

        if action == "DONE":
            print("Delivery complete!")
            break
        elif action == "PICKUP":
            state = (state[0], state[1], True)
        else:
            state = drone.step(state, action)
            if not state[2] and state[:2] == package_pos:
                state = (state[0], state[1], True)

        path.append(state)
        visited.add(state)

    # --- ANIMATION ---
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(bw_map, cmap="gray")

    p_mark, = ax.plot(package_pos[1], package_pos[0], 'ys', markersize=10)
    d_mark, = ax.plot(delivery_pos[1], delivery_pos[0], 'gx', markersize=12)
    drone_mark, = ax.plot([], [], 'ro', markersize=7)

    def update(frame):
        r, c, has_pkg = path[frame]
        drone_mark.set_data([c], [r])
        drone_mark.set_color('orange' if has_pkg else 'red')
        p_mark.set_visible(not has_pkg)
        return drone_mark, p_mark, d_mark

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=120)
    plt.show()