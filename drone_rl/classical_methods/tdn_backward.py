import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions

class DeliveryDroneTDLambda:
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
        
        # --- Optimistic initialization ---
        self.V = {}
        for s in self.lane_coords:
            self.V[(s[0], s[1], False)] = 50.0
            self.V[(s[0], s[1], True)] = 50.0
        
        # Make delivery state extremely high value to prevent oscillation
        self.V[(delivery_pos[0], delivery_pos[1], True)] = 10000.0

        self.train(episodes=1000)

    def get_valid_actions(self, state):
        r, c, _ = state
        valid = []
        for dr, dc in self.actions:
            if (r+dr, c+dc) in self.lane_coords:
                valid.append((dr, dc))
        return valid

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
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return self.actions[0]
        if np.random.rand() < self.epsilon:
            return valid_actions[np.random.randint(len(valid_actions))]
        vals = []
        for a in valid_actions:
            ns = self.step(state, a)
            if not state[2] and ns[:2] == self.package_pos:
                ns = (ns[0], ns[1], True)
            vals.append(self.V.get(ns, 0) + 1e-5*np.random.rand())
        max_v = max(vals)
        best_indices = [i for i, v in enumerate(vals) if v == max_v]
        return valid_actions[best_indices[np.random.randint(len(best_indices))]]

    def train(self, episodes=600):
        max_steps = 1500
        decay_rate = 0.99
        success_count = 0

        for ep in range(episodes):
            state = (self.start_pos[0], self.start_pos[1], False)
            E = {s: 0 for s in self.V}
            self.epsilon = max(0.01, self.epsilon * decay_rate)
            
            for _ in range(max_steps):
                action = self.epsilon_greedy(state)
                next_state = self.step(state, action)
                if not state[2] and next_state[:2] == self.package_pos:
                    next_state = (next_state[0], next_state[1], True)
                
                r = self.reward(state, next_state)
                delta = r + self.gamma * self.V.get(next_state, 0) - self.V[state]
                E[state] += 1
                
                for s in self.V:
                    if E[s] > 0.001:
                        self.V[s] += self.alpha * delta * E[s]
                        E[s] *= self.gamma * self.lmbda
                
                state = next_state
                if state[2] and state[:2] == self.delivery_pos:
                    success_count += 1
                    break

            if ep % 100 == 0:
                print(f"Episode {ep} | Successes so far: {success_count} | Epsilon: {self.epsilon:.2f}")

    def bfs_distances(self, target):
        dist_map = {pos: float('inf') for pos in self.lane_coords}
        queue = deque()
        dist_map[target] = 0
        queue.append(target)
        while queue:
            r, c = queue.popleft()
            for dr, dc in self.actions:
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.lane_coords and dist_map[(nr, nc)] > dist_map[(r, c)] + 1:
                    dist_map[(nr, nc)] = dist_map[(r, c)] + 1
                    queue.append((nr, nc))
        return dist_map

    def get_action(self, state, visited=None):
        r, c, has_pkg = state
        if not has_pkg and (r, c) == self.package_pos:
            return "PICKUP"
        if has_pkg and (r, c) == self.delivery_pos:
            return "DONE"

        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return self.actions[0]

        target = self.package_pos if not has_pkg else self.delivery_pos
        dist_map = self.bfs_distances(target)

        # --- HARD GOAL CHECK: move directly into delivery if adjacent ---
        if has_pkg:
            for a in valid_actions:
                nr, nc = r + a[0], c + a[1]
                if (nr, nc) == self.delivery_pos:
                    return a  # move directly into delivery

        if not has_pkg:
            # Before pickup: move along BFS path
            min_dist = float('inf')
            best_actions = []
            for a in valid_actions:
                nr, nc = r + a[0], c + a[1]
                d = dist_map.get((nr, nc), float('inf'))
                if d < min_dist:
                    min_dist = d
                    best_actions = [a]
                elif d == min_dist:
                    best_actions.append(a)
            return best_actions[np.random.randint(len(best_actions))]

        # After pickup: value-weighted + BFS tiebreak
        best_v = -float('inf')
        best_actions = []
        for a in valid_actions:
            ns = self.step(state, a)
            if not has_pkg and ns[:2] == self.package_pos:
                ns = (ns[0], ns[1], True)
            v = self.V.get(ns, 0)
            if visited and ns in visited:
                v -= 0.5
            d = dist_map.get((ns[0], ns[1]), float('inf'))
            score = v - d  # combine value + reachable distance
            if score > best_v:
                best_v = score
                best_actions = [a]
            elif score == best_v:
                best_actions.append(a)
        return best_actions[np.random.randint(len(best_actions))]

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

    print("Training Drone... Please wait.")
    drone = DeliveryDroneTDLambda(bw_map, start_pos, package_pos, delivery_pos)

    # --- POST-TRAINING EVALUATION ---
    state = (start_pos[0], start_pos[1], False)
    path = [state]
    visited = set([state])
    max_steps = 500

    for _ in range(max_steps):
        action = drone.get_action(state, visited)
        if action == "DONE":
            print("Delivery complete!")
            break
        elif action == "PICKUP":
            state = (state[0], state[1], True)
            path.append(state)
        else:
            next_state = drone.step(state, action)
            if not next_state[2] and next_state[:2] == package_pos:
                next_state = (next_state[0], next_state[1], True)
            state = next_state
            path.append(state)
        visited.add(state)

    # --- ANIMATION ---
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(bw_map, cmap="gray")
    p_mark, = ax.plot(package_pos[1], package_pos[0], 'ys', markersize=10, label="Package")
    d_mark, = ax.plot(delivery_pos[1], delivery_pos[0], 'gx', markersize=12, label="Delivery")
    drone_mark, = ax.plot([], [], 'ro', markersize=7, label="Drone")
    status_text = ax.set_title("TD(λ) Delivery (Corrected)")

    def update(frame):
        r, c, has_pkg = path[frame]
        drone_mark.set_data([c], [r])
        if has_pkg:
            drone_mark.set_color('orange')
            p_mark.set_visible(False)
            status_text.set_text(f"Delivering - Step {frame}")
        else:
            drone_mark.set_color('red')
            p_mark.set_visible(True)
            status_text.set_text(f"Fetching Package - Step {frame}")
        return drone_mark, p_mark, d_mark, status_text

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=120, blit=False, repeat=False)
    plt.legend(loc='lower right')
    plt.show()