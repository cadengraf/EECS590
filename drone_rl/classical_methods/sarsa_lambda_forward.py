import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions
from utils.saliency import run_saliency_suite

class SARSADrone:
    def __init__(self, bw_map, package_pos, delivery_pos,
                 alpha=0.1, gamma=0.95, epsilon=1.0):

        self.bw_map = bw_map
        self.package_pos = package_pos
        self.delivery_pos = delivery_pos

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.actions = [(-1,0),(1,0),(0,-1),(0,1)]
        self.lane_coords = [tuple(p) for p in np.argwhere(bw_map == 1)]
        self.Q = {}

    def get_Q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.actions))
        return self.Q[state]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.actions))
        q = self.get_Q(state)
        return np.random.choice(np.flatnonzero(q == q.max()))

    def step_env(self, state, action_idx, visited):
        r, c, has_pkg = state
        dr, dc = self.actions[action_idx]
        nr, nc = r + dr, c + dc

        if (nr, nc) not in self.lane_coords:
            nr, nc = r, c

        new_has_pkg = has_pkg
        reward = -1.0  # base step penalty

        # Loop/backtracking penalty
        if (nr, nc, has_pkg) in visited:
            reward -= 3.0
        visited.add((nr, nc, has_pkg))

        # Pickup reward
        if not has_pkg and (nr, nc) == self.package_pos:
            new_has_pkg = True
            reward += 150

        # Delivery reward (dominant)
        if new_has_pkg and (nr, nc) == self.delivery_pos:
            reward += 500

        # Directional shaping
        neighbors = [(nr+dr2, nc+dc2) for dr2, dc2 in self.actions if (nr+dr2, nc+dc2) in self.lane_coords]
        unvisited_neighbors = sum(1 for nb in neighbors if (nb[0], nb[1], new_has_pkg) not in visited)
        reward += 0.5 * unvisited_neighbors  

        return (nr, nc, new_has_pkg), reward

    def train_lambda_forward(self, start_pos, lmbda=0.8, episodes=3000, max_steps=500):
        print("Training forward-view SARSA(lambda)...")

        for ep in range(episodes):
            visited = set()
            episode = []

            state = (start_pos[0], start_pos[1], False)
            action = self.choose_action(state)

            for _ in range(max_steps):
                next_state, reward = self.step_env(state, action, visited)
                next_action = self.choose_action(next_state)
                episode.append((state, action, reward))

                state = next_state
                action = next_action

                if state[2] and (state[0], state[1]) == self.delivery_pos:
                    episode.append((state, action, 0))
                    break

            T = len(episode)

            # Precompute returns
            G = np.zeros(T + 1)
            for t in reversed(range(T)):
                G[t] = episode[t][2] + self.gamma * G[t + 1]

            # Lambda returns
            for t in range(T - 1):
                G_lambda = 0.0
                lambda_pow = 1.0
                for n in range(1, T - t):
                    G_n = G[t] - (self.gamma ** n) * G[t + n]
                    if t + n < T:
                        s_n, a_n, _ = episode[t + n]
                        G_n += (self.gamma ** n) * self.get_Q(s_n)[a_n]
                    G_lambda += lambda_pow * G_n
                    lambda_pow *= lmbda
                G_lambda *= (1 - lmbda)

                s, a, _ = episode[t]
                Q_s = self.get_Q(s)
                Q_s[a] += self.alpha * (G_lambda - Q_s[a])

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            if ep % 50 == 0:
                print(f"Episode {ep}, epsilon={self.epsilon:.3f}")

        print("Training complete!")

    def get_action(self, state):
        q = self.get_Q(state)
        return self.actions[np.random.choice(np.flatnonzero(q == q.max()))]

# --- MAIN ---
if __name__ == "__main__":
    grid_size = (6, 6)
    pg = PipeGrid(*grid_size)
    vis = PipeVisualizerBW(lanes=2, base=3)
    bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))

    lane_coords = [tuple(p) for p in np.argwhere(bw_map == 1)]
    start_pos = lane_coords[0]
    package_pos = lane_coords[len(lane_coords)//2]
    delivery_pos = lane_coords[-1]

    drone = SARSADrone(bw_map, package_pos, delivery_pos)
    drone.train_lambda_forward(start_pos, episodes=3000, max_steps=500)

    # --- RUN POLICY ---
    curr_state = (start_pos[0], start_pos[1], False)
    path = [curr_state]

    for step in range(500):
        action = drone.get_action(curr_state)
        r, c, has_pkg = curr_state
        nr, nc = r + action[0], c + action[1]
        if (nr, nc) not in drone.lane_coords:
            nr, nc = r, c

        if not has_pkg and (nr, nc) == package_pos:
            curr_state = (nr, nc, True)
        else:
            curr_state = (nr, nc, has_pkg)

        path.append(curr_state)
        if curr_state[2] and (nr, nc) == delivery_pos:
            print(f"Delivered in {len(path)} steps!")
            break

    run_saliency_suite(drone, path, bw_map, show=True)

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bw_map, cmap="gray")
    ax.plot(package_pos[1], package_pos[0], 'ys', markersize=10)
    ax.plot(delivery_pos[1], delivery_pos[0], 'gx', markersize=12)
    drone_mark, = ax.plot([], [], 'ro', markersize=7)

    def update(frame):
        r, c, has_pkg = path[frame]
        drone_mark.set_data([c], [r])
        drone_mark.set_color('orange' if has_pkg else 'red')
        return drone_mark,

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=60)
    ani.save("forward_lambda.gif", writer="pillow")
    plt.show()
