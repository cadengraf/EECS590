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
        return np.argmax(self.get_Q(state))

    # --- FIXED STEP FUNCTION (NO DISTANCE) ---
    def step_env(self, state, action_idx, prev_state, visit_counts):
        r, c, has_pkg = state
        dr, dc = self.actions[action_idx]

        nr, nc = r + dr, c + dc

        invalid_move = False
        if (nr, nc) not in self.lane_coords:
            nr, nc = r, c
            invalid_move = True

        new_has_pkg = has_pkg
        reward = -1.0  # base step penalty

        if invalid_move:
            reward -= 5.0

        visits = visit_counts.get((nr, nc), 0)
        reward -= 0.5 * visits

        if prev_state is not None and (nr, nc) == (prev_state[0], prev_state[1]):
            reward -= 3.0

        if not has_pkg and (nr, nc) == self.package_pos:
            new_has_pkg = True
            reward += 100

        elif has_pkg and (nr, nc) == self.delivery_pos:
            reward += 200

        return (nr, nc, new_has_pkg), reward

    def train(self, start_pos, episodes=2000, max_steps=1000):
        print("Training SARSA fixed...")

        for ep in range(episodes):
            state = (start_pos[0], start_pos[1], False)
            action = self.choose_action(state)
            prev_state = None
            visit_counts = {}

            for step in range(max_steps):
                # Track visits
                visit_counts[(state[0], state[1])] = visit_counts.get((state[0], state[1]), 0) + 1

                next_state, reward = self.step_env(state, action, prev_state, visit_counts)
                next_action = self.choose_action(next_state)

                Q_s = self.get_Q(state)
                Q_next = self.get_Q(next_state)

                Q_s[action] += self.alpha * (
                    reward + self.gamma * Q_next[next_action] - Q_s[action]
                )

                prev_state = state
                state = next_state
                action = next_action

                if state[2] and (state[0], state[1]) == self.delivery_pos:
                    break

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if ep % 50 == 0:
                print(f"Episode {ep}, epsilon={self.epsilon:.3f}")

        print("Training complete!")

    def get_action(self, state):
        return self.actions[np.argmax(self.get_Q(state))]


# --- MAIN ---
if __name__ == "__main__":
    grid_size = (12, 12)
    pg = PipeGrid(*grid_size)
    vis = PipeVisualizerBW(lanes=2, base=3)
    bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))

    lane_coords = [tuple(p) for p in np.argwhere(bw_map == 1)]

    start_pos = lane_coords[0]
    package_pos = lane_coords[len(lane_coords)//2]
    delivery_pos = lane_coords[-1]

    drone = SARSADrone(bw_map, package_pos, delivery_pos)

    drone.train(start_pos, episodes=2000)

    # --- RUN POLICY ---
    curr_state = (start_pos[0], start_pos[1], False)
    path = [curr_state]

    print("Running trained policy...")

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
            print(f"Delivered in {step} steps!")
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
    ani.save("sarsa_fixed.gif", writer="pillow")
    plt.show()
