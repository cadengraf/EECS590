import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions
from utils.replay import ReplayBuffer
from utils.saliency import run_saliency_suite

class SARSADrone:
    def __init__(self, bw_map, package_pos, delivery_pos,
                 alpha=0.1, gamma=0.95, epsilon=1.0, replay_capacity=10000):

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
        self.replay_buffer = ReplayBuffer(replay_capacity)

    def get_Q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.actions))
        return self.Q[state]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.actions))
        return np.argmax(self.get_Q(state))

    # Step funct 
    def step_env(self, state, action_idx, prev_state, visit_counts):
        r, c, has_pkg = state
        dr, dc = self.actions[action_idx]

        nr, nc = r + dr, c + dc
        invalid_move = False

        if (nr, nc) not in self.lane_coords:
            nr, nc = r, c
            invalid_move = True

        new_has_pkg = has_pkg
        reward = -1.0  # base step cost

        # INVALID MOVE PENALTY
        if invalid_move:
            reward -= 5.0

        # VISIT PENALTY (anti-loop)
        visits = visit_counts.get((nr, nc), 0)
        reward -= 0.5 * visits

        # BACKTRACK PENALTY
        if prev_state is not None and (nr, nc) == (prev_state[0], prev_state[1]):
            reward -= 3.0

        # PICKUP
        if not has_pkg and (nr, nc) == self.package_pos:
            new_has_pkg = True
            reward += 100

        # DELIVERY
        elif has_pkg and (nr, nc) == self.delivery_pos:
            reward += 200

        return (nr, nc, new_has_pkg), reward

    # --- FIXED N-STEP SARSA ---
    def train_n_step(self, start_pos, n=5, episodes=1000, max_steps=500):
        print(f"Training {n}-step SARSA...")

        for ep in range(episodes):
            states = []
            actions = []
            rewards = [0]

            visit_counts = {}

            state = (start_pos[0], start_pos[1], False)
            action = self.choose_action(state)

            states.append(state)
            actions.append(action)

            T = float('inf')
            t = 0
            prev_state = None

            while True:
                if t < T:
                    # Track visits
                    visit_counts[(state[0], state[1])] = visit_counts.get((state[0], state[1]), 0) + 1

                    next_state, reward = self.step_env(state, action, prev_state, visit_counts)
                    rewards.append(reward)
                    done = next_state[2] and (next_state[0], next_state[1]) == self.delivery_pos

                    if done:
                        T = t + 1
                    else:
                        next_action = self.choose_action(next_state)
                        states.append(next_state)
                        actions.append(next_action)

                    self.replay_buffer.add(state, action, reward, next_state, done)

                    prev_state = state
                    state = next_state
                    action = next_action if t + 1 < T else None

                tau = t - n + 1

                if tau >= 0:
                    G = 0.0

                    for i in range(tau + 1, min(tau + n, T) + 1):
                        G += (self.gamma ** (i - tau - 1)) * rewards[i]

                    if tau + n < T:
                        G += (self.gamma ** n) * self.get_Q(states[tau + n])[actions[tau + n]]

                    Q_tau = self.get_Q(states[tau])
                    Q_tau[actions[tau]] += self.alpha * (G - Q_tau[actions[tau]])

                if tau == T - 1:
                    break

                t += 1

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if ep % 50 == 0:
                print(
                    f"Episode {ep}, epsilon={self.epsilon:.3f}, "
                    f"replay_size={len(self.replay_buffer)}"
                )

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

    drone.train_n_step(start_pos, n=5, episodes=2000)

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
    ani.save("sarsa_nstep.gif", writer="pillow")
    plt.show()
