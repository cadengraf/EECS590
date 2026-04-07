import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions

class DeliveryDroneTDN:
    def __init__(self, bw_map, start_pos, package_pos, delivery_pos,
                 alpha=0.1, gamma=0.99, epsilon=1.0, n=5):
        self.bw_map = bw_map
        self.start_pos = start_pos
        self.package_pos = package_pos
        self.delivery_pos = delivery_pos

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n

        self.actions = [(-1,0), (1,0), (0,-1), (0,1)]
        self.lane_coords = [tuple(p) for p in np.argwhere(bw_map == 1)]

        # Optimistic initialization to encourage exploration
        self.V = {}
        for s in self.lane_coords:
            self.V[(s[0], s[1], False)] = 50.0
            self.V[(s[0], s[1], True)] = 50.0

        print(f"Training with TD({n}) forward-view...")
        self.train(episodes=1500)

    def step(self, state, action):
        r, c, has_pkg = state
        dr, dc = action
        nr, nc = r + dr, c + dc
        if (nr, nc) not in self.lane_coords:
            nr, nc = r, c
        if not has_pkg and (nr, nc) == self.package_pos:
            has_pkg = True
        return (nr, nc, has_pkg)

    def reward(self, state, next_state):
        r, c, has_pkg = state
        nr, nc, n_has_pkg = next_state
        if not has_pkg and n_has_pkg:
            return 200
        if n_has_pkg and (nr, nc) == self.delivery_pos:
            return 500
        target = self.delivery_pos if n_has_pkg else self.package_pos
        d_curr = abs(r - target[0]) + abs(c - target[1])
        d_next = abs(nr - target[0]) + abs(nc - target[1])
        if d_next < d_curr: return 20
        if d_next > d_curr: return -5
        return -1

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return self.actions[np.random.randint(4)]
        values = []
        for a in self.actions:
            ns = self.step(state, a)
            values.append(self.V.get(ns, 0))
        return self.actions[np.argmax(values)]

    def train(self, episodes=1500):
        max_steps = 400
        decay_rate = 0.995

        for ep in range(episodes):
            self.epsilon = max(0.05, self.epsilon * decay_rate)
            state = (self.start_pos[0], self.start_pos[1], False)
            states = [state]
            rewards = [0]
            T = float('inf')
            t = 0

            while True:
                if t < T:
                    action = self.epsilon_greedy(state)
                    next_state = self.step(state, action)
                    r = self.reward(state, next_state)
                    states.append(next_state)
                    rewards.append(r)
                    if next_state[2] and (next_state[0], next_state[1]) == self.delivery_pos:
                        T = t + 1
                    state = next_state

                tau = t - self.n + 1
                if tau >= 0:
                    # compute n-step return
                    upper = int(min(tau + self.n, T))
                    G = sum([self.gamma**(i - tau - 1) * rewards[i] for i in range(tau + 1, upper + 1)])
                    if tau + self.n < T:
                        G += (self.gamma**self.n) * self.V.get(states[tau + self.n], 0)
                    s_tau = states[tau]
                    self.V[s_tau] += self.alpha * (G - self.V[s_tau])
                if tau == T - 1:
                    break
                t += 1
            if ep % 250 == 0:
                print(f"Episode {ep}, epsilon {self.epsilon:.2f}")

    def get_action(self, state):
        if not state[2] and (state[0], state[1]) == self.package_pos:
            return "PICKUP"
        if state[2] and (state[0], state[1]) == self.delivery_pos:
            return "DONE"
        q_vals = [self.V.get(self.step(state,a),-100) for a in self.actions]
        best_actions = [a for a,q in zip(self.actions,q_vals) if q==max(q_vals)]
        return best_actions[np.random.randint(len(best_actions))]

# --- MAIN ---
if __name__ == "__main__":
    grid_size = (12,12)
    pg = PipeGrid(*grid_size)
    vis = PipeVisualizerBW(lanes=2, base=3)
    bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))

    lane_coords = [tuple(p) for p in np.argwhere(bw_map==1)]
    start_pos = lane_coords[0]
    package_pos = lane_coords[len(lane_coords)//2]
    delivery_pos = lane_coords[-1]

    drone = DeliveryDroneTDN(bw_map, start_pos, package_pos, delivery_pos)

    # --- SIMULATION ---
    state = (start_pos[0], start_pos[1], False)
    path = [state]
    for step in range(500):
        action = drone.get_action(state)
        if action=="DONE":
            print(f"Delivered in {step} steps!")
            break
        elif action=="PICKUP":
            state = (state[0], state[1], True)
        else:
            state = drone.step(state, action)
        path.append(state)

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(bw_map, cmap="gray")
    p_mark, = ax.plot(package_pos[1], package_pos[0],'ys',markersize=10,label="Package")
    d_mark, = ax.plot(delivery_pos[1], delivery_pos[0],'gx',markersize=12,label="Delivery")
    drone_mark, = ax.plot([],[], 'ro', markersize=7,label="Drone")
    status_text = ax.set_title("Status: TD(n) Drone")

    def update(frame):
        r,c,has_pkg = path[frame]
        drone_mark.set_data([c],[r])
        if has_pkg:
            drone_mark.set_color('orange')
            p_mark.set_visible(False)
            status_text.set_text(f"Status: Delivering (Step {frame})")
        else:
            drone_mark.set_color('red')
            p_mark.set_visible(True)
            status_text.set_text(f"Status: Fetching (Step {frame})")
        return drone_mark,p_mark,d_mark,status_text

    ani = animation.FuncAnimation(fig,update,frames=len(path),interval=50,blit=False)
    plt.legend(loc='lower right')
    ani.save("tdn_forward_deal.gif",writer="pillow")
    plt.show()