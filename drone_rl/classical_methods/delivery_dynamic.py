import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions

class DPPolicyIteration:
    def __init__(self, bw_map, goal, gamma=0.95):
        self.bw = bw_map
        self.goal = goal
        self.gamma = gamma
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # N, S, W, E
        self.states = [tuple(p) for p in np.argwhere(bw_map == 1)]
        
        # Initialize Values and Policy
        self.V = {s: 0.0 for s in self.states}
        self.policy = {s: random.choice(range(len(self.actions))) for s in self.states}

    def get_next_state(self, s, a_idx):
        dr, dc = self.actions[a_idx]
        ns = (s[0] + dr, s[1] + dc)
        return ns if ns in self.V else s

    def solve(self, theta=1e-4):
        """Standard Policy Iteration: Evaluation + Improvement"""
        for i in range(100):
            # 1. Policy Evaluation
            while True:
                delta = 0
                for s in self.states:
                    if s == self.goal: continue
                    old_v = self.V[s]
                    ns = self.get_next_state(s, self.policy[s])
                    reward = 0.0 if ns == self.goal else -1.0
                    self.V[s] = reward + self.gamma * self.V[ns]
                    delta = max(delta, abs(old_v - self.V[s]))
                if delta < theta: break

            # 2. Policy Improvement
            stable = True
            for s in self.states:
                if s == self.goal: continue
                old_action = self.policy[s]
                
                q_values = []
                for a_idx in range(len(self.actions)):
                    ns = self.get_next_state(s, a_idx)
                    reward = 0.0 if ns == self.goal else -1.0
                    q_values.append(reward + self.gamma * self.V[ns])
                
                self.policy[s] = np.argmax(q_values)
                if old_action != self.policy[s]: stable = False
            
            if stable:
                print(f"  Converged for goal {self.goal} in {i} iterations.")
                break

# ---------------- MAIN EXECUTION ----------------

def main():
    # 1. Setup Environment
    grid_size = (10, 10) 
    pg = PipeGrid(*grid_size)
    vis = PipeVisualizerBW(lanes=2, base=3)
    bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))
    
    lane_pixels = [tuple(p) for p in np.argwhere(bw_map == 1)]
    start_pos = lane_pixels[0]
    package_pos = lane_pixels[len(lane_pixels)//2]
    delivery_pos = lane_pixels[-1]

    # 2. Solve DP for both phases
    print("Solving Fetch Policy...")
    fetch_solver = DPPolicyIteration(bw_map, package_pos)
    fetch_solver.solve()

    print("Solving Delivery Policy...")
    deliver_solver = DPPolicyIteration(bw_map, delivery_pos)
    deliver_solver.solve()

    # 3. Simulate Mission
    path = []
    curr = start_pos
    has_package = False
    
    for _ in range(500):
        path.append((curr, has_package))
        
        if not has_package:
            if curr == package_pos:
                has_package = True
                print("Package Picked Up!")
                continue # Stay at pos for one frame to show pickup
            # Use fetch policy
            action_idx = fetch_solver.policy[curr]
            curr = fetch_solver.get_next_state(curr, action_idx)
        else:
            if curr == delivery_pos:
                print("Delivery Complete!")
                break
            # Use delivery policy
            action_idx = deliver_solver.policy[curr]
            curr = deliver_solver.get_next_state(curr, action_idx)

    # 4. Animate with State-Aware Logic
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bw_map, cmap="gray", origin='upper')
    
    package_marker, = ax.plot(package_pos[1], package_pos[0], 'ys', markersize=10, label="Package")
    ax.plot(delivery_pos[1], delivery_pos[0], 'gx', markersize=12, label="Drop Zone")
    agent_marker, = ax.plot([], [], "ro", markersize=8, label="Drone")
    
    title = ax.set_title("Status: Fetching Package")

    def update(frame):
        (r, c), carrying = path[frame]
        agent_marker.set_data([c], [r])
        
        if carrying:
            agent_marker.set_color("orange")
            package_marker.set_visible(False)
            title.set_text("Status: Delivering Package")
        else:
            agent_marker.set_color("red")
            package_marker.set_visible(True)
            
        return agent_marker, package_marker, title

    ani = animation.FuncAnimation(fig, update, frames=len(path), blit=False, interval=40)
    plt.legend()
    ani.save("dp_delivery.gif", writer="pillow")
    plt.show()

if __name__ == "__main__":
    main()