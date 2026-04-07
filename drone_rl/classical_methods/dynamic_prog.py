'''
Here is a really simple implementation of dynamic programming using policy iteration and policy improvement.
'''

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
        self.Q = {s: np.zeros(len(self.actions)) for s in self.states}
        self.policy = {s: random.choice(range(len(self.actions))) for s in self.states}

    def get_next_state(self, s, a_idx):
        dr, dc = self.actions[a_idx]
        ns = (s[0] + dr, s[1] + dc)
        return ns if ns in self.V else s

    def evaluate_policy(self, theta=1e-3):
        """Policy Evaluation: Stability check on V values"""
        while True:
            delta = 0
            for s in self.states:
                if s == self.goal: continue
                old_v = self.V[s]
                ns = self.get_next_state(s, self.policy[s])
                # Reward: 0 at goal, -1 elsewhere
                reward = 0.0 if ns == self.goal else -1.0
                self.V[s] = reward + self.gamma * self.V[ns]
                delta = max(delta, abs(old_v - self.V[s]))
            if delta < theta: break

    def improve_policy(self):
        """Policy Improvement: Update Q-values and extract best actions"""
        stable = True
        for s in self.states:
            if s == self.goal: continue
            old_action = self.policy[s]
            
            # Compute Q(s, a) for all actions
            for a_idx in range(len(self.actions)):
                ns = self.get_next_state(s, a_idx)
                reward = 0.0 if ns == self.goal else -1.0
                self.Q[s][a_idx] = reward + self.gamma * self.V[ns]
            
            self.policy[s] = np.argmax(self.Q[s])
            if old_action != self.policy[s]: stable = False
        return stable

    def solve(self):
        print("Solving with Policy Iteration...")
        for i in range(100):
            self.evaluate_policy()
            if self.improve_policy():
                print(f"Converged at iteration {i}")
                break

# ---------------- MAIN EXECUTION ----------------

def main():
    # 1. Setup Environment
    grid_size = (8, 8) 
    pipe_opts = PipeOptions()
    pg = PipeGrid(*grid_size)
    grid_ids = pg.to_pipe_ids(pipe_opts)
    
    vis = PipeVisualizerBW(lanes=2, base=3)
    bw_map = vis.render(grid_ids)
    
    # 2. Define Waypoints
    lane_pixels = [tuple(p) for p in np.argwhere(bw_map == 1)]
    start_pos = lane_pixels[0]

    # Random goal position 
    idx = np.random.randint(0, len(lane_pixels))
    goal_pos = tuple(lane_pixels[idx])

    # 3. Solve MDP using DP
    solver = DPPolicyIteration(bw_map, goal_pos)
    solver.solve()

    # 4. Generate Path from Policy
    path = [start_pos]
    curr = start_pos
    for _ in range(400):
        if curr == goal_pos: break
        curr = solver.get_next_state(curr, solver.policy[curr])
        path.append(curr)

    # 5. Animate
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bw_map, cmap="gray", origin='upper')
    ax.plot(goal_pos[1], goal_pos[0], 'gx', markersize=12, label="Waypoint")
    
    agent, = ax.plot([], [], "ro", markersize=6, label="Vehicle")
    
    def update(frame):
        r, c = path[frame]
        agent.set_data([c], [r])
        return (agent,)

    ani = animation.FuncAnimation(fig, update, frames=len(path), blit=True, interval=30)
    plt.title("DP Policy Iteration: Lane Navigation")
    plt.legend()
    ani.save("lane_navigation.gif", writer="pillow")
    print("Animation saved as lane_navigation.gif")
    plt.show()

if __name__ == "__main__":
    main()