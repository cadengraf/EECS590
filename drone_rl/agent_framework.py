import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
from utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions

class DPSolver:
    def __init__(self, bw_map, goal, gamma=0.95):
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # N, S, W, E
        self.states = [tuple(p) for p in np.argwhere(bw_map == 1)]
        self.goal = goal
        self.gamma = gamma
        self.V = {s: 0.0 for s in self.states}
        self.pi = {s: random.choice(range(len(self.actions))) for s in self.states}

    def solve(self, theta=1e-3):
        for i in range(100):
            # 1. Evaluation
            while True:
                delta = 0
                for s in self.states:
                    if s == self.goal: continue
                    old_v = self.V[s]
                    dr, dc = self.actions[self.pi[s]]
                    ns = (s[0] + dr, s[1] + dc)
                    ns = ns if ns in self.V else s
                    reward = 0.0 if ns == self.goal else -1.0
                    self.V[s] = reward + self.gamma * self.V[ns]
                    delta = max(delta, abs(old_v - self.V[s]))
                if delta < theta: break
            
            # 2. Improvement
            stable = True
            for s in self.states:
                if s == self.goal: continue
                old_a = self.pi[s]
                q_vals = []
                for dr, dc in self.actions:
                    ns = (s[0] + dr, s[1] + dc)
                    ns = ns if ns in self.V else s
                    reward = 0.0 if ns == self.goal else -1.0
                    q_vals.append(reward + self.gamma * self.V[ns])
                self.pi[s] = np.argmax(q_vals)
                if old_a != self.pi[s]: stable = False
            
            if stable: break
        return self.V, self.pi

class DroneAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.v_kernel = {}
        self.pi_kernel = {}

    def save(self):
        with open(f"{self.agent_id}.pkl", "wb") as f:
            pickle.dump({"V": self.v_kernel, "Pi": self.pi_kernel}, f)

    def load(self):
        if os.path.exists(f"{self.agent_id}.pkl"):
            with open(f"{self.agent_id}.pkl", "rb") as f:
                data = pickle.load(f)
                self.v_kernel, self.pi_kernel = data["V"], data["Pi"]
            return True
        return False

def run_task(mode, agent, bw_map, hparams):
    if mode == "TRAIN":
        solver = DPSolver(bw_map, hparams['goal'], hparams['gamma'])
        agent.v_kernel, agent.pi_kernel = solver.solve(hparams['theta'])
        agent.save()
        print(f"Training Complete for {agent.agent_id}")
    
    elif mode == "EVAL":
        if not agent.load():
            print("No brain found. Please train first.")
            return []
        
        curr = hparams['start']
        path = [curr]
        for _ in range(hparams['max_steps']):
            if curr == hparams['goal']: break
            a_idx = agent.pi_kernel.get(curr, None)
            if a_idx is None: break
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][a_idx]
            next_s = (curr[0] + dr, curr[1] + dc)
            curr = next_s if next_s in agent.pi_kernel else curr
            path.append(curr)
        return path

if __name__ == "__main__":
    # Setup Map
    pg = PipeGrid(8, 8)
    vis = PipeVisualizerBW(lanes=2)
    bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))

    pipe_ids = pg.to_pipe_ids(PipeOptions())


    lane_pixels = [tuple(p) for p in np.argwhere(bw_map == 1)]
    
    # Setup Parameters
    hyperparams = {
        'start': lane_pixels[0],
        'goal': lane_pixels[-1],
        'gamma': 0.99,
        'theta': 1e-4,
        'max_steps': 500
    }
    
    # Initialize Agent
    drone = DroneAgent("Beta_Drone")

    # EXECUTION: Choose "TRAIN" or "EVAL"
    # To test loading the kernel, run TRAIN once, then switch to EVAL
    task_mode = "TRAIN" 
    
    if task_mode == "TRAIN":
        run_task("TRAIN", drone, bw_map, hyperparams)
        task_mode = "EVAL" # Automatically switch to eval to show result
    
    trajectory = run_task("EVAL", drone, bw_map, hyperparams)

    # Visualization
    if trajectory:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(bw_map, cmap="gray")
        ax.plot(hyperparams['goal'][1], hyperparams['goal'][0], 'gx', markersize=12)
        agent_marker, = ax.plot([], [], 'ro', markersize=6)

        def update(i):
            agent_marker.set_data([trajectory[i][1]], [trajectory[i][0]])
            return agent_marker,

        ani = animation.FuncAnimation(fig, update, frames=len(trajectory), blit=True, interval=30)
        plt.title(f"Agent: {drone.agent_id} | Path Length: {len(trajectory)}")
        ani.save("trained_agent.gif", writer="pillow")
        plt.show()


