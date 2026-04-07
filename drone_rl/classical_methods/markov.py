'''
This is a simple implementation for a finite markov that converges to the final destination without any drift or physics. 
This is the first part of the project so I will have a lot to add to this but it is a good start for the first part... 
Things that will need to be added...
1. Physics for the drone/car whatever I decide on because right now I have lanes which could still be useful for the drone. 
2. Add in package pick up so that it picks up a package and then delivers to the waypoint. 
3. Add in bootstrap learning. you will need your agent to eventually bootstrap its own understanding of the reward function and stochastic transitions (also called model-based RL or world-model learning). You will eventually need to give agents a way to store and update prior beliefs of what the foundation environment is as an MDP.
4. Multi-agent learning.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.pipes import PipeVisualizerBW, PipeGrid, PipeOptions
from abc import ABC, abstractmethod
from typing import Dict, Hashable, Iterable

# --- MDP BASE CLASSES ---
State = Hashable
TransitionTable = Dict[State, Dict[State, float]]
RewardTable = Dict[State, Dict[State, float]]

class FiniteMarkovProcess(ABC):
    def __init__(self, states: Iterable[State], transitions: TransitionTable):
        self.states = set(states)
        self.transitions = transitions
    
    def next_steps(self, s: State) -> Dict[State, float]:
        return self.transitions.get(s, {})

    @abstractmethod
    def step(self, s: State, s_next: State): pass

class RewardMarkovProcess(FiniteMarkovProcess):
    def __init__(self, states: Iterable[State], transitions: TransitionTable, rewards: RewardTable):
        self.rewards = rewards
        super().__init__(states, transitions)

    def step(self, s: State, s_next: State):
        return self.rewards.get(s, {}).get(s_next, 0.0)

# --- GRIDWORLD NAVIGATION ---
class LaneNavigator(RewardMarkovProcess):
    def __init__(self, bw_map, start_pos, goal_pos, gamma=0.98):
        self.mask = self._create_mdp_mask(bw_map, goal_pos)
        self.gamma = gamma
        self.rows, self.cols = self.mask.shape
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        
        states_list = [(r, c) for r in range(self.rows) for c in range(self.cols) 
                       if not np.isneginf(self.mask[r, c])]
        
        policy_transitions, rewards_table = self.solve_mdp(states_list)
        super().__init__(states_list, policy_transitions, rewards_table)

    def _create_mdp_mask(self, bw, goal):
        # 1 = Lane/Path, 0 = Wall
        mask = np.full(bw.shape, -np.inf)
        mask[bw == 1] = -1.0 # Step cost for lanes
        mask[goal] = 0.0     # Goal reward
        return mask

    def solve_mdp(self, states):
        V = {s: 0.0 for s in states}
        policy = {}
        # Value Iteration
        for _ in range(150): # Increased iterations for larger pixel grid
            new_V = V.copy()
            for s in states:
                if self.mask[s] == 0: continue 
                q_values = []
                for dr, dc in self.actions:
                    ns = (s[0] + dr, s[1] + dc)
                    if ns not in V: ns = s
                    q_values.append(self.mask[ns] + self.gamma * V[ns])
                best_idx = np.argmax(q_values)
                new_V[s] = q_values[best_idx]
                policy[s] = best_idx
            V = new_V

        trans_table, rew_table = {}, {}
        for s in states:
            if self.mask[s] == 0:
                trans_table[s] = {s: 1.0}; rew_table[s] = {s: 0.0}
            else:
                a = self.actions[policy[s]]
                ns = (s[0] + a[0], s[1] + a[1])
                if ns not in V: ns = s
                trans_table[s] = {ns: 1.0}
                rew_table[s] = {ns: float(self.mask[ns])}
        return trans_table, rew_table

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Generate the Pipe World
    grid_size = (10, 10) # Smaller for faster MDP solving
    pg = PipeGrid(*grid_size)
    pipe_ids = pg.to_pipe_ids(PipeOptions())
    
    vis = PipeVisualizerBW(lanes=2, base=3)
    bw_map = vis.render(pipe_ids)

    # 2. Define Start and Waypoint (Goal)
    lane_coords = np.argwhere(bw_map == 1)
    start_pos = tuple(lane_coords[0])     # Top-left-ish

    # Random goal position 
    idx = np.random.randint(0, len(lane_coords))
    goal_pos = tuple(lane_coords[idx])

    print(f"Solving MDP for {bw_map.shape} grid...")
    navigator = LaneNavigator(bw_map, start_pos, goal_pos)

    # 3. Generate Trajectory
    traj = [start_pos]
    curr = start_pos
    for _ in range(500):
        if curr == goal_pos: break
        next_steps = navigator.next_steps(curr)
        if not next_steps: break
        curr = list(next_steps.keys())[0]
        traj.append(curr)

    # 4. Animation
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bw_map, cmap="gray")
    ax.plot(goal_pos[1], goal_pos[0], 'gx', markersize=15, label="Waypoint") # Goal
    
    agent, = ax.plot([], [], "ro", markersize=8, label="Vehicle")
    
    def update(frame):
        r, c = traj[frame]
        agent.set_data([c], [r])
        return (agent,)

    ani = animation.FuncAnimation(fig, update, frames=len(traj), blit=True, interval=50)
    plt.legend()
    plt.title("Drone Navigation in Lane MDP")
    ani.save("drone_path.gif", writer="pillow")
    print("Mission complete. Animation saved as drone_path.gif")
    # plt.show()