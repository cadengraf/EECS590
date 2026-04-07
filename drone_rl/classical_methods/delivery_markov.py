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

# --- DELIVERY DRONE NAVIGATOR ---
class DeliveryDrone(RewardMarkovProcess):
    def __init__(self, bw_map, start_pos, package_pos, delivery_pos, gamma=0.95):
        self.bw_map = bw_map
        self.gamma = gamma
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # N, S, W, E
        self.lane_coords = [tuple(p) for p in np.argwhere(bw_map == 1)]
        
        self.package_pos = package_pos
        self.delivery_pos = delivery_pos

        # Solve two distinct MDP layers
        print("Solving Phase 1: Fetch Policy...")
        self.policy_fetch = self.solve_layer(self.lane_coords, goal=package_pos)
        
        print("Solving Phase 2: Delivery Policy...")
        self.policy_deliver = self.solve_layer(self.lane_coords, goal=delivery_pos)

    def solve_layer(self, valid_pixels, goal, theta=1e-4):
        """Value Iteration: Propagates rewards until convergence."""
        V = {s: 0.0 for s in valid_pixels}
        policy = {s: 0 for s in valid_pixels}
        
        iteration = 0
        while True:
            delta = 0
            new_V = V.copy()
            for s in valid_pixels:
                if s == goal: continue
                
                q_values = []
                for dr, dc in self.actions:
                    ns = (s[0] + dr, s[1] + dc)
                    if ns not in V: ns = s # Wall collision logic
                    
                    reward = 0.0 if ns == goal else -1.0
                    q_values.append(reward + self.gamma * V[ns])
                
                best_val = max(q_values)
                delta = max(delta, abs(V[s] - best_val))
                new_V[s] = best_val
                policy[s] = np.argmax(q_values)
            
            V = new_V
            iteration += 1
            if delta < theta: break
            if iteration > 2000: break # Safety exit
            
        print(f"  Converged in {iteration} iterations.")
        return policy

    def get_action(self, state):
        r, c, has_package = state
        curr_pos = (r, c)
        
        if not has_package:
            if curr_pos == self.package_pos: return "PICKUP"
            a_idx = self.policy_fetch.get(curr_pos)
        else:
            if curr_pos == self.delivery_pos: return "DONE"
            a_idx = self.policy_deliver.get(curr_pos)

        return self.actions[a_idx] if a_idx is not None else None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Environment Setup
    grid_size = (12, 12) 
    pg = PipeGrid(*grid_size)
    vis = PipeVisualizerBW(lanes=2, base=3)
    bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))

    lane_coords = [tuple(p) for p in np.argwhere(bw_map == 1)]
    start_pos = lane_coords[0]
    package_pos = lane_coords[len(lane_coords)//2]
    delivery_pos = lane_coords[-1]

    # 2. Solver Initialization
    drone = DeliveryDrone(bw_map, start_pos, package_pos, delivery_pos)

    # 3. Simulation Loop
    # State format: (row, col, has_package_bool)
    curr_state = (start_pos[0], start_pos[1], False)
    path = [curr_state]
    
    print("Beginning Mission...")
    for step in range(1000):
        action = drone.get_action(curr_state)
        r, c, has_pkg = curr_state

        if action == "DONE":
            print(f"Mission Success in {step} steps!")
            break
        elif action == "PICKUP":
            print(f"Package picked up at step {step}!")
            curr_state = (r, c, True)
        elif action is None:
            print(f"Drone stuck at {r, c}")
            break
        else:
            curr_state = (r + action[0], c + action[1], has_pkg)
            
        path.append(curr_state)

    # 4. Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bw_map, cmap="gray")
    
    p_mark, = ax.plot(package_pos[1], package_pos[0], 'ys', markersize=10, label="Package")
    d_mark, = ax.plot(delivery_pos[1], delivery_pos[0], 'gx', markersize=12, label="Goal")
    drone_mark, = ax.plot([], [], 'ro', markersize=7, label="Drone")
    
    status_text = ax.set_title("Status: En Route to Package")

    def update(frame):
        r, c, has_pkg = path[frame]
        drone_mark.set_data([c], [r])
        
        if has_pkg:
            drone_mark.set_color('orange')
            p_mark.set_visible(False)
            status_text.set_text("Status: Delivering Package")
        else:
            drone_mark.set_color('red')
            p_mark.set_visible(True)
            status_text.set_text("Status: Fetching Package")
            
        return drone_mark, p_mark, status_text

    ani = animation.FuncAnimation(fig, update, frames=len(path), blit=False, interval=40)
    plt.legend(loc='lower right')
    ani.save("delivery_complete.gif", writer="pillow")
    plt.show()