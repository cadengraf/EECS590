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

# --- DELIVERY DRONE (UNCHANGED CORE LOGIC) ---
class DeliveryDrone(RewardMarkovProcess):
    def __init__(self, bw_map, start_pos, package_pos, delivery_pos, gamma=0.95):
        self.bw_map = bw_map
        self.gamma = gamma
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.lane_coords = [tuple(p) for p in np.argwhere(bw_map == 1)]
        
        self.package_pos = package_pos
        self.delivery_pos = delivery_pos

        print("Solving Phase 1: Fetch Policy...")
        self.policy_fetch = self.solve_layer(self.lane_coords, goal=package_pos)
        
        print("Solving Phase 2: Delivery Policy...")
        self.policy_deliver = self.solve_layer(self.lane_coords, goal=delivery_pos)

    def solve_layer(self, valid_pixels, goal, theta=1e-4):
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
                    if ns not in V:
                        ns = s
                    
                    reward = 0.0 if ns == goal else -1.0
                    q_values.append(reward + self.gamma * V[ns])
                
                best_val = max(q_values)
                delta = max(delta, abs(V[s] - best_val))
                new_V[s] = best_val
                policy[s] = np.argmax(q_values)
            
            V = new_V
            iteration += 1
            if delta < theta or iteration > 2000:
                break
            
        print(f"  Converged in {iteration} iterations.")
        return policy

    def get_action(self, state):
        r, c, has_package = state
        curr_pos = (r, c)
        
        if not has_package:
            if curr_pos == self.package_pos:
                return "PICKUP"
            a_idx = self.policy_fetch.get(curr_pos)
        else:
            if curr_pos == self.delivery_pos:
                return "DONE"
            a_idx = self.policy_deliver.get(curr_pos)

        return self.actions[a_idx] if a_idx is not None else None

# --- MULTI-DRONE CONTROLLER ---
class MultiDroneController:
    def __init__(self, drone_model, start_positions):
        self.drone_model = drone_model
        self.drones = [
            {"pos": pos, "has_package": False, "id": i}
            for i, pos in enumerate(start_positions)
        ]
        self.package_taken = False

    def step(self):
        new_positions = set()

        for drone in self.drones:
            r, c = drone["pos"]
            has_pkg = drone["has_package"]

            state = (r, c, has_pkg)
            action = self.drone_model.get_action(state)

            # --- PICKUP ---
            if action == "PICKUP" and not self.package_taken:
                drone["has_package"] = True
                self.package_taken = True
                print(f"Drone {drone['id']} picked up package!")
                new_positions.add((r, c))
                continue

            # --- DELIVERY ---
            if action == "DONE" and has_pkg:
                print(f"Drone {drone['id']} delivered package!")
                return True

            # --- MOVEMENT ---
            if isinstance(action, tuple):
                nr, nc = r + action[0], c + action[1]

                if (nr, nc) not in new_positions:
                    drone["pos"] = (nr, nc)
                    new_positions.add((nr, nc))
                else:
                    # Collision: stay put
                    new_positions.add((r, c))

        return False

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Environment Setup
    grid_size = (12, 12)
    pg = PipeGrid(*grid_size)
    vis = PipeVisualizerBW(lanes=2, base=3)
    bw_map = vis.render(pg.to_pipe_ids(PipeOptions()))

    lane_coords = [tuple(p) for p in np.argwhere(bw_map == 1)]

    start_positions = lane_coords[:3]  # MULTIPLE DRONES
    package_pos = lane_coords[len(lane_coords)//2]
    delivery_pos = lane_coords[-1]

    # 2. Solver
    drone_model = DeliveryDrone(bw_map, start_positions[0], package_pos, delivery_pos)

    # 3. Multi-drone controller
    controller = MultiDroneController(drone_model, start_positions)

    # 4. Simulation
    path = []

    print("Beginning Multi-Drone Mission...")
    for step in range(1000):
        done = controller.step()

        snapshot = [
            (d["pos"][0], d["pos"][1], d["has_package"])
            for d in controller.drones
        ]
        path.append(snapshot)

        if done:
            print(f"Mission complete in {step} steps!")
            break

    # 5. Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bw_map, cmap="gray")

    ax.plot(package_pos[1], package_pos[0], 'ys', markersize=10, label="Package")
    ax.plot(delivery_pos[1], delivery_pos[0], 'gx', markersize=12, label="Goal")

    drone_marks = [
        ax.plot([], [], 'o', markersize=6)[0]
        for _ in controller.drones
    ]

    def update(frame):
        states = path[frame]

        for i, (r, c, has_pkg) in enumerate(states):
            drone_marks[i].set_data([c], [r])
            drone_marks[i].set_color('orange' if has_pkg else 'red')

        return drone_marks

    ani = animation.FuncAnimation(
        fig, update, frames=len(path), interval=60, blit=False
    )

    plt.legend(loc='lower right')
    ani.save("multi_drone_delivery.gif", writer="pillow")
    plt.show()