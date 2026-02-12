# EECS590
This project is a reinforcement learning project for EECS590 (Advanced Topic in Electrical Engineering and Computer Science). The project is designed to train autonomous agents (drones/vehicles) to navigate through an environment. 

To run the project first clone the repository by running the following command:
```
git clone https://github.com/cadengraf/EECS590.git
cd EECS590
```
Install the required packages by running the following command:
```
pip install -r requirements.txt
```

Then cd into the project directory and run the markov program by running the following command:

```
cd drone_rl/
python delivery_markov.py
```

To run the dynamic progamming algorithm run: 

```
python delivery_dynamic.py
```

The agent_framework.py file is still under development as of right now it is just training an agent that goes to the waypoint. But, it can't pick up a package and deliver it to the waypoint.


## Future work to be done: 
1. Physics for the drone/car whatever I decide on because right now I have lanes which could still be useful for the drone. 
2. Add in package pick up so that it picks up a package and then delivers to the waypoint. 
3. Add in bootstrap learning. you will need your agent to eventually bootstrap its own understanding of the reward function and stochastic transitions (also called model-based RL or world-model learning). You will eventually need to give agents a way to store and update prior beliefs of what the foundation environment is as an MDP.
4. Multi-agent learning.
5. Value Iteration, Monte Carlo methods, forward view TD(n) via bootstrapped returns/values + greedy improvements on expected values, backward view TD(lambda) via Eligibility Traces + greedy improvements on expected values. Forward view Sarsa(n) via bootstrapped returns/Q-values + greedy improvements on Q-values,  backward view Sarsa(lambda) via Eligibility Traces + greedy improvements on Q-values, Exploration injection, Q-learning

