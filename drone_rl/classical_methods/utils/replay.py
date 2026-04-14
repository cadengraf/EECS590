import random

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        assert capacity > 0, "Replay buffer capacity must be positive."
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        assert batch_size > 0, "batch_size must be positive."
        assert batch_size <= len(self.buffer), "batch_size cannot exceed the current replay buffer size."

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states),
            np.asarray(actions),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def sample_raw(self, batch_size):
        assert batch_size > 0, "batch_size must be positive."
        assert batch_size <= len(self.buffer), "batch_size cannot exceed the current replay buffer size."
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()
        self.position = 0

    def __len__(self):
        return len(self.buffer)
