import random
from collections import namedtuple, deque
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PERMemory(object):
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, *args):
        """Save a transition with priority."""
        max_priority = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
            self.priorities.append(max_priority)
        else:
            self.memory[self.position] = Transition(*args)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of transitions based on priorities."""
        probabilities = np.array(self.priorities) ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        # Calculate importance-sampling weights
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, errors, offset=0.01):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + offset) ** self.alpha

    def __len__(self):
        return len(self.memory)