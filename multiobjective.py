import random
import numpy as np

class MODQNTrainer:

    def __init__(self, action_size):
        self.memory = []
        self.q_values_dqn1 = []
        self.q_values_dqn2 = []
        self.action_size = action_size

    def get_action(self, q_values, eps=0.0):
        # Epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(q_values.cpu().data.numpy())
            return action
        else:
            return random.choice(np.arange(self.action_size))
