import random
import torch
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

    def scaled_q_values(self, q_values):
        min_value = q_values.min()
        max_value = q_values.max()

        # Escalar el tensor
        scaled_tensor = (q_values - min_value) / (max_value - min_value)
        return scaled_tensor

    def sum_weighted_q_values(self, dictionary, num_objectives, priorities):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        q_values = 0

        for i in range(num_objectives):
            clave = f'dqn_{i+1}'
            dictionary[clave]['q_values'] = dictionary[clave]['q_values'] * priorities[i] * dictionary[clave]['d_values']


        for i in range(num_objectives):
            clave = f'dqn_{i+1}'
            adder = dictionary[clave]['q_values']
            q_values += adder

        #mu = (torch.rand(len(q_values)) * 0.0001).to(device)
        #q_values = q_values + mu

        return q_values
