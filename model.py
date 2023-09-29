"""
Example Neural Network Model for Vector Observation DQN Agent
DQN Model for Unity ML-Agents Environments using PyTorch

Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code expanded and adapted from code examples provided by Udacity DRL Team, 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    #################################################
    Initialize neural network model
    Initialize parameters and build model.
    """

    def __init__(self, state_size, action_size, seed, num_dvs=1, fc1_units=128, fc2_units=128):
        """
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.fc_action_dv = nn.Linear(action_size, num_dvs)

        # Parámetros para el escalado de las variables de estado
        #self.dv_scale_mean = nn.Parameter(torch.Tensor([1.0]))
        self.dv_scale_mean = nn.Parameter(torch.Tensor([1.]))
        self.dv_scale_std = nn.Parameter(torch.Tensor([0.0]))

    """
    ###################################################
    Build a network that maps state -> action values.
    """

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        value_outs = self.fc_action_dv(q_values)

        dv_outs = torch.sigmoid((value_outs + self.dv_scale_std) * self.dv_scale_mean)

        return q_values, value_outs, dv_outs
