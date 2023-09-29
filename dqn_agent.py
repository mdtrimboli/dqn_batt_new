"""
DQN Agent for Vector Observation Learning

Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code expanded and adapted from code examples provided by Udacity DRL Team, 2018.
"""

# Import Required Packages
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from replay_buffer import ReplayBuffer

# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
##################################################
Agent Class
Defines DQN Agent Methods
Agent interacts with and learns from an environment.
"""


class Agent():
    """
    Initialize Agent, inclduing:
        DQN Hyperparameters
        Local and Targat State-Action Policy Networks
        Replay Memory Buffer from Replay Buffer Class (define below)
    """

    def __init__(self, state_size, action_size, dqn_type='DQN', replay_memory_size=5e6, batch_size=256, gamma=0.995,
                 learning_rate=1e-3, target_tau=2e-3, update_rate=8, seed=0):

        """
        DQN Agent Parameters
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            dqn_type (string): can be either 'DQN' for vanillia dqn learning (default) or 'DDQN' for double-DQN.
            replay_memory size (int): size of the replay memory buffer (typically 5e4 to 5e6)
            batch_size (int): size of the memory batch used for model updates (typically 32, 64 or 128)
            gamma (float): paramete for setting the discoun ted value of future rewards (typically .95 to .995)
            learning_rate (float): specifies the rate of model learing (typically 1e-4 to 1e-3))
            seed (int): random seed for initializing training point.
        """
        self.dqn_type = dqn_type
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        #self.seed = random.seed(seed)

        """
        # DQN Agent Q-Network
        # For DQN training, two nerual network models are employed;
        # (a) A network that is updated every (step % update_rate == 0)
        # (b) A target network, with weights updated to equal the network at a slower (target_tau) rate.
        # The slower modulation of the target network weights operates to stablize learning.
        """
        self.network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    ########################################################
    # STEP() method
    #
    def step(self, state, action, dv, reward, dv_reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, dv, reward, dv_reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    ########################################################
    # ACT() method
    #
    def get_values(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = torch.from_numpy(state).float().squeeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values, value_outs, dv_values = self.network(state)

        self.network.train()
        return action_values, value_outs, dv_values


    ########################################################
    # LEARN() method
    # Update value parameters using given batch of experience tuples.
    def learn(self, experiences, gamma, DQN=True):

        """
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, d_values, rewards, dv_rewards, next_states, dones = experiences

        # Get Q values from current observations (s, a) using model nextwork
        output_tuple = self.network(states)

        Qsa = output_tuple[0].gather(1, actions)
        s_values = output_tuple[1]
        dvs = output_tuple[2]

        #Qsa, dvs, sdvs = self.network(states).gather(1, actions)

        if (self.dqn_type == 'DDQN'):
            # Double DQN
            # ************************
            output_tuple = self.network(states)

            Qsa_prime_actions = output_tuple[0].detach().max(1)[1]


            Qsa_prime_targets, dv_targets_values, sdv_targets = self.target_network(next_states)
            # Ver si esto está bien
            Qsa_prime_targets = Qsa_prime_targets[torch.arange(64), Qsa_prime_actions].unsqueeze(-1)
            #Qsa_prime_targets = Qsa_prime_targets[Qsa_prime_actions].unsqueeze(1)
            #Qsa_prime_targets = self.target_network(next_states)[Qsa_prime_actions].unsqueeze(1)

        else:
            # Regular (Vanilla) DQN
            # ************************
            # Get max Q values for (s',a') from target model

            Qsa_prime_target_values, dv_targets_values, sdv_targets = self.target_network(next_states)
            Qsa_prime_target_values = Qsa_prime_target_values.detach()
            dv_targets_values = dv_targets_values.detach()

            Qsa_prime_targets = Qsa_prime_target_values.max(1)[0].unsqueeze(1)


            # Compute Q targets for current states

        Qsa_targets = rewards + (gamma * Qsa_prime_targets * (1 - dones))
        dv_targets = dv_rewards + (gamma * dv_targets_values * (1 - dones))

        #sdv_scale_error1 = (0.5 - torch.mean(dvs)).to(self.device)
        #sdv_scale_error2 = (1.0 - (torch.max(dvs) - torch.min(dvs))).to(self.device)

        sdv_scale_error1 = torch.mean(dvs) - 1.
        sdv_scale_error2 = torch.sqrt(torch.mean((dvs - torch.mean(dvs))**2))


        # Compute loss (error)

        loss_qv = F.huber_loss(Qsa, Qsa_targets)
        loss_dv = F.huber_loss(s_values, dv_targets)
        loss_sdv = torch.sum(torch.pow(sdv_scale_error1, 2)) + torch.sum(torch.pow(sdv_scale_error2, 2))

        loss = loss_qv + loss_dv + loss_sdv

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_network, self.tau)

    ########################################################
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """

    def soft_update(self, local_model, target_model, tau):
        """
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)