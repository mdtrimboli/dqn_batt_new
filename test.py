
"""
Test DQN Model for Unity ML-Agents Environments using PyTorch

This example tests a trained DQN NN model on a modified version of the Unity ML-Agents Banana Collection Example Environment.
The environment includes a single agent, who can turn left or right and move forward or backward.
The agent's task is to collect yellow bananas (reward of +1) that are scattered around an square
game area, while avoiding purple bananas (reward of -1). For the version of Bananas employed here,
the environment is considered solved when the average score over the last 100 episodes > 13.

Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
"""

###################################
# Import Required Packages
import torch
import yaml
import os
import time
import random
import numpy as np
from dqn_agent import Agent
from battery import Battery

import graphics as g

"""
###################################
STEP 1: Set the Test Parameters
======
        num_episodes (int): number of test episodes
"""
num_episodes = 1
temp_buffer = []
current_buffer = []
voltage_buffer = []
soc_buffer = []

with open("default.yml", "r") as f:
    config = yaml.safe_load(f)


"""
###################################
STEP 2: Start the Unity Environment
# Use the corresponding call depending on your operating system 
"""
env = Battery(config)
# - **Mac**: "Banana.app"
# - **Windows** (x86): "Banana_Windows_x86/Banana.exe"
# - **Windows** (x86_64): "Banana_Windows_x86_64/Banana.exe"
# - **Linux** (x86): "Banana_Linux/Banana.x86"
# - **Linux** (x86_64): "Banana_Linux/Banana.x86_64"
# - **Linux** (x86, headless): "Banana_Linux_NoVis/Banana.x86"
# - **Linux** (x86_64, headless): "Banana_Linux_NoVis/Banana.x86_64"

"""
#######################################
STEP 3: Get The Unity Environment Brian
Unity ML-Agent applications or Environments contain "BRAINS" which are responsible for deciding 
the actions an agent or set of agents should take given a current set of environment (state) 
observations. The Banana environment has a single Brian, thus, we just need to access the first brain 
available (i.e., the default brain). We then set the default brain as the brain that will be controlled.
"""

"""
#############################################
STEP 4: Determine the size of the Action and State Spaces
# 
# The simulation contains a single agent that navigates a large environment.  
# At each time step, it can perform four possible actions:
# - `0` - walk forward 
# - `1` - walk backward
# - `2` - turn left
# - `3` - turn right
# 
# The state space has `37` dimensions and contains the agent's velocity, 
# along with ray-based perception of objects around agent's forward direction.  
# A reward of `+1` is provided for collecting a yellow banana, and a reward of 
# `-1` is provided for collecting a purple banana. 
"""

# Set the number of actions or action size
action_size = env.action_size

# Set the size of state observations or state size
state_size = env.observation_size

"""
###################################
STEP 5: Initialize a DQN Agent from the Agent Class in dqn_agent.py
A DQN agent initialized with the following state, action and DQN hyperparameters.
    DQN Agent Parameters
    ======
    state_size (int): dimension of each state (required)
    action_size (int): dimension of each action (required)

The DQN agent specifies a local and target neural network for training.
The network is defined in model.py. The input is a real (float) value vector of observations.
(NOTE: not appropriate for pixel data). It is a dense, fully connected neural network,
with 2 x 128 node hidden layers. The network can be modified by changing model.py.

Here we initialize an agent using the Unity environments state and action size determined above 
We also load the model parameters from training
"""
#Initialize Agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# Load trained model weights
agent.network.load_state_dict(torch.load('model/dqnAgent_Trained_Model.pth'))

"""
###################################
STEP 6: Play Banana for specified number of Episodes
"""

# reset the unity environment at the beginning of each episode
# set train mode to false
state = env.reset()
state = np.array([valor[0] for valor in state.values()])

# set the initial episode score to zero.
score = 0
step = 0

# Run the episode loop;
# At each loop step take an action as a function of the current state observations
# If environment episode is done, exit loop...
# Otherwise repeat until done == true
while True:
    step += 1
    # determine epsilon-greedy action from current sate
    action_index = agent.act(state)
    action = action_index * 0.01

    # send the action to the environment and receive resultant environment information
    next_state, reward, done, soh  = env.step(action)

    temp_buffer.append(next_state["agent_position"] * 45.)
    current_buffer.append(action * -46)
    voltage_buffer.append(next_state["agent_voltage"])
    soc_buffer.append(next_state["agent_soc"])

    next_state = np.array([valor[0] for valor in next_state.values()])

    # set new state to current state for determining next action
    state = next_state

    # Update episode score
    score += reward

    # If unity indicates that episode is done,
    # then exit episode loop, to begin new episode
    if done:
        elec_variables = {'current': current_buffer,
                          'temperature': temp_buffer,
                          'voltage': voltage_buffer,
                          'soc': soc_buffer}

        timestr = time.strftime("%Y%m%d-%H%M%S")
        # Save the recorded Scores data
        scores_filename = "dqnAgent_variables_Ito1" + timestr + ".csv"
        result_folder = "curves"
        filename = os.path.join(result_folder, scores_filename)
        score_path = os.path.join(os.getcwd(), filename)
        np.savetxt(score_path, elec_variables, delimiter=",")
        break

#g.electric_plot(temp_buffer, voltage_buffer, current_buffer, soc_buffer)


# END :) #############


