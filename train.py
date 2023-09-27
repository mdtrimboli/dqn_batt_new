"""
DQN for Unity ML-Agents Environments using PyTorch
Includes examples of the following DQN training algorithms:
  -> Vanilla DNQ,
  -> Double-DQN (DDQN)

The example uses a modified version of the Unity ML-Agents Banana Collection Example Environment.
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
import time
import yaml
import os
import random
import numpy as np
from collections import deque

from dqn_agent import Agent
from multiobjective import MODQNTrainer as MODQN
from battery import Battery
"""
###################################
STEP 1: Set the Training Parameters
======
        num_episodes (int): maximum number of training episodes
        epsilon (float): starting value of epsilon, for epsilon-greedy action selection
        epsilon_min (float): minimum value of epsilon
        epsilon_decay (float): multiplicative factor (per episode) for decreasing epsilon
        scores (float): list to record the scores obtained from each episode
        scores_average_window (int): the window size employed for calculating the average score (e.g. 100)
        solved_score (float): the average score required for the environment to be considered solved
        (here we set the solved_score a little higher than 13 [i.e., 14] to ensure robust learning).
    """
num_episodes = 15000
num_objectives = 2
priorities = [1, 1]

dict_dqn = {}
dqn_values = {}
scores = [0] * num_objectives
scores_avg = {}

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99
episodic_rew = []
scores_average_window = 200
solved_score = -20

with open("default.yml", "r") as f:
    config = yaml.safe_load(f)

"""
###################################
STEP 2: Start Environment
# Use the corresponding call depending on your operating system 
"""
env = Battery(config)

"""
#############################################
STEP 4: Determine the size of the Action and State Spaces
"""

# Set the number of actions or action size
action_size = env.action_size

# Set the size of state observations or state size
state_size = env.observation_size

"""
###################################
STEP 5: Create a DQN Agent from the Agent Class in dqn_agent.py
A DQN agent initialized with the following state, action and DQN hyperparameters.
    DQN Agent Parameters
    ======
    state_size (int): dimension of each state (required)
    action_size (int): dimension of each action (required)
    dqn_type (string): can be either 'DQN' for vanillia dqn learning (default) or 'DDQN' for double-DQN.
    replay_memory size (int): size of the replay memory buffer (default = 1e5)
    batch_size (int): size of the memory batch used for model updates (default = 64)
    gamma (float): parameter for setting the discounted value of future rewards (default = 0.99)
    learning_rate (float): specifies the rate of model learning (default = 5e-4)
    seed (int): random seed for initializing training point (default = 0)

The DQN agent specifies a local and target neural network for training.
The network is defined in model.py. The input is a real (float) value vector of observations.
(NOTE: not appropriate for pixel data). It is a dense, fully connected neural network,
with 2 x 128 node hidden layers. The network can be modified by changing model.py.

Here we initialize an agent using the Unity environments state and action size determined above 
and the default DQN hyperparameter settings.
"""

# Create dictionaries
for i in range(num_objectives):
    dqn = {
        "model": "",
        "q_values": "",
        "d_values": "",
        "sd_values": ""
    }
    dict_dqn[f'dqn_{i + 1}'] = dqn




for dqn in range(num_objectives):
    dqn_module = Agent(state_size=state_size, action_size=action_size, dqn_type='DQN')
    dict_dqn[f"dqn_{dqn + 1}"]['model'] = dqn_module
    scores_avg[f"dqn_{dqn + 1}"] = []

modqn = MODQN(action_size)

"""
###################################
STEP 6: Run the DQN Training Sequence
The DQN RL Training Process involves the agent learning from repeated episodes of behaviour 
to map states to actions the maximize rewards received via environmental interaction.
The artificial neural network is expected to converge on or approximate the optimal function 
that maps states to actions. 

The agent training process involves the following:
(1) Reset the environment at the beginning of each episode.
(2) Obtain (observe) current state, s, of the environment at time t
(3) Use an epsilon-greedy policy to perform an action, a(t), in the environment 
    given s(t), where the greedy action policy is specified by the neural network.
(4) Observe the result of the action in terms of the reward received and 
	the state of the environment at time t+1 (i.e., s(t+1))
(5) Calculate the error between the actual and expected Q value for s(t),a(t),r(t) and s(t+1)
	to update the neural network weights.
(6) Update episode score (total reward received) and set s(t) -> s(t+1).
(7) If episode is done, break and repeat from (1), otherwise repeat from (3).

Below we also exit the training process early if the environment is solved. 
That is, if the average score for the previous 100 episodes is greater than solved_score.
"""

# loop from num_episodes
for i_episode in range(1, num_episodes + 1):

    #   reset the unity environment at the beginning of each episode
    state = env.reset()
    state = np.array([valor[0] for valor in state.values()])

    # set the initial episode score to zero.
    scores = np.zeros(num_objectives, dtype=float)
    step = 0
    actions = 0
    episodic_rew = []
    # Run the episode training loop;
    # At each loop step take an epsilon-greedy action as a function of the current state observations
    # Based on the resultant environmental state (next_state) and reward received update the Agent network
    # If environment episode is done, exit loop...
    # Otherwise repeat until done == true
    while True:
        step += 1

        for dqn in range(num_objectives):
            qv, dv, sdv = dict_dqn[f"dqn_{dqn + 1}"]['model'].get_values(state)

            dict_dqn[f"dqn_{dqn + 1}"]['q_values'] = modqn.scaled_q_values(qv)
            dict_dqn[f"dqn_{dqn + 1}"]['d_values'] = dv
            dict_dqn[f"dqn_{dqn + 1}"]['sd_values'] = sdv

        # determine epsilon-greedy action from current state
        q_values_unified = modqn.sum_weighted_q_values(dict_dqn, num_objectives, priorities)
        action_index = modqn.get_action(q_values_unified, epsilon)

        #action_index = modqn.get_action(q_values, epsilon)
        action = action_index/100

        actions += action
        #print('Episode {}\tStep: {}\tAction: {}\tSUM_Action: {}'.format(i_episode, step, action, actions), end="")

        # send the action to the environment and receive resultant environment information
        next_state, reward, done, dv_reward = env.step(action)
        next_state = np.array([valor[0] for valor in next_state.values()])

        # Send (S, A, R, S') info to the DQN agent for a neural network update
        for dqn in range(num_objectives):
            dict_dqn[f"dqn_{dqn + 1}"]['model'].step(state, action, dict_dqn[f"dqn_{dqn + 1}"]['d_values'], reward[dqn], dv_reward[dqn], next_state, done)
            scores[dqn] += reward[dqn]

        #agent.step(state, action, reward, next_state, done)
        # set new state to current state for determining next action
        state = next_state

        # Update episode score
        #score_O1 += reward[0]
        #score_O2 += reward[1]

        # If unity indicates that episode is done,
        # then exit episode loop, to begin new episode
        if done:
            break

    # Add episode score to Scores and...
    # Calculate mean score over last 100 episodes
    # Mean score is calculated over current episodes until i_episode > 100
    for dqn in range(num_objectives):
        scores_avg[f"dqn_{dqn + 1}"].append(scores[dqn])
    #scores.append(score)
    for dqn in range(num_objectives):
        episodic_rew.append(np.mean(scores_avg[f"dqn_{dqn + 1}"][i_episode - min(i_episode, scores_average_window):i_episode + 1]))

    #average_score = np.mean(scores_avg[i_episode - min(i_episode, scores_average_window):i_episode + 1])
    #episodic_rew.append(average_score)
    # Decrease epsilon for epsilon-greedy policy by decay rate
    # Use max method to make sure epsilon doesn't decrease below epsilon_min
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    # (Over-) Print current average score
    print('\rEpisode {} \tSteps:{} \tScore: {}\tAvg_Action: {:.2f}'.format(i_episode, step, episodic_rew, actions/step), end="")

    # Print average score every scores_average_window episodes
    if i_episode % scores_average_window == 0:
        print('\rEpisode {}\tSteps:{}\tAvg_score: {}\tAvg_Action: {:.2f}'.format(i_episode, step, episodic_rew, actions/step))

    # Check to see if the task is solved (i.e,. avearge_score > solved_score).
    # If yes, save the network weights and scores and end training.
    if episodic_rew[0] >= solved_score:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, episodic_rew[0]))

        # Save trained neural network weights
        timestr = time.strftime("%Y%m%d-%H%M%S")
        weight_folder = "model"
        nn_filename = "dqnAgent_Trained_Model_" + timestr + ".pth"
        filename = os.path.join(weight_folder, nn_filename)
        weight_path = os.path.join(os.getcwd(), filename)
        #torch.save(agent.network.state_dict(), weight_path)
        # TODO: Guardar pesos de ambos modelos

        # Save the recorded Scores data
        # TODO: Ver que variables guardar
        scores_filename = "dqnAgent_scores_" + timestr + ".csv"
        result_folder = "score"
        filename = os.path.join(result_folder, scores_filename)
        score_path = os.path.join(os.getcwd(), filename)
        #np.savetxt(score_path, scores, delimiter=",")
        break

"""
###################################
STEP 7: Everything is Finished -> Close the Environment.
"""
#env.close()

# END :) #############
