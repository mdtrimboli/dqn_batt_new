import gym
from gym.spaces import Box, Dict, Discrete
import numpy as np
from numpy import linalg as LA
import os
import scipy.io as spio     # Work with .mat files


class Battery(gym.Env):
    def __init__(self, config):
        # Set the properties for spaces
        self._config = config["model"]["hyperparameters"]
        action_range = np.arange(0.01, 1.01, 0.01)
        self.action_space = Discrete(len(action_range))
        self.action_size = len(action_range)
        #self.action_space = Box(low=-1, high=1, shape=(self._config.n,), dtype=np.float32)          # Maxi: Current
        self.observation_space = Dict({
            'agent_position': Box(low=0., high=1., shape=(self._config["n"],), dtype=np.float32),      # Maxi: T = [5, 45]
            'agent_soc': Box(low=0, high=1, shape=(self._config["n"],), dtype=np.float32),
            'agent_voltage': Box(low=2., high=3.6, shape=(self._config["n"],), dtype=np.float32)
        })
        self.observation_size = len(self.observation_space)
        # Sets all the episode specific variables

        self.sigma = [self._config['ito_soc'], self._config["ito_soh"]]
        self.cbat = self._config["Cn"] * 3600
        self.soc = self._config["init_soc"]
        self.soh = self._config["init_soh"]
        self.vc1 = self._config["vc1"]
        self.vc2 = self._config["vc2"]
        self.dt = self._config["frequency_ratio"]
        self.count = 0
        self.Tc = self._config["Tc"]
        self.Ts = self._config["Ts"]

        ## Load maps
        dir_current = os.getcwd()
        #mat = spio.loadmat(dir_current + '/safe_charge/env/battery_mappings.mat', squeeze_me=True)
        mat = spio.loadmat(dir_current + '\\env\\battery_mappings.mat', squeeze_me=True)
        self.ocv_map = np.array(mat['ocv_curve'])
        self.soc_map = np.linspace(0, 1, len(self.ocv_map))
        self.ocv = self.calculate_ocv(self.soc)
        self.v_batt = self.ocv
        self.curr_action = 0

        self.reset()

    def reset(self):
        if self._config["random_start"]:
            self.soc = np.random.random()
            self.soh = np.random.random()
            self.Tc = self._config["Tc"]
            self.Ts = self._config["Ts"]
            self.vc1 = 0
            self.vc2 = 0
            self.ocv = self.calculate_ocv(self.soc)
        else:
            self.soc = self._config["init_soc"]
            self.soh = self._config["init_soh"]
            # self.soh = np.load('curves/final_soh.npy')
            self.Tc = self._config["Tc"]  # default:23
            self.Ts = self._config["Ts"]  # default:23
            self.vc1 = 0
            self.vc2 = 0
            self.ocv = self.calculate_ocv(self.soc)

        self.done = False
        self._current_time = 0.
        self._move_agent(1.)        # Default (0.) ... no deberia ser 1?

        observation = {
            "agent_position": self._agent_position,
            "agent_soc": self.soc,
            "agent_voltage": self.v_batt*np.ones(self._config["n"],  dtype=np.float32)
        }

        return observation
        # return self.step(np.zeros(self._config.n))[0]

    def _get_reward(self):
        reward_soc = self.soc - 1
        #reward_soc = self.curr_action + 1 / self._current_time

        if self._agent_position > 1.:
            reward_soh = np.array([-1.])

        else:
            reward_soh = np.array([0.])
            #reward_soh = self._agent_position


        reward = [reward_soc, reward_soh]
        dw_reward = [abs(x) for x in reward]

        #return reward, dw_reward
        return reward

    def _move_agent(self, current_norm):
        # Old: Assume that frequency of motor is 1 (one action per second)
        current = np.clip(-46. * current_norm, -46., 0)        # default: [0 -46]
        self.curr_action = current
        self.calculate_params(current)          # Tm
        self.soc = self.calculate_soc(current)
        self.soh = self.calculate_soh(current)
        self.ocv = self.calculate_ocv(self.soc)
        self.v_batt = self.ocv - current * self.R0 - self.compute_vc1(current) - self.compute_vc2(current)

    def _is_agent_outside_boundary(self):
        return np.any(self.soc < 0) or np.any(self.soc > 1)

    def _is_agent_outside_shaping_boundary(self):
        return np.any(self._agent_position < self._config["reward_shaping_slack"]) \
               or np.any(self._agent_position > 1 - self._config["reward_shaping_slack"])

    def _update_time(self):
        # Assume that frequency of motor is 1 (one action per second)
        #self._current_time += self._config.frequency_ratio
        self._current_time += 1

    def step(self, action):

        # Increment time
        self._update_time()

        # Calculate new position of the agent
        self._move_agent(action)

        # Find reward
        reward, dv_reward = self._get_reward()

        # Prepare return payload
        observation = {
            "agent_position": self._agent_position,
            "agent_soc": self.soc,
            "agent_voltage": self.v_batt*np.ones(self._config["n"],  dtype=np.float32)
        }

        soh = self.soh

        done = self._is_agent_outside_boundary()
               #or int(self._current_time // 1) > self._config.episode_length

        #return observation, [reward, dv_reward], done, soh
        return observation, reward, done, soh

    def calculate_params(self, i):
        self.crate = abs(i / self.cbat)     # c-rate
        self.M = 1687.2 * self.crate**3 - 9522 * self.crate**2 + 6806.8 * self.crate + 32658  # per-exponential factor
        self.Ea = 31700 - 370.3 * self.crate  # activation energy
        self.Atol = (20 / (self.M * np.exp(-self.Ea /(self._config["R"]*(self.Tc + 273.15)))))**(1/self._config["z"])  # total discharged Ah
        self.N = 3600 * self.Atol / self.cbat       # number of cycles
        self.Tm = np.array([(self.Ts + self.Tc) / 2.]).flatten()  # average temperature
        self._agent_position = (self.Tm - 5.) / (45. - 5.)

        if i < 0:
            self.R0 = self._config["R0c"] * np.exp(self._config["Tref_R0c"] / (self.Tm - self._config["Tshift_R0c"]))
            self.R1 = (self._config["R1c"][0] + self._config["R1c"][1] * self.soc +
                      self._config["R1c"][2] * self.soc ** 2) * np.exp(self._config["Tref_R1c"] /
                                                                        (self.Tm - self._config["Tshift_R1c"]))
            self.R2 = (self._config["R2c"][0] + self._config["R2c"][1] * self.soc +
                      self._config["R2c"][2] * self.soc ** 2) * np.exp(self._config["Tref_R2c"] / self.Tm)
            self.C1 = self._config["C1c"][0] + self._config["C1c"][1] * self.soc + \
                      self._config["C1c"][2] * self.soc ** 2 + (self._config["C1c"][3] + self._config["C1c"][4] *
                      self.soc + self._config["C1c"][5] * self.soc ** 2) * self.Tm
            self.C2 = self._config["C2c"][0] + self._config["C2c"][1] * self.soc + self._config["C2c"][2] * \
                      self.soc ** 2 + (self._config["C2c"][3] + self._config["C2c"][4] * self.soc +
                      self._config["C2c"][5] * self.soc ** 2) * self.Tm
        else:
            self.R0 = self._config["R0d"] * np.exp(self._config["Tref_R0d"] / (self.Tm - self._config["Tshift_R0d"]))
            self.R1 = (self._config["R1d"][0] + self._config["R1d"][1] * self.soc + self._config["R1d"][2] * self.soc ** 2) * \
                      np.exp(self._config["Tref_R1d"] / (self.Tm - self._config["Tshift_R1d"]))
            self.R2 = (self._config["R2d"][0] + self._config["R2d"][1] * self.soc + self._config["R2d"][2] * self.soc ** 2) * \
                      np.exp(self._config["Tref_R2d"] / self.Tm)
            self.C1 = self._config["C1d"][0] + self._config["C1d"][1] * self.soc + self._config["C1d"][2] * self.soc ** 2 + \
                      (self._config["C1d"][3] + self._config["C1d"][4] * self.soc + self._config["C1d"][5] * self.soc ** 2) * self.Tm
            self.C2 = self._config["C2d"][0] + self._config["C2d"][1] * self.soc + self._config["C2d"][2] * self.soc ** 2 + \
                      (self._config["C2d"][3] + self._config["C2d"][4] * self.soc + self._config["C2d"][5] * self.soc ** 2) * self.Tm

        self.alfa1 = 1 / (self.R1 * self.C1)
        self.beta1 = 1 / self.C1
        self.alfa2 = 1 / (self.R2 * self.C2)
        self.beta2 = 1 / self.C2

    def calculate_soc(self, i):
        e = np.random.randn(1) * self._config["pdfstd"] + self._config["pdfmean"]
        dw = (e * (np.sqrt(self._config["dtao"])))  # brownian motion
        w_soc = 1e-3 * self.sigma[0] * dw  # variability soc        #default 1e-3
        #w_soc = [0.]
        return self.soc + (-i / self.cbat) * self.dt + w_soc


    def calculate_soh(self, i):

        e = np.random.randn(1) * self._config["pdfstd"] + self._config["pdfmean"]
        dw = (e * (np.sqrt(self._config["dtao"])))  # brownian motion
        w_tc = 1e-3 * self._config["ito_temp"] * dw  # variability soc        #default 1e-3 [ito(delta Tc)]
        #w_tc = [0.]

        self.Tc = self.Tc + ((self.Ts - self.Tc) / (self._config["Rc"] * self._config["Cc"]) + i *
                             (self.vc1 + self.vc2 + self.R0 * i) / self._config["Cc"]) * self.dt + w_tc

        self.Ts = self.Ts + ((self._config["Tf"] - self.Ts) / (self._config["Ru"] * self._config["Cs"]) -
                             (self.Ts - self.Tc) / (self._config["Rc"] * self._config["Cc"])) * self.dt

        e = np.random.randn(1) * self._config["pdfstd"] + self._config["pdfmean"]
        dw = (e * (np.sqrt(self._config["dtao"])))  # brownian motion
        w_soh = 1e-7 * self.sigma[1] * dw  # variability soc
        return self.soh - (np.absolute(i) / (2 * self.N * self.cbat)) * self.dt + w_soh

    def calculate_ocv(self, soc):
        return np.interp(soc, self.soc_map, self.ocv_map)

    def compute_vc1(self, i):
        self.vc1 = self.vc1 + (-self.alfa1 * self.vc1 + self.beta1 * i) * self.dt
        return self.vc1

    def compute_vc2(self, i):
        self.vc2 = self.vc2 + (-self.alfa2 * self.vc2 + self.beta2 * i) * self.dt
        return self.vc2
