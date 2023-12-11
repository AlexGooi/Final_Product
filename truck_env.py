from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
from stable_baselines3 import PPO
from sim_manager import SimManager
from limit import limit
#-------------------------------------------------------------------------------
def scale_value(input_value, input_lower, input_upper, output_lower, output_upper):
    # Scale the input value
    scaled_value = ((input_value - input_lower) / (input_upper - input_lower)) * (output_upper - output_lower) + output_lower
    scaled_value = limit(output_lower,scaled_value,output_upper)
    return scaled_value
#-------------------------------------------------------------------------------


class Train_Class(Env):
    def __init__(self):
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=float)
        self.observation_space = Box(low=0, high=500, shape=(2,), dtype=np.float32)
        self.state = 0
        self.done = False
        self.running = False
        self.man = SimManager(2, 1400)
        self.man.run_sim()

    def step(self, action):
        reward = 0
        info = {}      
        done,avarage,length,charge = self.man.rl_Run(action)    
        diffrence = abs(7 - avarage)
        reward = 20 - diffrence

        if done:
            wait_time = self.man.rl_reset()
            print(wait_time[0])

            diffrence = abs(7 - wait_time[0])
            reward = 20 - diffrence

        return_list = [np.float32(length), np.float32(charge)]
        return return_list, reward, True, False, info

    def render(self):
        #This method is required by the framework but doesn't have to do anything
        pass

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Reset the simmulation enviroment

        info = {}
        return_list = [np.float32(0), np.float32(0)]
        return return_list, info

rl_env = Train_Class()
log_path = os.path.join('.','logs')
model = PPO('MlpPolicy', rl_env, verbose = 1, tensorboard_log = log_path)

#model.learn(total_timesteps= 300000,progress_bar= True)