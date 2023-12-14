from gymnasium import Env
from gymnasium.spaces import Discrete, Box,Dict,MultiBinary
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
    def __init__(self, amount_of_poles):
        self.action_space = Box(low=-1, high=1, shape=(amount_of_poles,), dtype=float)
        #self.observation_space = Box(low=0, high=500, shape=(2,), dtype=np.float32)
        self.observation_space = Dict({
        #Avarage wait time before service
        'obs1': Box(low=0, high=500, shape=(1,), dtype=np.float32),
        #Avarage time for the whole operation
        'obs2': Box(low=0,high=600,shape=(1,),dtype= np.float32), 
        #Total amount of charging request in the waiting line
        'obs3': Box(low=0,high= 10000,shape=(1,),dtype=np.float32),
        #Poles that are current;y in use
        'obs4': MultiBinary(amount_of_poles)
        })       
        #Init the state of the enviroment
        self.state = 0
        self.done = False
        self.running = False
        self.man = SimManager(amount_of_poles, 2400,spread_type=5)
        self.amount_of_poles = amount_of_poles
        #Create a dummy list of booleans (used for the reset function)
        self.dummy = []
        #Create a action space scaled version (later used in the step method)
        self.action_scaled = []
        for i in range(amount_of_poles):
            self.dummy.append(False)
            self.action_scaled.append(np.float32(0))
        

    def step(self, action):
        reward = 0
        info = {}      
        #Scale the action into the amount of energy given to each charging pole
        for i in range(self.amount_of_poles):
            self.action_scaled[i] = np.float32(scale_value(action[i],-1,1,0.001,300))
        #Run 1 step of the simmulation

        done, rl_data = self.man.rl_Run(self.action_scaled)    
        #print(rl_data['avg_tot'])
        diffrence = abs(30 - rl_data['avg_tot'])
        #print(diffrence)
        reward = 20 - diffrence
        #When the simmulation has run for a day return the values one last time and reset the env
        
        if done:
            rl_data,temp = self.man.rl_reset()
            print(rl_data['avg_tot'])
            diffrence = abs(30 - rl_data['avg_tot'])
            reward = 20 - diffrence
        #Return the state of the simmulation to the rl model
        #print(rl_data['avg_wait'])
        #return_list = [np.float32(rl_data['avg_wait']), np.float32(rl_data['avg_tot']),rl_data['pole_data']]

        observation  = {'obs1':np.float32(rl_data['avg_wait']), 'obs2': np.float32(rl_data['avg_tot']),'obs3':np.float32(rl_data['Charge_Request']), 'obs4': rl_data['pole_data']}
        return observation, reward/2400, done, False, info

    def render(self):
        #This method is required by the framework but doesn't have to do anything
        pass

    #This method resets the eniroment
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Reset the simmulation enviroment
        info = {}
        observation  = {'obs1':np.float32(0), 'obs2': np.float32(0), 'obs3' :np.float32(0),'obs4': self.dummy}
        return observation, info
    
#Create the reinfrocement learning model
rl_env = Train_Class(amount_of_poles=1)
log_path = os.path.join('.','logs')
model = PPO('MultiInputPolicy', rl_env, verbose = 1, tensorboard_log = log_path,learning_rate=0.007,clip_range=0.4)

model.learn(total_timesteps= 300000,progress_bar= True)