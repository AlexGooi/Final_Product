from gymnasium import Env
from gymnasium.spaces import Discrete, Box,Dict,MultiBinary
import numpy as np
import os
from stable_baselines3 import PPO
from sim_manager import SimManager
from limit import limit
from limit import scale_value

#-------------------------------------------------------------------------------


class Train_Class(Env):
    def __init__(self, amount_of_poles,grid_supply):
        self.action_space = Box(low=-1, high=1, shape=(amount_of_poles,), dtype=float)
        #self.observation_space = Box(low=0, high=500, shape=(2,), dtype=np.float32)
        self.observation_space = Dict({
        #Power factor per charging station
        'obs1': Box(low=0, high=1, shape=(amount_of_poles,), dtype=np.float32),
        #Percentage of total power used
        'obs2': Box(low=0, high=100, shape=(1,),dtype=np.float32),
        #Poles that are current;y in use
        'obs3': MultiBinary(amount_of_poles)
        })       
        #Init the state of the enviroment
        self.state = 0
        self.done = False
        self.running = False
        self.man = SimManager(amount_of_poles, 2400,spread_type=5,grid_supply=grid_supply)
        self.amount_of_poles = amount_of_poles
        #Create a dummy list of booleans (used for the reset function)
        self.dummy = []
        #Create a action space scaled version (later used in the step method)
        self.action_scaled = []
        for i in range(amount_of_poles):
            self.dummy.append(False)
            self.action_scaled.append(np.float32(5))
        #Setup the systme with 1 car already gone thorugh the system (this is done to make sure that the system is up and running)
        self.man.loop_first_car(self.action_scaled)

    def step(self, action):
        reward = 0
        info = {}      
        #Scale the action into the amount of energy given to each charging pole
        for i in range(self.amount_of_poles):
            self.action_scaled[i] = np.float32(scale_value(action[i],-1,1,0.5,10))
        #Run 1 step of the simmulation

        done, rl_data = self.man.rl_Run(self.action_scaled)    

        #Reward function
        power_used_factor = rl_data['Percentage_Used'] * (rl_data["Avg_ChargePpercentage"] /100)
        reward = power_used_factor -50  
        print(len(self.man.shedual.trucks))
        #When the simmulation has run for a day return the values one last time and reset the env
        #print(rl_data['Avg_ChargePpercentage'],rl_data['pole_data'])
        if done:
            rl_data,temp = self.man.rl_reset()
            print(rl_data['Avg_ChargePpercentage'],action[0],action)
            power_used_factor = rl_data['Percentage_Used'] * (rl_data["Avg_ChargePpercentage"] / 100)
            reward = power_used_factor -50
        #Return the state of the simmulation to the rl model
        #print(rl_data['avg_wait'])
        #return_list = [np.float32(rl_data['avg_wait']), np.float32(rl_data['avg_tot']),rl_data['pole_data']]

        observation  = {'obs1':np.float32(rl_data['pole_charge_factors']), 'obs2': np.float32(rl_data['Percentage_Used']),'obs3': rl_data['pole_data']}
        return observation, reward/2400, done, False, info

    def render(self):
        #This method is required by the framework but doesn't have to do anything
        pass

    #This method resets the eniroment
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Reset the simmulation enviroment
        info = {}
        #Init the next run(make sure there is already a second car in the system)
        self.man.loop_first_car(self.action_scaled)
        #Reset the observation
        observation  = {'obs1':np.float32(0), 'obs2': np.float32(0), 'obs3' :np.float32(0),'obs4': self.dummy}
        return observation, info
    
#Create the reinfrocement learning model
rl_env = Train_Class(amount_of_poles=1,grid_supply=10)
log_path = os.path.join('.','logs')
model = PPO('MultiInputPolicy', rl_env, verbose = 1, tensorboard_log = log_path,learning_rate=0.007,clip_range=0.4)

model.learn(total_timesteps= 300000,progress_bar= True)