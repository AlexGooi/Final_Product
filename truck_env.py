from gymnasium import Env
from gymnasium.spaces import Discrete, Box,Dict,MultiBinary
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
from sim_manager import SimManager
from multiprocessing import Process
from limit import limit
from limit import scale_value

#-------------------------------------------------------------------------------


class Train_Class(Env):
    def __init__(self, amount_of_poles,grid_supply,ratio):
        self.action_space = Box(low=-1, high=1, shape=(amount_of_poles,), dtype=float)
        #self.observation_space = Box(low=0, high=500, shape=(2,), dtype=np.float32)
        self.observation_space = Dict({
        #Power factor per charging station
        'obs1': Box(low=0, high=1, shape=(amount_of_poles,), dtype=np.float32),
        #Percentage of total power used
        'obs2': Box(low=0, high=100, shape=(1,),dtype=np.float32),
        #Poles that are current;y in use
        'obs3': MultiBinary(amount_of_poles),
        #Charging time remaining on the charging poles
        'obs4': Box(low=0,high= 1000,shape=(amount_of_poles,),dtype= np.float32),
        #Desired battery level for each pole
        'obs5': Box(low= 0,high= 100,shape=(amount_of_poles,),dtype=np.float32),
        #Actual battery levels per pole
        'obs6': Box(low=0,high=100,shape=(amount_of_poles,),dtype=np.float32)
        })       
        #Init the state of the enviroment
        self.state = 0
        self.done = False
        self.running = False
        self.ratio = ratio
        self.extra_pos = 0
        self.extra_neg = 0
        self.neg_prog = 0
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
        info = {
            'poles_hold_back': 0
        }      
        #Scale the action into the amount of energy given to each charging pole
        for i in range(self.amount_of_poles):
            self.action_scaled[i] = np.float32(scale_value(action[i],-1,1,0.5,10))
        #Run 1 step of the simmulation

        done, rl_data = self.man.rl_Run(self.action_scaled)    

        #Reward function
        power_used_factor = self.__importance_ratio__(ratio=self.ratio,part1= rl_data['Percentage_Used'],part2=(rl_data["Avg_ChargePpercentage"]))
        reward = self.__reward__(power_used_factor=power_used_factor,ratio= self.ratio)
        #print(rl_data['Avg_ChargePpercentage'],rl_data['Percentage_Used'],reward)
        #scale_value(rl_data['Percentage_Used'])
        #print(self.__importance_ratio__(ratio=100,part1= 70,part2=30))
        #print(len(self.man.shedual.trucks))
        #When the simmulation has run for a day return the values one last time and reset the env
        #print(rl_data['Avg_ChargePpercentage'],rl_data['Percentage_Used'])
        if done:
            rl_data,temp = self.man.rl_reset()
            #print(rl_data['Avg_ChargePpercentage'],action[0],action)
            #power_used_factor = rl_data['Percentage_Used'] * (rl_data["Avg_ChargePpercentage"] / 100)
            power_used_factor = self.__importance_ratio__(ratio=self.ratio,part1= rl_data['Percentage_Used'],part2=(rl_data["Avg_ChargePpercentage"]))
            reward = self.__reward__(power_used_factor=power_used_factor,ratio= self.ratio)
            print(rl_data['Avg_ChargePpercentage'],rl_data['Percentage_Used'],reward)
        #Return the state of the simmulation to the rl model
        #print(rl_data['avg_wait'])
        #return_list = [np.float32(rl_data['avg_wait']), np.float32(rl_data['avg_tot']),rl_data['pole_data']]

        observation  = {'obs1':np.float32(rl_data['pole_charge_factors']),
                        'obs2': np.float32(rl_data['Percentage_Used']),
                        'obs3': rl_data['pole_data'],
                        'obs4': np.float32(rl_data['Poles_Time_Remaining']),
                        'obs5': np.float32(rl_data['Poles_Desired_Battery']),
                        'obs6': np.float32(rl_data['Poles_Battery_Levels'])
                        }
        observation['obs2'] = np.reshape(observation['obs2'], newshape=(1,1))
        info['poles_hold_back'] = rl_data['Poles_hold_back']
        #print(observation)
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
        observation  = {'obs1':np.float32([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), 
                        'obs2': np.float32([0]),
                        'obs3': np.float32([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                        'obs4': self.dummy, 
                        'obs5': np.float32([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                        'obs6': np.float32([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                    }
        observation['obs2'] = np.reshape(observation['obs2'], newshape=(1,1))
        return observation, info
    
    def __importance_ratio__ (self,ratio,part1,part2):
        #Limit the ration internal
        ratio_i = limit(0,ratio,100)
        #This method is used to balacne the reward (more towards happy customers or mer towards grid power)
        output1 = scale_value(input_value=part1,input_lower=0,input_upper= 100, output_lower=0,output_upper = ratio_i)
        output2 = scale_value(input_value=part2,input_lower=0,input_upper= 100, output_lower=0,output_upper = 100- ratio_i)
        #Return the combination of the 2 outputs
        return output1 + output2
    
    def __reward__(self,power_used_factor,ratio):
        # Normalize power_used_factor to be between 0 and 1
        normalized_power = power_used_factor / 100.0

        # Apply a non-linear transformation
        reward = normalized_power ** 2

        # Normalize reward to be between -1 and 1
        normalized_reward = 2 * (reward - 0.5)
        #Intergrator that adds extra bonus when the power consumption is 100% for a longer period
        if power_used_factor == 100 and ratio == 100:
            try:                               
                self.extra_pos += 0.5
            except:
                self.extra_pos = 0
        else:
            self.extra_pos = 0
        if power_used_factor < 95:
            self.extra_neg -= 0.05
            if power_used_factor < 90 and power_used_factor > 85:
                self.extra_neg -= 0.1    
            else:
                self.extra_neg -= (100 - power_used_factor ) /10
        else:
           self.extra_neg = 0
        return normalized_reward + self.extra_pos + self.extra_neg

model = 3
train = False
multi = False

if multi == False:
    #Create the reinfrocement learning model
    rl_env = Train_Class(amount_of_poles=20,grid_supply=70,ratio=100)
    log_path = os.path.join('.','logs')
    if model == 1:
        #PPO
        #learning_rate=0.000005 clip_range=0.1
        model = PPO('MultiInputPolicy', rl_env, verbose = 1, tensorboard_log = log_path,use_sde= True,sde_sample_freq= 4,device='cpu')
    elif model == 2:
        #AC2
        model = A2C('MultiInputPolicy', rl_env, verbose = 1, tensorboard_log = log_path)
    elif model == 3:
        model = SAC('MultiInputPolicy', rl_env, verbose = 1, tensorboard_log = log_path,device='cuda',buffer_size=7000000)
    elif model == 4:
        #DDPG
        model = DDPG('MultiInputPolicy', rl_env, verbose = 1, tensorboard_log = log_path,device='cuda',gamma=0.99,learning_rate=0.003)
    if train:
    #Create the reinfrocement learning model
        #,learning_rate=0.007,clip_range=0.4
        model = SAC.load('Power_Only_SAC_V2_3')
        model.set_env(rl_env)
        model.learn(total_timesteps= 1000000,progress_bar= True)
        model.save('Power_Only_SAC_V2_4')
    else:
        #Test the trained model
        model.load('Power_Only_SAC_V2_4')
        #model =  model.to('cuda') 
    #man = SimManager(10, 2400,spread_type=5,grid_supply=20)
        dummy = []
    #Create a action space scaled version (later used in the step method)
        actions = [0,0,0,0,6,6,6,6,6,5,5,5,5,5,5,5,5]    
        #done,rl_data = man.rl_Run(actions)
        observation , test  = rl_env.reset()
        #print(observation)
        for i in range(1):       
            done = False
            while done == False:  
                #print(observation['obs2'])
                actions , __states =  model.predict(observation)
                #print(actions)
                data,temp1,done,temp2,temp3 = rl_env.step(actions[0])
                #print("observatiosn = ",temp3['poles_hold_back'])
                #done,rl_data = man.rl_Run(actions)
                observation  = data
                print(data["obs2"])
            #print("Charge_Percentage", data['Avg_ChargePpercentage'])
            #print(data['avg_wait'],lent)
            #print(sim_data['Charge_Request'])
            
            rl_env.man.plot_consumption(moving_amount=100)

#------------------------------------------------------------------
names = ['100V2','50V2','0V2']
ratios = [100,50,0]
#Multi processing
def rl_model(name,ratio):
    rl_env = Train_Class(amount_of_poles=10,grid_supply=20,ratio=ratio)
    log_path = os.path.join('.','logs')
    model = PPO('MultiInputPolicy', rl_env, verbose = 1)


    model.learn(total_timesteps= 6000000,progress_bar= True)
    model.save(name)

processes = []
    # Create new processes and add them to your list
if multi:
    for i in range(3): # Change this number based on your needs
        process = Process(target=rl_model, args=(names[i],ratios[i]))
        processes.append(process)

    # Start all processes
    print(len(processes))
    for process in processes:
        process.start()

    # Ensure all processes have finished execution
    for process in processes:
        process.join()