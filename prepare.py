import random
import salabim as sim
import time
from scipy.stats import gamma, lognorm, beta
#from Elaad_distribution import calculate_distribution_params #now via pickle
from data import Truck
import pickle
import matplotlib.pyplot as plt
import numpy as np

#PICKLEEEE
#Arrival time parameters of best fitting distribution
with open('params_gamma_at.pkl', 'rb') as f:
    params_gamma_at = pickle.load(f)
#Avalaible service time of best fitting distribution
with open('params_lognorm_ast.pkl', 'rb') as f:
    params_lognorm_ast = pickle.load(f)
#Total Energy parameters of best fitting distribution
with open('params_lognorm_te.pkl', 'rb') as f:
    params_lognorm_te = pickle.load(f)

#Morning segment distributions
#Arrival time parameters of best fitting distributions
with open('params_gamma_at_morning.pkl', 'rb') as f:
    params_gamma_at_morning = pickle.load(f)

with open('params_lognorm_at_morning.pkl', 'rb') as f:
    params_lognorm_at_morning = pickle.load(f)

with open('params_gamma_ast_morning.pkl', 'rb') as f:
    params_gamma_ast_morning = pickle.load(f)

with open('params_gamma_te_morning.pkl', 'rb') as f:
    params_gamma_te_morning = pickle.load(f)

with open('params_lognorm_te_morning.pkl', 'rb') as f:
    params_lognorm_te_morning = pickle.load(f)


class Prepare:
    '''Class that prepares a car arrival set'''

    def __init__(self, total_time):
        # Create an empty list where we can store the truck scedual
        self.trucks = []
        self.total_time = total_time
        self.arrival_times = sim.Exponential(60 / 40)
        self.service_times = sim.Exponential(60 / 50)
        random.seed(time.time())
        self.avg_wait_time = []

    def prepare_data(self, spread_type):
        self.trucks = []
        time = 0
        first = False

        # Lists to store individual truck data for min, max, and avg calculations
        battery_levels = []
        arrival_times = []
        total_times = []
        wait_times = []
        desired_battery_levels = []
        max_wait_times = []

        # Loop until a day is finished
        while time < self.total_time:
            if spread_type == 1:
                # Create a new data object
                truck_data = Truck(
                    battery=sim.Uniform(20, 80).sample(),
                    arrival_Time=time,
                    total_time=0,
                )
            elif spread_type == 2:
                truck_data = Truck(
                    battery=sim.Uniform(40).sample(),
                    arrival_Time=time,
                    total_time=0,
                    total_wait_Time=0,
                )
            #When this spread is chosen the waiting time before charging should avarge arround 4.8 mintes (used for checks)
            elif spread_type == 3:
                arrival, service_time = self.poison()
                self.avg_wait_time.append(service_time)
                service_invert = 100.0 - service_time
                truck_data = Truck(
                    battery=service_invert,
                    arrival_time=time,
                    total_time=0,
                    total_wait_time=0,
                    desired_wait_time=0
                )
                #print("prepare")
            elif spread_type == 4:
                arrival = gamma(*params_gamma_at).rvs() # Generate arrival time using the Gamma distribution\
                max_wait_time = lognorm(*params_lognorm_ast).rvs() # Generate max wait time using the Lognormal distribution for available service time
                total_energy = min(70, lognorm(*params_lognorm_te).rvs()) # Generate total energy demand and ensure it does not exceed 70 kWh
                battery_level = max(0, min(100, (total_energy / 70) * 100)) # Calculate the desired battery level based on total energy demand
                desired_battery = 100 - battery_level
                truck_data = Truck(
                    battery=battery_level,
                    arrival_time=time,
                    total_time=0,
                    total_wait_time=0, 
                    desired_wait_time=0,
                    desired_battery=desired_battery,
                    max_wait_time=max_wait_time    
                )
                print("Truck Data for Spread Type 4:", truck_data.__dict__)
                # Collecting data for calculations
                battery_levels.append(truck_data.battery)
                arrival_times.append(truck_data.arrival_time)
                total_times.append(truck_data.total_time)
                wait_times.append(truck_data.total_wait_time)
                desired_battery_levels.append(truck_data.desired_battery)
                max_wait_times.append(truck_data.max_wait_time)
            #Poison with a larger charging time 
            elif spread_type == 5:
                arrival, service_time = self.poison()
                self.avg_wait_time.append(service_time)
                
                battery_level = 100.0 - random.randint(10, 100)
                desired_level = random.randint(70,100)
                max_wait_time = random.randint(700,730)
                truck_data = Truck(
                    battery=battery_level,
                    arrival_time=time,
                    total_time=0,
                    total_wait_time=0,
                    desired_wait_time=0,
                    desired_battery= desired_level,
                    max_wait_time = max_wait_time
                )
                print("Truck Data for Spread Type 5:", truck_data.__dict__)
                # Collecting data for calculations
                battery_levels.append(truck_data.battery)
                arrival_times.append(truck_data.arrival_time)
                total_times.append(truck_data.total_time)
                wait_times.append(truck_data.total_wait_time)
                desired_battery_levels.append(truck_data.desired_battery)
                max_wait_times.append(truck_data.max_wait_time)
            elif spread_type == 6:
                alpha, beta_params = 2, 0.5  
                variability_factor = beta(alpha, beta_params).rvs()  # Generate variability factor using beta distribution
                
                arrival = gamma(*params_gamma_at_morning).rvs() # Generate arrival time using the Gamma distribution
                
                #Available_Service_Time (AST) / Wait_time distribution mean is too high; distribution needs to be skewed
                baseline_max_wait_time = gamma(*params_gamma_ast_morning).rvs() # Generate max wait time using the Gamma distribution for AST
                mean_shift = 100 # You can also add ' * variability_factor ' to Adjust the mean shift based on the variability factor
                skewed_max_wait_time = baseline_max_wait_time - mean_shift  # Directly adjust the mean
                max_wait_time = max(15, min(skewed_max_wait_time, 1440)) #Set min. and max wait time. Min is now 15 min. and max is now 24 hours.

                #Total_energy distribution does not have enough variability, so a variability factor is used
                baseline_total_energy = lognorm(*params_lognorm_te_morning).rvs()  # Generate baseline energy demand using lognormal distribution


                weight_for_baseline_te = 0  # Adjust this weight to control the influence of the baseline. Since TE does not have enough variability, close to 0 is better for the simulation.
                weight_for_variability_te = 1 - weight_for_baseline_te
                total_energy = (weight_for_baseline_te * baseline_total_energy) + (weight_for_variability_te * variability_factor * 70) # Weighted sum of baseline energy and variability factor
                total_energy = max(1, min(total_energy, 70))  # Ensure total energy is within min max of battery, in this case 1-70
                battery_level = max(0, min(100, (total_energy / 70) * 100))  # Calculate desired battery level based on total energy demand

                desired_battery = 100 - battery_level  # Ensure battery_level is logically consistent with desired_battery and with that total_energy
                
                truck_data = Truck(
                    battery=battery_level,
                    arrival_time=time,
                    total_time=0,
                    total_wait_time=0, 
                    desired_wait_time=0,
                    desired_battery=desired_battery,
                    max_wait_time=max_wait_time    
                )
                print("Truck Data for Spread Type 6:", truck_data.__dict__)
                # Collecting data for calculations
                battery_levels.append(truck_data.battery)
                arrival_times.append(truck_data.arrival_time)
                total_times.append(truck_data.total_time)
                wait_times.append(truck_data.total_wait_time)
                desired_battery_levels.append(truck_data.desired_battery)
                max_wait_times.append(truck_data.max_wait_time)

            # Append the data to the list
            self.trucks.append(truck_data)

            # Determine the new arrival time
            if first == False:
                time += arrival
            else:
                first = True

        # After all trucks are generated, calculate and print min, max, avg
        print("Minimum, Average, and Maximum values for Spread Type", spread_type)
        print("Min Battery Level:", min(battery_levels))
        print("Avg Battery Level:", sum(battery_levels) / len(battery_levels))
        print("Max Battery Level:", max(battery_levels))

        print("Min Arrival Time:", min(arrival_times))
        print("Avg Arrival Time:", sum(arrival_times) / len(arrival_times))
        print("Max Arrival Time:", max(arrival_times))

    def poison(self): #function is called poison, but is actually an exponential function.
        # Given parameters

        # Generate inter-arrival times for the students (Poisson process)
        # Generate service times for the students (exponential distribution)

        return self.arrival_times.sample(), self.service_times.sample()