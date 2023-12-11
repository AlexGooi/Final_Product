from typing import List, Tuple
import salabim as sim
import numpy as np
from customer_generator import CustomerGenerator
from power_supply import PowerSupply
from charging_station import ChargingStation
from prepare import Prepare
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

class SimManager:
    def __init__(self, charging_stations: int, total_time: int,spread_type):
        self.shedual = Prepare(total_time=total_time)
        self.charging_stations = charging_stations

        # Prepare the truck data
        self.shedual.prepare_data(spread_type=spread_type)
        self.total_time = total_time

        # Create varaibles for monitoring
        self.wait_times = []
        self.total_times = []
        self.time_before_service = []
        self.desired_times = []
        self.difference_times = []
        self.poles_active = []
        self.power_consumption_trend = [] #List used for monitopring power consumption during the day
        self.first = False
        self.old_time = 0
        self.spread_type = spread_type
        # Setup the enviroment
        self.env_sim = sim.App(
            trace=False,
            random_seed="*",
            name="Simmulation",
            do_reset=False,
            yieldless=True,
        )
        # Create the power supply
        self.power_supply_o = PowerSupply(env=self.env_sim, max_power_from_grid=30000,power_consumption_trend=self.power_consumption_trend)

        # Create the waiting room
        self.waiting_room = sim.Queue(name="waitingline88", monitor=False)
        self.waiting_room.length_of_stay.monitor(value=True)
        self.waiting_room.length_of_stay.reset_monitors(stats_only=True)

        # Create the charing stations
        self.stations = [
            ChargingStation(
                waiting_room=self.waiting_room,
                env=self.env_sim,
                power_supply=self.power_supply_o,
                max_power_delivery=200000,
                diffrence_desired= self.difference_times
            )
            for _ in range(self.charging_stations)
        ]
        #Set up the charging pole in use list
        for _ in range(self.charging_stations):
            self.poles_active.append(False)

        # Create the EV generator
        self.generator = CustomerGenerator(
            waiting_room=self.waiting_room,
            env=self.env_sim,
            clerks=self.stations,
            wait_times=self.wait_times,
            total_times= self.total_times,
            time_before_service=self.time_before_service,
            shedual=self.shedual.trucks,
            desired_times=self.desired_times,
            difference_desired=self.difference_times
        )

    # This function runs the simmulation
    def run_sim(self) -> Tuple[int, int, int]:
        # Create random numbers for the max power supplys
        self.power_supply_o.distribution_rl = [1, 1, 1]
        self.power_supply_o.strategy = 2
        # Start the simmulation
        self.env_sim.run(till=self.total_time)
        while len(self.waiting_room) != 0:
            #print("empty")
            temp =self.waiting_room.pop()
            temp.in_loop = False

        # Get the output of the simmulation
        avg = sum(self.wait_times) / len(self.wait_times)
        min_o = min(self.wait_times)
        max_o = max(self.wait_times)
        self.old_time += self.total_time
        return avg, int(min_o), int(max_o)
    
    #This method is used to rerun the simmulation (not used for RL)        
    def rerun(self):
        while len(self.waiting_room) != 0:
            #print("empty")
            temp =self.waiting_room.pop()
            temp.in_loop = False
        #print(self.env_sim.get_time_unit())
        self.wait_times.clear()
        self.waiting_room.clear()
        self.env_sim.reset_now()
        self.shedual.prepare_data(spread_type=self.spread_type)
        self.generator.shedual = self.shedual.trucks
        self.generator.reset()
        self.env_sim.run(till=self.total_time)
        #self.generator.activate()
        #print(len(self.wait_times),"length")
        avg = sum(self.wait_times) / len(self.wait_times)
        min_o = min(self.wait_times)
        max_o = max(self.wait_times)
        self.old_time += self.total_time
        self.__get_waitingline_data()
        return avg, int(min_o), int(max_o)

 #-------------------------------------------------------------------------------   
    def rl_Run(self,power_input): 
        #Set the charging strategy to RL
        self.power_supply_o.strategy = 2
        #Input the power form the RL model
        self.power_supply_o.distribution_rl = power_input
        #Run the simmulation for 1 time unit
        self.env_sim.run(till=self.old_time + 1)
        self.old_time += 1
        #Calculate the total chrage request from vehicles that are waiting in the waiting line
        charge_request = 0
        if len(self.waiting_room) > 0:
            for i in self.waiting_room:
                charge_request += 100 - i.battery_charge
        else:
            charge_request = 0          
        #Get the data from the simmulation
        sim_data = self.__get_env_Data__()
        #Return data to the RL model (diffrence is in the reset)
        if self.old_time >= self.total_time:
            return True,sim_data
        else:
            return False,sim_data
        
    #This resets the current simmulation back to time 0
    def rl_reset(self):
        self.old_time = 0
        #self.__get_waitingline_data() 
        while len(self.waiting_room) != 0:
                #print("empty")
            temp =self.waiting_room.pop()
            temp.in_loop = False

        #Clear all the data from the lists (to start with a clean simmulation)
        self.wait_times.clear()
        self.waiting_room.clear()
        self.total_times.clear()
        self.difference_times.clear()
        #Reset the enviroment to time unit 0
        self.env_sim.reset_now()
        #Prepare a new shedual
        self.shedual.prepare_data(spread_type=self.spread_type)
        self.generator.shedual = self.shedual.trucks
        #Reset the vehicle generator
        self.generator.reset()
        #Return the measured state of the enviroment
        #Get the data from the simmulation
        sim_data = self.__get_env_Data__()
        return sim_data
#-------------------------------------------------------------------------------
    #This method is used to extract the waiting times from the system
    def __get_waiting_data__(self):
        #Calculate the avarge waiting time before charging begins
        if len(self.wait_times) != 0: 
            avg = sum(self.wait_times) / len(self.wait_times)
        else:
            avg = 0
        #Check if there is a least 1 data point in the list, to avoid devideing by 0
        if len(self.total_times):
            avg_tot = sum(self.total_times) / len(self.total_times)
            min_tot = min(self.total_times)
            max_tot = max(self.wait_times)
            max_diff = max(self.difference_times)
            min_dif = min(self.difference_times)  
            #print(self.difference_times)
        else:
            avg_tot = 0
            min_tot = 0
            max_tot = 0
            max_diff = 0
            min_dif = 0     
        #Return the calculated data
        return avg_tot,min_tot,max_tot,max_diff,min_dif,avg   

    #This metohd detects which charging poles are being used
    def __get_pole_data__(self):
        i = 0
        #loop through all the charging poles
        for pole in self.stations:
            if pole.ispassive():
                self.poles_active[i] = False
            else:
                self.poles_active[i] = True
            i +=1

    #This method gets the data from the waiting line
    def __get_wait_line_data__(self):
        total_charge_request = 0
        #loop through the waiting line
        if len(self.waiting_room) != 0:
            for i in self.waiting_room:
                total_charge_request += (100 - i.battery_charge)
        return total_charge_request

    #This method is used to combine all the data from the simmulation 
    def __get_env_Data__(self):
        #Call the all the getter methods
        wait_data = self.__get_waiting_data__()
        self.__get_pole_data__()
        total_charge_request = self.__get_wait_line_data__()
        #Create a dictonary with all the data that commes from the dsimmulation
        sim_values={
            'avg_tot': wait_data[0], 
            'min_tot': wait_data[1],
            'max_tot': wait_data[2],
            'max_diff': wait_data[3],
            'min_diff': wait_data[4],
            'avg_wait': wait_data[5],
            'pole_data': self.poles_active, 
            'Charge_Request' : total_charge_request            
        }
        #Return the dictonary
        return sim_values
#-------------------------------------------------------------------------------
    def reset_shedual(
        self,
    ):  # This method resets the complete simmulation to its starting position
        self.env_sim.reset()
        self.waiting_room.clear()
        self.env_sim.reset_now(0)
        self.first = True
        self.env_sim = sim.App(
            trace=False,
            random_seed="*",
            name="Simmulation",
            do_reset=True,
            yieldless=True,
        )
        # Make sure all the charging stations are disabled
        self.shedual.prepare_data(spread_type=self.spread_type)
        self.generator.shedual = self.shedual.trucks
#-------------------------------------------------------------------------------
    #This method plots the power consumption during the day
    def plot_consumption(self):
        x = []
        j = 1
        print(type(self.power_consumption_trend))
        for i in self.power_consumption_trend:
            x.append(j)
            j += 1

        X_Y_Spline = make_interp_spline(x, self.power_consumption_trend)
        X_ = np.linspace(0, j, 50)
        Y_ = X_Y_Spline(X_)

        plt.plot(X_,Y_)
        plt.title("Plot Smooth Curve Using the scipy.interpolate.make_interp_spline() Class")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()