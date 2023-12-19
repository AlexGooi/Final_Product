import random
import time
import salabim as sim
from data import Consumption
from limit import limit
from limit import scale_value

from power_supply import PowerSupply
from customer import Customer

class ChargingStation(sim.Component):
    def __init__(
        self,
        waiting_room: sim.Queue,
        env: sim.App,
        power_supply: PowerSupply,
        max_power_delivery: int,
        diffrence_desired : list,
        number
    ):
        super().__init__(name="Station")
        random.seed(time.time())
        self.waiting_room = waiting_room
        self.vehicle = 0
        self.power_supply = power_supply
        self.env = env
        self.max_power_delivery = max_power_delivery
        self.power_consumption = Consumption(0, 0, 0)
        self.diffrence_desired = diffrence_desired
        self.charge_factor = 1
        self.charging_percentage = 100
        self.charging = False
        self.reset = False
        self.number = number
        # Append the power consumption to the consumtion list
        self.power_supply.power_used_list.append(self.power_consumption)
        self.power_consumption.max_power_request = self.max_power_delivery
        self.first = True
        self.mode.monitor(False)
        self.status.monitor(False)   
        self.empty = False     

    def process(self):
        while True:
            #print("Pole_Number",self.number)
            # Continu looping until a vehicle shows up in the waiting line
            while len(self.waiting_room) == 0:
                self.passivate()
            self.vehicle = self.waiting_room.pop()
            #print(self.vehicle.battery_charge,"Battery charge")
            self.charge_car(bat_sim= True)
            #print("Done chargig")
            self.vehicle.in_loop = False

    # This method charges car and stops when the car has been charged
    def charge_car(self,bat_sim):
        loop = 0
        add_charge = 0
        self.vehicle.wait_times.append(self.env.now() - self.vehicle.creation_time)
        #Notify the system that the pole is charging

        #((self.vehicle.creation_time == 0  and self.env.now() == 0) or self.first == False) and 
        if self.vehicle.in_loop:
            self.first = False
            self.charging = True
            #Get the starting time of the charging process
            start_charging = self.env.now()
            time_charging = 0
            #print(self.vehicle.battery_charge,"Battery Charge")
            while (self.vehicle.battery_charge < self.vehicle.desired_battery * 10) and time_charging < self.vehicle.max_wait_time :
                #Determine the charging time
                time_charging = self.env.now() - start_charging   
                if self.vehicle.battery_charge <= 1000 - 1:   
                    
                    add_charge = limit(0,self.power_consumption.max_power_consumption, 1000 - self.vehicle.battery_charge)
                    #Limit the charge when the battery is higher then 80 %
                    if bat_sim:
                        #Limit the charge when the battery is above 80%
                        if self.vehicle.battery_charge >= 800:
                        
                            self.charge_factor = scale_value(input_value= self.vehicle.battery_charge, input_lower= 1000, input_upper= 800, output_lower= 0.1,output_upper= 1.0)
                            #factor = scale_value(input = self.vehicle.battery_charge, input_lower= 800, input_upper= 1000, output_lower= 1.0,output_upper= 0.1)
                            add_charge = add_charge * self.charge_factor
                        else:
                            self.charge_factor = 1
                            #print("Last 80 %")

                    #print(add_charge,"Add Charge")  
                    self.hold(1)
                else:
                    
                    add_charge = limit(
                        0,
                        self.power_consumption.max_power_consumption,
                        1000 - self.vehicle.battery_charge,
                    )
                    self.hold(add_charge / 10)
                    #print("add",self.vehicle.battery_charge+ add_charge)
                # Note to the power supply much power is being used from it
                

                # Hold the simulation for 1 minute
                self.vehicle.battery_charge += add_charge
                #self.power_consumption.power_consumption = add_charge
                loop += 1
                #print(add_charge)
                self.power_consumption.power_consumption = add_charge
            #Set the total waiting time of the customber
            self.vehicle.total_time =(self.env.now() - self.vehicle.creation_time)
            #Append the total time to the loop
            #if self.vehicle.total_time < 0 and self.reset: 
            #    print("WTFFFDF", self.env.now(), self.vehicle.creation_time)
            self.reset = False
            if self.vehicle.total_time >= 0: 
                self.vehicle.total_times.append(self.vehicle.total_time)
                #Append the diffrence of the actual wait time and the desired wait time     
                self.vehicle.diffrence_desired.append(self.vehicle.total_time - self.vehicle.desired_wait_time)       
                #Calculate the amount of charge the vehicle has been given during charging
                charging_percentage = limit(0,(self.vehicle.battery_charge / (self.vehicle.desired_battery *10)) * 100, 100)
                self.vehicle.charge_percentage.append(charging_percentage)



            #Let the system know the pole has stopped charging
            self.charging = False
        return loop

    # This method calculates the maximum amount of charge the charging pole is allowed to give
    def max_power_consumption(self):
        # Calculate the total amount of power already used by the charging stations
        """TODO power_user is unused"""
        power_used = 0
        for i in self.power_supply.power_used:
            power_used += i

    #This method makes sure the station is clear of trucks
    def clear_station(self): 
        
        try:
            self.vehicle.battery_charge = 999
            self.reset = True            
            try:
                if self.vehicle.battery_charge < 1000:
                    self.vehicle.battery_charge = 1000
                    self.charge_car(bat_sim= False)
                    self.vehicle.in_loop = False
            except Exception as e:
                print("Exception!!!",e)
                pass
            self.first = True
            print("passivate")
            
        except Exception as e:
            print("Exception!!!",e)
        self.passivate()
        return False
