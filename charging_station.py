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
        self.time_remaining = 0
        self.first_test = False
        self.charging = False
        self.reset = False
        self.number = number
        # Append the power consumption to the consumtion list
        self.power_supply.power_used_list.append(self.power_consumption)
        self.power_consumption.max_power_request = self.max_power_delivery
        self.hold_back = False
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
            self.charge_car(bat_sim= True,clear_station= False)
            #print("Done chargig")
            self.vehicle.in_loop = False

    # This method charges car and stops when the car has been charged
    def charge_car(self,bat_sim, clear_station):
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
           # print(self.vehicle.max_wait_time)       
            while (self.vehicle.battery_charge < self.vehicle.desired_battery * 10) and time_charging < self.vehicle.max_wait_time :
                #Calculate the remaining charge time
                self.time_remaining = limit(0,self.vehicle.max_wait_time - time_charging,self.vehicle.max_wait_time)
                #Determine the charging time
                time_charging = self.env.now() - start_charging
                #print( self.vehicle.max_wait_time - time_charging)
                if self.vehicle.battery_charge < 1000 - 1:    
                    #print(self.power_consumption.max_power_consumption)
                    add_percent = self.__calculate_charge_percentage(self.power_consumption.max_power_consumption) * 10
                    #print(add_percent)
                    add_charge = limit(0,add_percent, 1000 - self.vehicle.battery_charge)
                    #Limit the charge when the battery is higher then 80 %
                    if bat_sim:
                        #Limit the charge when the battery is above 80%
                        if self.vehicle.battery_charge >= 800:
                            self.hold_back = True
                            self.charge_factor = scale_value(input_value= self.vehicle.battery_charge, input_lower= 1000, input_upper= 800, output_lower= 0.1,output_upper= 1.0)
                            #factor = scale_value(input = self.vehicle.battery_charge, input_lower= 800, input_upper= 1000, output_lower= 1.0,output_upper= 0.1)
                            add_charge = add_charge * self.charge_factor
                            if self.first_test == False and self.vehicle.battery_charge >= 990:
                                #print("almost full",time_charging)
                                self.first_test = True
                        else:
                            self.hold_back = False
                            self.charge_factor = 1
                            self.first_test = False
                            #print("Last 80 %")

                    #print(add_charge,"Add Charge")  
                else:
                    #print("100%! Time aited = ",time_charging)
                    #print("last cycle")
                    add_percent = self.__calculate_charge_percentage(self.power_consumption.max_power_consumption) * 10
                    add_charge = limit(
                        0,
                        add_percent,
                        1000 - self.vehicle.battery_charge
                    )
                    add_charge = 0
                self.hold(1)
                    #print("add",self.vehicle.battery_charge+ add_charge)
                    #Check if the car is at its limit charging time
                #if time_charging >= self.vehicle.max_wait_time:
                    #print("Leaving early")      

                # Hold the simulation for 1 minute
                if self.vehicle.battery_charge + add_charge < 999:
                    self.vehicle.battery_charge += add_charge
                else:
                    self.vehicle.battery_charge = 999
                    
                #self.power_consumption.power_consumption = add_charge
                loop += 1
                #print(add_charge)
                add_KWH = self.__calculate_power_kwh(add_charge / 10)
                #print("in", self.power_consumption.max_power_consumption, "out", add_KWH)
                self.power_consumption.power_consumption = add_KWH
                if clear_station:
                    time_charging = self.vehicle.max_wait_time
                    #print("clear")
                    
            
            #Set the total waiting time of the customber
            self.vehicle.total_time =(self.env.now() - self.vehicle.creation_time)
            #Append the total time to the loop
            #if self.vehicle.total_time < 0 and self.reset: 
            self.reset = False
            if self.vehicle.total_time >= 0: 
                self.vehicle.total_times.append(self.vehicle.total_time)
                #Append the diffrence of the actual wait time and the desired wait time     
                self.vehicle.diffrence_desired.append(self.vehicle.total_time - self.vehicle.desired_wait_time)       
                #Calculate the amount of charge the vehicle has been given during charging
                charging_percentage = limit(0,(self.vehicle.battery_charge / (self.vehicle.desired_battery *10)) * 100, 100)
                self.vehicle.charge_percentage.append(charging_percentage)
                self.vehicle.sat.append(charging_percentage)


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
                    self.charge_car(bat_sim= False,clear_station= True)
                    self.vehicle.in_loop = False
            except Exception as e:
                #print("Exception!!!",e)
                pass
            self.first = True
            #print("passivate")
            
        except Exception as e:
            #print("jj",e)
            pass
        self.passivate()
        return False
    

    def __calculate_charge_percentage(self,power_kwh):
        # Convert power from kW/h to kW/m
        power_kwm = power_kwh / 60.0

        # Calculate the added charge to the battery in percentage
        added_charge_percentage = (power_kwm / 70) * 100

        return added_charge_percentage
    
    def __calculate_power_kwh(self,added_charge_percentage):
        # Convert the added charge percentage to power in kW/m
        power_kwm = (added_charge_percentage / 100) * 70

        # Convert power from kW/m to kW/h
        power_kwh = power_kwm * 60

        return power_kwh

print(scale_value(input_value=98,input_lower=98,input_upper= 93, output_lower=0,output_upper = 8))