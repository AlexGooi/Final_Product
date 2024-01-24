from typing import List
import numpy as np
import salabim as sim


class Customer(sim.Component):
    def __init__(
        self,
        waiting_room: sim.Queue,
        env: sim.App,
        stations,
        wait_times: List,
        desired_times: List,
        charge_percentage :List,
        time_before_service: List,
        diffrence_desired: List,
        sat: List,
        battery_charge: np.int16,
        desired_battery : np.int16,
        max_wait_time,
        creation_time,
        number,
        total_times
    ):
        super().__init__(name="Truck")
        self.waiting_room = waiting_room
        self.env = env
        self.stations = stations
        self.battery_charge = battery_charge
        self.creation_time = creation_time
        self.wait_times = wait_times
        self.total_times = total_times
        self.desired_times = desired_times
        self.desired_battery = desired_battery
        self.max_wait_time = max_wait_time
        self.charge_percentage = charge_percentage
        self.time_before_service = time_before_service
        self.diffrence_desired = diffrence_desired
        self.sat = sat
        self.mode.monitor(False)
        self.status.monitor(False)
        self.desired_wait_time = 0
        self.total_time = 0
        self.number = number
        self.in_loop = True

    def process(self):
        #print("Process starts = ", self.number)
        # Put the vehicle in the waiting room
        enterd = False
        checked = False
        while checked == False:
            try:
                #Only 1 vehicle is allowed waiting in the waiting room
                if len(self.waiting_room) == 0:
                    self.enter(self.waiting_room)
                    enterd = True
                else:
                    #print("leaving")
                    pass
                checked = True
            #When a new round has been started a vehicle could be stuck in the waiting line, when this is the case clear the waiting line
            except Exception as e:
                print("Exception = ",e)
                #Clear the waiting 
                while len(self.waiting_room) != 0:
                    #print("empty")
                    temp =self.waiting_room.pop()
                    self.hold(1)
        #print("Enter ")
        # Check if there is a station that is passive
        if enterd:                    
            for station in self.stations:
                if station.ispassive():
                    station.activate()
                    #print("Activate_Station")
                    break  # activate at most one clerk
        #print("All stations active")
        #If no charging station is found, wait in the waiting line
        self.passivate()
