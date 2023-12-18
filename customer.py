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
        self.mode.monitor(False)
        self.status.monitor(False)
        self.desired_wait_time = 0
        self.total_time = 0
        self.number = number
        self.in_loop = True

    def process(self):
        #print("Process starts = ", self.number)
        # Put the vehicle in the waiting room
        self.enter(self.waiting_room)
        #print("Enter ")
        # Check if there is a station that is passive
        for station in self.stations:
            if station.ispassive():
                station.activate()
                #print("Activate_Station")
                break  # activate at most one clerk
        #print("All stations active")
        self.passivate()
