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
        time_before_service: List,
        diffrence_desired: List,
        battery_charge: np.int16,
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
        # Check if there is a station that is passive
        for station in self.stations:
            if station.ispassive():
                station.activate()
                break  # activate at most one clerk
        self.passivate()
