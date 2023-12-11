from dataclasses import dataclass
import numpy as np


@dataclass
class Truck:
    """Dataclass that holds the information regarding a truck"""
    battery: np.int16
    arrival_time: np.int16
    total_time: np.int16
    total_wait_time: np.int16
    desired_wait_time: np.int16
    

@dataclass
class Consumption:
    """Truck that hold charging station information"""

    power_consumption: np.real  # Current consumption
    max_power_consumption: np.real  # Max consumption from the
    max_power_request: np.real  # Max power the charging station is able to get
