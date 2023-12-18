import salabim as sim

from limit import limit


# This class resables the general power supply that the chraging stations are coupled to
class PowerSupply(sim.Component):
    def __init__(self, env, max_power_from_grid,power_consumption_trend):
        super().__init__(name="Supply")
        self.max_power_from_grid = max_power_from_grid
        self.power_used_list = []
        self.distribution_rl = []
        self.power_consumption_trend = power_consumption_trend
        self.power_used = 0
        self.Total_Used = 0
        self.percentage_used = 0
        self.env = env
        self.strategy = 0
        self.max_reached = False
        self.mode.monitor(False)
        self.status.monitor(False)
        self.total_distributed = 0

    def process(self):
        # Calculate the amount of energy that is currently being used
        while True:
            #print(self.power_used_list)
            """TODO variable total is unused"""
            total = 0
            # Select the charging strategy
            if self.strategy == 0:
                self.__distribute_power_simple()
            elif self.strategy == 1:
                self.__disrtibute_power_share_equal()
            elif self.strategy == 2:
                self.__distribute_power_rl(rl_distribution=self.distribution_rl)
            # Check if the list has members
            if len(self.power_used_list) != 0:
                self.Total_Used =0
                # Loop through all the charging stations
                for i in self.power_used_list:
                    self.Total_Used += i.power_consumption
                self.power_consumption_trend.append(self.Total_Used)
                #Get the total useage of the power supply
                self.__calculate_usage__()
                self.hold(1)
        print("Process_Stop")

    def __distribute_power_simple(self):
        """This method resembles the simplest distribution (give max until it is out)"""

        # Loop through all the power cinsumers
        total_distributed = 0
        
        for i in self.power_used_list:
            # Calculate the max distribution left
            max_allowed = limit(
                0,
                self.max_power_from_grid - total_distributed,
                self.max_power_from_grid,
            )
            i.max_power_consumption = limit(0, i.max_power_request, max_allowed)
            total_distributed += i.max_power_consumption

    def __disrtibute_power_share_equal(self):
        """This method resables a equal share to all the charging stations"""

        # Loop through all the power consumers
        """# TODO variable total_distributed is unused"""
        total_distributed = 0
        if len(self.power_used_list) != 0:
            available_per_station = self.max_power_from_grid / len(self.power_used_list)
            for i in self.power_used_list:  # Calculate the total amount
                # Give the allowed power to the stations
                i.max_power_consumption = limit(
                    0, i.max_power_request, available_per_station
                )

    def __distribute_power_rl(
        self, rl_distribution
    ):  # This method is used to distribute the power with the help of reinforcemnt learning
        self.total_distributed = 0
        counter = 0
        #print("Lenght =", len(self.power_used_list))
        if len(self.power_used_list) != 0:
            for i in self.power_used_list:
                
                max_allowed = limit(
                    0,
                    self.max_power_from_grid -  self.total_distributed,
                    self.max_power_from_grid,
                )
                max_allowed = limit(0, max_allowed, i.max_power_request)
                max_allowed = limit(
                    0,
                    max_allowed,
                    limit(
                        0,
                        self.max_power_from_grid -  self.total_distributed,
                        self.max_power_from_grid -  self.total_distributed,
                    ),                    
                )
                # Note to the system when the maximum energy consumption is reached
                if max_allowed == 0:
                    self.max_reached = True
                else:
                    self.max_reached = False
                # Insert the max power consumption from the reinforcement learning model into
                
                i.max_power_consumption = limit(0, rl_distribution[counter], max_allowed)
                #print(i.max_power_consumption, "Allowed")
                self.total_distributed += i.max_power_consumption
                counter += 1

    #Calculate the percantage of tha avialable power used from the system
    def __calculate_usage__(self):
        percentage = (self.Total_Used / self.max_power_from_grid) * 100
        self.percentage_used = limit(0,percentage,100 )
