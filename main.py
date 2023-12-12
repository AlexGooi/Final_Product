from sim_manager import SimManager
from customer import Customer
import gc
import salabim as sim

array = [20,400.0,400.0,20.0,20.0,20,20,20,20,20,5,5,5,5,5,5,5,5]
if __name__ == "__main__":
    man = SimManager(3, 14,spread_type=5)
    for i in range(1):        
        done = False
        while done == False:
            done,sim_data = man.rl_Run(array)
            #print(sim_data['Charge_Request'])
    man.plot_consumption()
    man.rl_reset()

