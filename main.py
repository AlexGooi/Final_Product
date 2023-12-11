from sim_manager import SimManager
from customer import Customer
import gc
import salabim as sim

array = [1,20.0,20.0,20.0,20.0,20,20,20,20,20,5,5,5,5,5,5,5,5]
if __name__ == "__main__":
    man = SimManager(1, 14000,spread_type=3)
    for i in range(1):        
        done = False
        while done == False:
            done,sim_data = man.rl_Run(array)
            print(sim_data['avg_wait'])
    man.plot_consumption()
    print(man.rl_reset())

