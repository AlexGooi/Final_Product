from sim_manager import SimManager
from customer import Customer
import gc
import salabim as sim

array = [1,400.0,400.0,20.0,20.0,20,20,20,20,20,5,5,5,5,5,5,5,5]
if __name__ == "__main__":
    man = SimManager(1, 1400,spread_type=3)
    for i in range(200000):        
        done = False
        while done == False:
            
            done,sim_data = man.rl_Run(array)
        
        data,lent = man.rl_reset()
        print(data['avg_wait'],lent)
            #print(sim_data['Charge_Request'])
    #man.plot_consumption()
    
