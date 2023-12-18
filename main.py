from sim_manager import SimManager
from customer import Customer
import gc
import salabim as sim
from limit import scale_value


array = [0.5,0.5,5,5,6,6,6,6,6,6,5,5,5,5,5,5,5,5]
if __name__ == "__main__":
    man = SimManager(2, 2400,spread_type=5,grid_supply=15)
    #for i in man.shedual.trucks:
        #print(i.arrival_time)
    #print(man.shedual.trucks)
    for i in range(8):       
        done = False
        #Make sure there is at least 1 car through the systme
        man.loop_first_car(array)
        while done == False:            
            done,rl_data = man.rl_Run(array)
            #print("loop", i)    
            #print("Wait",len(man.waiting_room))        
            #print(len(man.shedual.trucks))
            #print(sim_data['Percentage_Used'])
            #print(sim_data['Avg_ChargePpercentage'])
            print(rl_data['Avg_ChargePpercentage'],rl_data['Percentage_Used'])
        data,lent = man.rl_reset()
        #print("Charge_Percentage", data['Avg_ChargePpercentage'])
        #print(data['avg_wait'],lent)
            #print(sim_data['Charge_Request'])


    man.plot_consumption()

    
