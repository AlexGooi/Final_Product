from sim_manager import SimManager
from customer import Customer
import gc
import salabim as sim
from limit import scale_value

array = []
#Share the charge equally over the charging poles
for i in range(20):
    array.append(3.5)

if __name__ == "__main__":
    #Create the simmanger object (object to run the simmulation)
    man = SimManager(20, 2400,spread_type=6,grid_supply=70)
    #for i in man.shedual.trucks:
        #print(i.arrival_time)
    #print(man.shedual.trucks)c
    for i in range(1):       
        done = False
        #Make sure there is at least 1 car through the system, eliminating the simulation startup
        man.loop_first_car(array)
        while done == False:           
            #Run the simulation for 1 cycle 
            done,rl_data = man.rl_Run(array)
            amount_1 = 0
            #Passivate the charging stations
            for station in man.stations:
                if station.ispassive():
                    amount_1 +=1
            #print("Amount of passive stations", amount_1)
            #print("loop", i)    
            #print("Wait",len(man.waiting_room))        
            #print(len(man.shedual.trucks))
            #print(rl_data['Poles_Battery_Levels'])
            #print(sim_data['Avg_ChargePpercentage'])
        #Print the end result
        print(rl_data['Avg_ChargePpercentage'],rl_data['Percentage_Used'])
        data,lent = man.rl_reset()
        print("Amount of passive stations", amount_1)
            
        print("Charge_Percentage", data['Avg_ChargePpercentage'])
        #print(data['avg_wait'],lent)
            #print(sim_data['Charge_Request'])
    #Plot the results
    man.plot_max_energy_usage(rl_data)


    