To use this project, please follow these steps

1:
Create the enviroment from the tetris_charge.yml file (we used mamba for this)
2:
The project can be run into 3 ways (uncontrolled, FCFS, Equal share, RL
The configuration in this repository is supporting Eqaul share and RL out of the box

Equal share:
Run the main.py file

RL:
Run the truck_env.py file

If you want to run the other systems please some code needs to be changed

Uncontrolled:
In power_supply.py navigate to row 107 and change max allowd by 100
In Main.py naviage to row 10 and change 3.5 to 7
After this run the main again

FCFS:
If not already done change row 107 power_supply back to the original
In Main.py naviage to row 10 and change 3.5 to 7
After this run the main again

If you want to see RL in the uncapped version (also featured in the report) do the following:
In power_supply.py navigate to row 107 and change max allowd by 100
And run truck_env.py
This configuration shows the potential of RL the mode sindce it controles to 70 KW for most of the time.
Note that it is allowed to go much higher.

