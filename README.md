# aerial_robotics_hardwre_project_group12



## rmqs : 
- Pressing emergency stop or q, you will need to reboot the drone

- make sure to place the drone with yaw=0, i.e. facing x positive

- time between point >= 0.1 otherwise value in for loop < 1 => not working

- don't forget to recalibrate (be familiar with procedure)


## to do

- tricky gates poses (big height difference and close)

- 2nd lap -> improve speed -> filter -> velocity control

- nonlinear interpolation (waypoints), i.e. more points in tricky passages and less points elsewhere

- take into account theta of gates when planning trajectory -> to avoid colliding into gates' frame

## presentation

- length <= 4 slides

- duration = 7 min

- content :

    - 1 slide on the gate setup for your experiment (Show environment layout(s) tested on)

    - 1 slide on the strategy (Algorithm, what you spend most time on)

    - 1 slide on the results (Statistics on mission time/success/…)

    - 1 optional slide (with anything relevant to add)


## usefull links : 
- Commander methods\
https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/commander/

- State estimates variables\
 https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate
 