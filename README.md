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


## usefull links : 
- Commander methods\
https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/commander/

- State estimates variables\
 https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate




 ## video 15.05 gates coordinates :
gate1 = [0.6, -0.32, 0.75, 0]  # x, y, z, yaw coordinates of the first gate relative to the starting point of the drone
gate2 = [2.09, 0.25, 1.29, 0]
gate3 = [0.11, 0.93, 1.16, 0]
gate4 = [-0.79, 0.4, 1.27, 0]

 