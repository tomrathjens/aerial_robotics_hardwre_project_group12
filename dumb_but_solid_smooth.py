# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2014 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Simple example that connects to the first Crazyflie found, logs the Stabilizer
and prints it to the console. 

The Crazyflie is controlled using the commander interface 

Press q to Kill the drone in case of emergency

After 50s the application disconnects and exits.
"""
import logging
import time
from threading import Timer
import threading
import numpy as np

from pynput import keyboard # Import the keyboard module for key press detection
from scipy.interpolate import splprep, splev

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

# # TODO: CHANGE THIS URI TO YOUR CRAZYFLIE & YOUR RADIO CHANNEL
uri = uri_helper.uri_from_env(default='radio://0/20/2M/E7E7E7E712')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


####################################

################parameters to fill in ############
nb_laps = 1 #number of laps to do
nb_points = 100 #to adjust so that the space between points is around 0.1 m
time_bwn_points = 0.1   #time to wait between points of the path in seconds
gate1 = [0.6, -0.43, 0.79, 0]  # x, y, z, yaw coordinates of the first gate relative to the starting point of the drone
gate2 = [2.09, 0.24, 1.27, 0]
gate3 = [0.09, 0.91, 1.17, 0]
gate4 = [-0.77, 0.23, 1.23, 0]

gates_in_order = [gate1, gate2, gate3, gate4]
after_take_off = [0,0,height_takeoff,0]

while nb_laps > 1:
    gates_in_order = gates_in_order + gates_in_order #repeat the gates in order nb_laps times
    nb_laps -= 1

# add a start and an end to the path
gates_in_order = [after_take_off] + gates_in_order #add the take off position at the beginning of the path
gates_in_order = gates_in_order + [after_take_off] #add the take off position at the end of the path

# Create multiple points between the gates to make the path smoother and equidistant
gates_in_order = np.array(gates_in_order)
x, y, z = gates_in_order[:, 0], gates_in_order[:, 1], gates_in_order[:, 2]

# Use B-spline to create a smooth path
tck, u = splprep([x, y, z], s=0.0)  # `s` is the smoothing factor; increase for smoother curves
u_fine = np.linspace(0, 1, 2000)  # Generate a dense set of points
x_smooth, y_smooth, z_smooth = splev(u_fine, tck)

# Calculate cumulative distances along the path
distances = np.cumsum(np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2 + np.diff(z_smooth)**2))
distances = np.insert(distances, 0, 0)  # Add the starting point

# Interpolate to get equidistant points
equidistant_distances = np.linspace(0, distances[-1], nb_points)
x_equidistant = np.interp(equidistant_distances, distances, x_smooth)
y_equidistant = np.interp(equidistant_distances, distances, y_smooth)
z_equidistant = np.interp(equidistant_distances, distances, z_smooth)

# Create waypoints with yaw set to 0
waypoints = [[x_equidistant[i], y_equidistant[i], z_equidistant[i], 0] for i in range(len(x_equidistant))]


# Check distacne between waypoints
if len(waypoints) > 1:
                total_distance = 0
                for i in range(len(waypoints) - 1):
                    point_a = np.array(waypoints[i][:3])
                    point_b = np.array(waypoints[i + 1][:3])
                    total_distance += np.linalg.norm(point_b - point_a)
                average_distance = total_distance / (len(waypoints) - 1)
                print(f"Average distance between points: {average_distance:.3f} m")



#######################################
# Fuctions 
#######################################
def get_next_waypoint(waypoints, pose_reached, current_position):
    """
    Dynamically updates the waypoints list by removing reached points and returning the next waypoint.
    
    Args:
        waypoints: List of waypoints [x, y, z, yaw].
        pose_reached: Distance threshold to consider a waypoint as reached.
        current_position: Current position of the drone [x, y, z].
    
    Returns:
        next_waypoint: The next waypoint to move to [x, y, z, yaw].
    """
    if not waypoints:
        return None  # No waypoints left

    # Check if the current waypoint is reached
    distance_to_waypoint = np.linalg.norm(np.array(current_position[:3]) - np.array(waypoints[0][:3]))
    if distance_to_waypoint < pose_reached:
        waypoints.pop(0)  # Remove the reached waypoint

    # Return the next waypoint if available
    return waypoints[0] if waypoints else None


#################################

class LoggingExample:
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 10s.
    """

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """

        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=50)
        self._lg_stab.add_variable('stateEstimate.x', 'float')
        self._lg_stab.add_variable('stateEstimate.y', 'float')
        self._lg_stab.add_variable('stateEstimate.z', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')

        # The fetch-as argument can be set to FP16 to save space in the log packet
        # self._lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        # Start a timer to disconnect in 50s     TODO: CHANGE THIS TO YOUR NEEDS
        t = Timer(50, self._cf.close_link)
        t.start()

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""

        # Print the data to the console
        # print(f'[{timestamp}][{logconf.name}]: ', end='')
        # for name, value in data.items():
        #     print(f'{name}: {value:3.3f} ', end='')
        # print()

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.is_connected = False


# Define your custom callback function
def emergency_stop_callback(cf):
    def on_press(key):
        try:
            if key.char == 'q':  # Check if the "space" key is pressed
                print("Emergency stop triggered!")
                cf.commander.send_stop_setpoint()  # Stop the Crazyflie
                cf.close_link()  # Close the link to the Crazyflie
                return False     # Stop the listener
        except AttributeError:
            pass

    # Start listening for key presses
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    le = LoggingExample(uri)
    cf = le._cf

    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

    # Replace the thread creation with the updated function
    emergency_stop_thread = threading.Thread(target=emergency_stop_callback, args=(cf,))
    emergency_stop_thread.start()

    # TODO : CHANGE THIS TO YOUR NEEDS
    print("Starting control")
    while le.is_connected:
        time.sleep(0.01)
        print("Sending commands")
        # Take-off

        ticks = int(time_takeoff / 0.1)  # Number of ticks to wait between points
        magic_number = ticks / height_takeoff

        # ex : takeoff in 5 seconds to 0.8m 
        # 5 seconds = 50 ticks
        # magic_number = 50 / 0.8 = 62.5

        for y in range(ticks):
            cf.commander.send_hover_setpoint(0, 0, 0, y / magic_number)
            time.sleep(0.1)
        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, height_takeoff)
            time.sleep(0.1)

        # Move qqq
        # Bullshit function usefeless
        # for _ in range(50):
        #     cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
        #     time.sleep(0.1)
        # for _ in range(50):
        #     cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
        #     time.sleep(0.1)

        # # Move like OG

        while waypoints:
            # Get the current position of the drone from the log data
            log_data = le._lg_stab.data_received_cb  # Example of accessing log data
            current_position = [
                log_data['stateEstimate.x'],
                log_data['stateEstimate.y'],
                log_data['stateEstimate.z']
            ]

            # Get the next waypoint
            next_waypoint = get_next_waypoint(waypoints, pose_reached, current_position)

            if next_waypoint:
                x, y, z, yaw = next_waypoint
                #ticks = int(time_bwn_points / 0.1)  # Number of ticks to wait between points

                #print("Moving to waypoint:", (x, y, z))

                #for _ in range(ticks):
                cf.commander.send_position_setpoint(x, y, z, yaw)
                    #time.sleep(0.1)
            else:
                print("All waypoints reached.")
                break
            





        # for _ in range(20):
        #     cf.commander.send_position_setpoint(0, 0.1 , 0.4, 0)
        #     time.sleep(0.1)

        # for _ in range(20):
        #     cf.commander.send_position_setpoint(0.1, 0.1 , 0.4, 0)
        #     time.sleep(0.1)

        # for _ in range(20):
        #     cf.commander.send_position_setpoint(0.1, -0.1 , 0.4, 0)
        #     time.sleep(0.1)

        # for _ in range(20):
        #     cf.commander.send_position_setpoint(0, -0.1 , 0.4, 0)
        #     time.sleep(0.1)

        # Land
        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, height_takeoff)
            time.sleep(0.1)

        ticks = int(time_takeoff / 0.1)  # Number of ticks to wait between points
        magic_number = ticks / height_takeoff

        for y in range(ticks):
            cf.commander.send_hover_setpoint(0, 0, 0, (ticks - y) / magic_number)
            time.sleep(0.1)

  

        cf.commander.send_stop_setpoint()
        break