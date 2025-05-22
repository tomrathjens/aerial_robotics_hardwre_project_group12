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
from scipy.interpolate import splprep, splev  # Import B-spline functions
import visualization_utils as visu

import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import csv
import os
import tempfile


from pynput import keyboard # Import the keyboard module for key press detection

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper



def correct_gate_format(gates):
    return [[i+1, *gate, 0.4] for i, gate in enumerate(gates)]


def compute_bspline_trajectory(waypoints, nb_points=500, degree=3):
    pts = np.array([[w[0], w[1], w[2]] for w in waypoints]).T
    N_wp = pts.shape[1]
    k = min(degree, N_wp - 1)
    tck, u = splprep(pts, k=k, s=0)
    u_fine = np.linspace(0, 1, nb_points)
    x_f, y_f, z_f = splev(u_fine, tck)
    traj = [[x_f[i], y_f[i], z_f[i], 0] for i in range(nb_points)]
    return traj


def find_cone_base_intersection(traj, plane_point, normal, radius):
    pts = np.array([[p[0], p[1], p[2]] for p in traj])
    d = np.dot(pts - plane_point, normal)
    eps = 1e-2
    idxs = np.where(np.abs(d) < eps)[0]
    for idx in idxs:
        proj = pts[idx] - d[idx] * normal
        if np.linalg.norm(proj - plane_point) <= radius:
            return proj

    # Discrétisation du cercle pour trouver le point le plus proche du waypoint précédent
    theta = np.linspace(0, 2*np.pi, 100)
    tmp = np.array([0, 0, 1]) if abs(normal[2]) < 0.9 else np.array([1, 0, 0])
    u = np.cross(normal, tmp); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    circle_pts = np.array([plane_point + radius * (np.cos(t) * u + np.sin(t) * v) for t in theta])

    # Point de référence = premier point de la trajectoire (le dernier waypoint précédent en pratique)
    ref = pts[0] if len(pts) == 0 else pts[np.argmin(np.linalg.norm(pts - plane_point, axis=1))]
    dists = np.linalg.norm(circle_pts - ref, axis=1)
    best_pt = circle_pts[np.argmin(dists)]
    return best_pt


def ensure_cone_passages(start, gates, end, csv_file=None, nb_points=500):
    waypoints = [[*start, 0]] + [[x, y, z, 0] for _, x, y, z, _, _ in gates] + [[*end, 0]]

    for i, (_, x, y, z, theta, size, size_cone) in enumerate(gates):
        size_cone = 0.25
        axis = np.array([-np.sin(theta), np.cos(theta), 0.0])
        axis /= np.linalg.norm(axis)
        h, r = size_cone, size_cone / 1.0
        center = [x, y, z, 0]

        # Définir les indices pour la sous-trajectoire
        idx_center = next(i for i, w in enumerate(waypoints) if np.allclose(w, center))

        # Partie amont
        wp_amont = waypoints[:idx_center+1]
        traj_amont = compute_bspline_trajectory(wp_amont, nb_points)
        plane_in = np.array([x, y, z]) - axis * h
        pin = find_cone_base_intersection(traj_amont, plane_in, -axis, r)
        waypoints.insert(idx_center, [*pin, 0])

        # Partie aval
        idx_center = next(i for i, w in enumerate(waypoints) if np.allclose(w, center))
        wp_aval = waypoints[idx_center:idx_center+2] if idx_center+2 <= len(waypoints) else waypoints[idx_center:]
        wp_aval = [waypoints[idx_center-1]] + wp_aval if idx_center > 0 else wp_aval
        traj_aval = compute_bspline_trajectory(wp_aval, nb_points)
        plane_out = np.array([x, y, z]) + axis * h
        pout = find_cone_base_intersection(traj_aval, plane_out, axis, r)
        waypoints.insert(idx_center + 1, [*pout, 0])

    traj = compute_bspline_trajectory(waypoints, nb_points)
    return waypoints, traj

# TODO: CHANGE THIS URI TO YOUR CRAZYFLIE & YOUR RADIO CHANNEL
uri = uri_helper.uri_from_env(default='radio://0/20/2M/E7E7E7E712') #Example for group 17

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

####################################

################parameters to fill in ############
nb_laps = 2 #number of laps to do
nb_points = 100 #to adjust so that the space between points is around 0.1 m
time_bwn_points = 0.1 #time to wait between points of the path in seconds
# gate1 = [0.6, -0.32, 0.75, 0]  # x, y, z, yaw coordinates of the first gate relative to the starting point of the drone
# gate2 = [2.09, 0.25, 1.29, 0]
# gate3 = [0.11, 0.93, 1.16, 0]
# gate4 = [-0.79, 0.4, 1.27, 0]
gates_csv_file = r".\gates.csv"

time_takeoff = 5 #time to take off in seconds
height_takeoff = 0.6 #height of the drone

pose_reached = 0.4 #distance to consider a waypoint as reached

estimate_traj = []
#################################################

gates_in_order = visu.extract_gates_from_csv(gates_csv_file, format='waypoints')
after_take_off = [0,0,height_takeoff,0]
last_point = [0,0,height_takeoff,0]
while nb_laps > 1:
    gates_in_order = gates_in_order + gates_in_order #repeat the gates in order nb_laps times
    nb_laps -= 1


# Create multiple points between the gates to make the path smoother and equidistant
gates = np.array(gates_in_order)


waypoints, traj = ensure_cone_passages(after_take_off, gates, last_point, csv_file=fp, nb_points=1000)
final_path = compute_bspline_trajectory(waypoints, 100)
final_path_points = [[p[0], p[1], p[2], 0] for p in final_path]

# Check distacne between waypoints
if len(final_path_points)) > 1:
                total_distance = 0
                for i in range(len(final_path_points) - 1):
                    point_a = np.array(final_path_points[i][:3])
                    point_b = np.array(final_path_points[i + 1][:3])
                    total_distance += np.linalg.norm(point_b - point_a)
                average_distance = total_distance / (len(final_path_points) - 1)
                print(f"Average distance between points: {average_distance:.3f} m")



#######################################
# Fuctions 
#######################################
def get_next_waypoint(final_path_points, pose_reached, current_position):
    """
    Dynamically updates the waypoints list by removing reached points and returning the next waypoint.
    
    Args:
        waypoints: List of waypoints [x, y, z, yaw].
        pose_reached: Distance threshold to consider a waypoint as reached.
        current_position: Current position of the drone [x, y, z].
    
    Returns:
        next_waypoint: The next waypoint to move to [x, y, z, yaw].
    """
    while final_path_points:
        distance_to_waypoint = np.linalg.norm(np.array(current_position[:3]) - np.array(final_path_points)[0][:3]))
        if distance_to_waypoint < pose_reached:
            final_path_points.pop(0)
        else:
            break

    # Return the next waypoint if available
    return final_path_points[0] if final_path_points else last_point


#################################

class LoggingExample:
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 10s.
    """

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """

        self._cf = Crazyflie(rw_cache='./cache')

        self.prev_z_ = [None, None, None, None, None]

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

        # Init states variables
        self.sensor_data = {}

        self.sensor_data['t'] = 0
        self.sensor_data["x"] = 0
        self.sensor_data["y"] = 0
        self.sensor_data["z"] = 0
        self.sensor_data["yaw"] = 0

        # Accumulators for smoothing / hand detection 
        self.accumulator_z = [0]*10

        # Boolean states
        self.emergency_stop = False
        self.block_callback = False 

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

        # Print the data to the console to see what the Logger is getting !
        #print(f'[{timestamp}][{logconf.name}]: ', end='')
        # for name, value in data.items():
        #     print(f'{name}: {value:3.3f} ', end='')
        # print()

        # Store the state estimate data you need in a dictionary
        if not(self.block_callback):
            self.block_callback = True  # Prevents the callback from being called again while processing

            # Update the other states
            for name, value in data.items():
                if name == 'stateEstimate.x':
                    self.sensor_data['x'] = value
                if name == 'stateEstimate.y':
                    self.sensor_data['y'] = value
                if name == 'stateEstimate.z':
                    self.sensor_data['z'] = value
                    # Update the accumulator
                    self.accumulator_z.append(value)
                    self.accumulator_z.pop(0)
                if name == 'stabilizer.yaw':
                    self.sensor_data['yaw'] = value

            self.block_callback = False

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
def emergency_stop_callback(le):
    cf = le._cf  # Access the Crazyflie instance from the LoggingExample
    
    def on_press(key):
        try:
            if key.char == 'q':  # Check if the "space" key is pressed
                print("Emergency stop triggered!")
                le.emergency_stop = True
                return False     # Stop the listener
        except AttributeError:
            pass

    # Start listening for key presses
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    # Send the stop setpoint to the Crazyflie if the emergency stop is triggered
    if le.emergency_stop:
        cf.commander.send_stop_setpoint()
        cf.close_link()

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
    emergency_stop_thread = threading.Thread(target=emergency_stop_callback, args=(le,))
    emergency_stop_thread.start()

    # TODO : CHANGE THIS TO YOUR NEEDS
    # print("plotting trajectory")
    # visu.visualize_gates(csv_file=gates_csv_file, target_traj=waypoints, close=False)


    print("Starting control")
    while le.is_connected:
        time.sleep(0.01)
        print("Sending commands")
        # Take-off
        ticks = int(time_takeoff / 0.1)  # Number of ticks to wait between points
        magic_number = ticks / height_takeoff
        print("magic calculated: ", magic_number)
        for y in range(ticks):
            cf.commander.send_hover_setpoint(0, 0, 0, y / magic_number)
            time.sleep(0.1)
        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, height_takeoff)
            time.sleep(0.1)

        # Move 
        while waypoints:
            # Get the current position of the drone from the log data
            #log_data = le._lg_stab.data_received_cb  # Example of accessing log data
            current_position = [le.sensor_data['x'], le.sensor_data['y'], le.sensor_data['z'], le.sensor_data['yaw']]
            estimate_traj.append(current_position)

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

        # Land
        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, height_takeoff)
            time.sleep(0.1)

        ticks = int(time_takeoff / 0.1)  # Number of ticks to wait between points
        magic_number = ticks / height_takeoff

        

        for y in range(ticks):
            cf.commander.send_hover_setpoint(0, 0, 0, (ticks - y) / magic_number)
            time.sleep(0.1)

        visu.visualize_gates(csv_file=gates_csv_file, close=False, estimate_traj=estimate_traj)

        cf.commander.send_stop_setpoint()
        break