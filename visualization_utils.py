import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import numpy as np
import csv
import os
from scipy.interpolate import splprep, splev


def visualize_gates(csv_file=None, target_traj=None, estimate_pose=None, estimate_traj=None, show_estimate_traj=False, close=True, save=False, savepath=None):
    """
    -------
    @ pars
    - csv_file : csv file (path) containing the gates infos in this fashion : Gate,x,y,z,theta,size (default: None)
    - target_traj : list of points (x, y, z) representing the target trajectory (default: None)
    - estimate_pose : tuple (x,y,z,theta) representing the current estimate pose of the drone (default: None)
    - estimate_traj : list of lists [[x,y,z], ...] representing the trajectory from estimate poses (default: None)
    - show_estimate_traj : bool. Enable displaying the actual trajectory taken by the drone (default: False)
    - close : bool. Close the figure window to prevent locking the code.
    - save : bool. Enable saving the figure
    - savepath : string. Path to save the figure
    -------
    @ return
    - None
    -------
    @ brief
    - 3D visualization of the gates' poses and optional trajectory, 
    -------
    """
    # ---- FUNCTION ATTRIBUTES ----
    if not hasattr(visualize_gates, "initialized"):
        visualize_gates.initialized = True
        visualize_gates.drone_traj = []     # stored drone's poses


    FIGURE_NAME = "visualization"

    ## load CSV file and extract infos
    if csv_file is not None:
        data = pd.read_csv(csv_file)
        
        x = data['x']
        y = data['y']
        z = data['z']
        theta = data['theta']
        size = data['size']
        gate_numbers = data['Gate']
    

    ## 3D plot
    fig = plt.figure(FIGURE_NAME)
    ax = fig.add_subplot(111, projection='3d')
    
    # gate position
    if csv_file is not None:
        ax.scatter(x, y, z, c='blue', label='Gates')
        
        # orientation and size
        for i in range(len(data)):
            dx = -np.sin(theta[i]) * size[i] / 2
            dy = np.cos(theta[i]) * size[i] / 2
            ax.quiver(x[i], y[i], z[i], dx, dy, 0, color='red')
            
            half_size = size[i] / 2
            corners = np.array([
                [-half_size, 0, -half_size],
                [half_size, 0, -half_size],
                [half_size, 0, half_size],
                [-half_size, 0, half_size]
            ])
            
            rotation_matrix = np.array([
                [-np.cos(theta[i]), np.sin(theta[i]), 0],
                [-np.sin(theta[i]), -np.cos(theta[i]), 0],
                [0, 0, 1]
            ])
            
            rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([x[i], y[i], z[i]])
            
            square = art3d.Poly3DCollection([rotated_corners], alpha=0.3, color='blue')
            ax.add_collection3d(square)
            
            ax.text(x[i], y[i], z[i] + 0.1, f"Gate {gate_numbers[i]}", color='black')
    
    # take-off (origin)
    ax.scatter(0, 0, 0, c='green', s=30, label='Take-off pose')
    
    # fixed XY plane at z=0
    x_grid = np.linspace(-2, 2, 10)
    y_grid = np.linspace(-2, 2, 10)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    z_grid = np.zeros_like(x_grid)
    ax.plot_wireframe(x_grid, y_grid, z_grid, color='gray', alpha=0.25, linewidth=0.5)
    
    # plot target trajectory if provided
    if target_traj is not None:
        target_traj = np.array(target_traj)
        ax.plot(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], c='orange', ls = ':', label='Target trajectory', linewidth=1)
        ax.scatter(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], c='orange', s=10)
    
    # plot drone's current estimated pose if provided
    if estimate_pose is not None:
        # store pose
        visualize_gates.drone_traj.append(estimate_pose)

        # plot current pose with orientation
        x_pose, y_pose, z_pose, theta_pose = estimate_pose
        dx_pose = np.cos(theta_pose) * 0.2
        dy_pose = np.sin(theta_pose) * 0.2
        ax.quiver(x_pose, y_pose, z_pose, dx_pose, dy_pose, 0, color='purple')
        ax.scatter(x_pose, y_pose, z_pose, c='purple', s=50, label='Drone Position')


    # plot drone's estimated trajectory until now
    if show_estimate_traj:
        if estimate_traj is not None:
            traj_array = np.array(estimate_traj)
        else:
            traj_array = np.array(visualize_gates.drone_traj)
        ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], c='cyan', ls = ':', label='Estimated trajectory', linewidth=1)
        ax.scatter(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], c='cyan', s=10)

    
    # plot pars
    ax.set_xlim([-1, 3]) 
    ax.set_ylim([-2, 2])
    if csv_file is not None: ax.set_zlim([0, max(z) + 0.5])

    ax.view_init(elev=20, azim=120, roll=0)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Visualization')
    ax.legend()
    
    if close:
        plt.show(block=False)
        # plt.pause(0.001)
        plt.close(fig=FIGURE_NAME)
    else:
        plt.show()
    

    # save figure
    if save:
        if savepath is None:
            savepath = r".\visualization.svg"

        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        print(f"Figure saved at: {savepath}")

    return



def create_gates_infos_csv(gates, name="current_gates_info.csv", path=None):
    """
    -------
    @ pars
    - gates : gates info in order. list of lists [[Gate,x,y,z,theta,size], ...]
    - name : (optional) name of the file to be created (string)
    - path : (optional) directory where the file should be created (string)
    -------
    @ return
    - file_path : path where the csv file is saved
    -------
    @ brief
    - create a .csv file containing the gates' info
    -------
    """

    NB_GATES = 4
    NB_INFO  = 6    # [Gate,x,y,z,theta,size]

    # check existence and format
    if gates is None:
        print("Error : no gates.")
        return None

    elif len(gates) != NB_GATES:
        print("Error : incorrect gates' number.")
        return None

    elif any(len(gate) != NB_INFO for gate in gates):
        print("Error : incorrect gate format; must be [Gate,x,y,z,theta,size].")
        return None

    # file path
    if path:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, name)
    else:
        file_path = name


    # create csv file
    try:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Gate", "x", "y", "z", "theta", "size"])
            writer.writerows(gates)
        print(f"CSV file '{name}' created successfully.")

    except Exception as e:
        print(f"Error: unable to create CSV file. {e}")

    return file_path




def correct_gate_format(gates):
    """
    -------
    @ pars
    - gates : gates info in order. list of lists [[x,y,z,theta], ...]
    -------
    @ return
    - gates : gates info in order and correct format. list of lists [[Gate,x,y,z,theta,size], ...]
    -------
    @ brief
    - convert simple gates info into correct format suitable to csv file
    -------
    """

    NB_GATES     = 4
    NB_INFO      = 4    # [x,y,z,theta]
    DEFAULT_SIZE = 0.4  # [m]

    # check existence and format
    if gates is None:
        print("Error : no gates.")
        return gates

    elif len(gates) != NB_GATES:
        print("Error : incorrect gates' number.")
        return gates

    elif any(len(gate) != NB_INFO for gate in gates):
        print("Error : incorrect gate format; must be [x,y,z,theta].")
        return gates

    # correct format
    for i, gate in enumerate(gates):
        gate = [i+1] + gate + [DEFAULT_SIZE]
        gates[i] = gate
    
    return gates






##################################
# example with computed trajectory

nb_points = 50 #to adjust so that the space between points is around 0.1 m
time_bwn_points = 1.5 #time to wait between points of the path in seconds
gate1 = [1.15, -0.54, 0.79, np.deg2rad(-180)]  # x, y, z, yaw coordinates of the first gate relative to the starting point of the drone
gate2 = [2.16, 0.34, 1.20, np.deg2rad(15)]
gate3 = [0.69, 1.14, 1.55, np.deg2rad(85)]
gate4 = [-0.7, 0.61, 1.65, np.deg2rad(210)]

gates_in_order = [gate1, gate2, gate3, gate4]
after_take_off = [0,0,0.4,0]


def new_gate_segments(gate):

    th = gate[3] + np.deg2rad(90)
    l = 0.1

    p2 = [gate[0] + l*np.cos(th), gate[1] + l*np.sin(th)]
    p1 = [gate[0] - l*np.cos(th), gate[1] - l*np.sin(th)]

    seg1 = p1 + gate[2:4]
    seg2 = p2 + gate[2:4]

    return [seg1, seg2]

gates_segments = []
for gate in gates_in_order:
    gates_segments.extend(new_gate_segments(gate))

gates_in_order = gates_segments

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


gates = correct_gate_format([gate1, gate2, gate3, gate4])
csv_path = create_gates_infos_csv(gates)
visualize_gates(csv_path, target_traj=waypoints, close=False)
