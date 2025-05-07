import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import numpy as np

def visualize_gates(csv_file, traj=None):
    """
    -------
    @ pars
    - csv_file : csv file (path) containing the gates infos in this fashion : Gate,x,y,z,theta,size
    - traj : list of points (x, y, z) representing the trajectory (default: None)
    -------
    @ return
    - None
    -------
    @ brief
    - 3D visualization of the gates' poses and optional trajectory
    -------
    """

    ## load CSV file and extract infos
    data = pd.read_csv(csv_file)
    
    x = data['x']
    y = data['y']
    z = data['z']
    theta = data['theta']
    size = data['size']
    gate_numbers = data['Gate']
    

    ## 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # gate position
    ax.scatter(x, y, z, c='blue', label='Gates')
    
    # orientation and size
    for i in range(len(data)):
        dx = np.cos(theta[i]) * size[i] / 2
        dy = np.sin(theta[i]) * size[i] / 2
        ax.quiver(x[i], y[i], z[i], dx, dy, 0, color='red', label='Orientation' if i == 0 else "")
        
        half_size = size[i] / 2
        corners = np.array([
            [-half_size, 0, -half_size],
            [half_size, 0, -half_size],
            [half_size, 0, half_size],
            [-half_size, 0, half_size]
        ])
        
        rotation_matrix = np.array([
            [-np.sin(theta[i]), -np.cos(theta[i]), 0],
            [np.cos(theta[i]), -np.sin(theta[i]), 0],
            [0, 0, 1]
        ])
        
        rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([x[i], y[i], z[i]])
        
        square = art3d.Poly3DCollection([rotated_corners], alpha=0.3, color='blue')
        ax.add_collection3d(square)
        
        ax.text(x[i], y[i], z[i] + 0.1, f"Gate {gate_numbers[i]}", color='black')
    
    # take-off (origin)
    ax.scatter(0, 0, 0, c='green', s=100, label='Take-off Pad')
    
    # fixed XY plane at z=0
    x_grid = np.linspace(-2, 2, 10)
    y_grid = np.linspace(-2, 2, 10)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    z_grid = np.zeros_like(x_grid)
    ax.plot_wireframe(x_grid, y_grid, z_grid, color='gray', alpha=0.25, linewidth=0.5)
    
    # plot trajectory if provided
    if traj is not None:
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='orange', ls = ':', label='Trajectory', linewidth=2)
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c='orange', s=10)
    
    # plot pars
    ax.set_xlim([-2, 2]) 
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, max(z) + 1])

    ax.view_init(elev=90, azim=90, roll=-90)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Gate poses visualization')
    ax.legend()
    
    plt.show()

# # example usage
# csv_path = r".\gates doc\gates_info.csv"
# trajectory = [(0, 0, 0), (0.2, -0.1, 0.5), (0.5, -0.5, 1), (1, 0, 1.5)]
# visualize_gates(csv_path, traj=trajectory)