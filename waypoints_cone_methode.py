import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import numpy as np
import csv
import os
import tempfile
from scipy.interpolate import splprep, splev


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

    for i, (_, x, y, z, theta, size) in enumerate(gates):
        axis = np.array([-np.sin(theta), np.cos(theta), 0.0])
        axis /= np.linalg.norm(axis)
        h, r = size, size / 1.0
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

def visualize_gates(csv_file=None, target_traj=None, waypoints=None, close=True, save=False, savepath=None):
    if not hasattr(visualize_gates, "initialized"):
        visualize_gates.initialized = True
        visualize_gates.drone_traj = []
    fig = plt.figure('visualization')
    ax = fig.add_subplot(111, projection='3d')
    if csv_file:
        data = pd.read_csv(csv_file)
        xs, ys, zs = data['x'], data['y'], data['z']
        thetas, sizes = data['theta'], data['size']
        gate_ids = data['Gate']
        ax.scatter(xs, ys, zs, c='blue', label='Portes (centre)')
        for x, y, z, theta, size, gid in zip(xs, ys, zs, thetas, sizes, gate_ids):
            half = size/1.0
            corners = np.array([[-half,0,-half],[half,0,-half],[half,0,half],[-half,0,half]])
            R = np.array([[-np.cos(theta), np.sin(theta),0],[-np.sin(theta),-np.cos(theta),0],[0,0,1]])
            pts = corners.dot(R.T) + np.array([x, y, z])
            ax.add_collection3d(art3d.Poly3DCollection([pts], alpha=0.3, color='cyan'))
            ax.text(x, y, z+0.05, f'Porte {gid}')
            axis = np.array([-np.sin(theta), np.cos(theta), 0.0])
            axis /= np.linalg.norm(axis)
            tmp = np.array([0,0,1]) if abs(axis[2])<0.9 else np.array([1,0,0])
            u1 = np.cross(axis, tmp); u1/=np.linalg.norm(u1)
            u2 = np.cross(axis, u1)
            h, r = size, half
            phis = np.linspace(0,2*np.pi,30); us = np.linspace(0,h,20)
            Phi, U = np.meshgrid(phis,us); Rm = (U/h)*r
            for sign, col in [(-1,'blue'),(1,'red')]:
                X = x + sign*axis[0]*U + (u1[0]*np.cos(Phi)+u2[0]*np.sin(Phi))*Rm
                Y = y + sign*axis[1]*U + (u1[1]*np.cos(Phi)+u2[1]*np.sin(Phi))*Rm
                Z = z + sign*axis[2]*U + (u1[2]*np.cos(Phi)+u2[2]*np.sin(Phi))*Rm
                ax.plot_surface(X, Y, Z, color=col, alpha=0.2, linewidth=0)
                center = np.array([x, y, z]) + sign*axis*h
                s = 0.5
                uu = np.linspace(-s,s,2); vv = np.linspace(-s,s,2)
                UU, VV = np.meshgrid(uu, vv)
                Xp = center[0] + UU*u1[0] + VV*u2[0]
                Yp = center[1] + UU*u1[1] + VV*u2[1]
                Zp = center[2] + UU*u1[2] + VV*u2[2]
                ax.plot_surface(Xp, Yp, Zp, color=col, alpha=0.1, linewidth=0)
    ax.scatter(0,0,0,c='green',s=30,label='Décollage')
    Xg, Yg = np.meshgrid(np.linspace(-2,2,10), np.linspace(-2,2,10)); Zg=np.zeros_like(Xg)
    ax.plot_wireframe(Xg, Yg, Zg, color='gray', alpha=0.25)
    if target_traj is not None:
        T = np.array(target_traj)
        ax.plot(T[:,0], T[:,1], T[:,2], c='orange', ls='--', label='Traj B-spline')
    if waypoints:
        N = (len(waypoints)-2)//3
        ent, ext = [], []
        for i in range(N):
            idx = 1+3*i; ent.append(waypoints[idx][:3]); ext.append(waypoints[idx+2][:3])
        ent, ext = np.array(ent), np.array(ext)
        ax.scatter(ent[:,0],ent[:,1],ent[:,2],c='magenta',s=50,marker='o',label='Entrées')
        ax.scatter(ext[:,0],ext[:,1],ext[:,2],c='black',s=50,marker='x',label='Sorties')
    ax.set_xlim(-1,3); ax.set_ylim(-2,2)
    if csv_file: ax.set_zlim(0, max(zs)+0.5)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.legend(); ax.view_init(elev=20, azim=120)
    if close:
        plt.show(block=False)
    else:
        plt.show()
    if save:
        os.makedirs(os.path.dirname(savepath) or '.', exist_ok=True); plt.savefig(savepath)

if __name__ == '__main__':
    start = [0,0,0.4]; end = [0,0,0.4]
    cone_size = 0.25
    gates = [
        [1, 1.15, -0.54, 0.79, np.deg2rad(-170), cone_size],
        [2, 2.16,  0.34, 1.20, np.deg2rad(116),   cone_size],
        [3, 0.69,  1.14, 1.0, np.deg2rad(85),   cone_size],
        [4,-0.7,   0.61, 1.65, np.deg2rad(210),  cone_size]
    ]

    formatted = correct_gate_format([[g[1],g[2],g[3],g[4]] for g in gates])
    fp = os.path.join(tempfile.gettempdir(),'gates.csv')
    with open(fp,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['Gate','x','y','z','theta','size']); w.writerows(gates)
    waypoints, traj = ensure_cone_passages(start, gates, end, csv_file=fp, nb_points=1000)
    final_path = compute_bspline_trajectory(waypoints, 100)
    final_path_points = [[p[0], p[1], p[2], 0] for p in final_path]
    visualize_gates(csv_file=fp, target_traj=traj, waypoints=waypoints, close=False)
    output_path = os.path.join(os.path.dirname(__file__), 'final_path_points.txt')
    with open(output_path, 'w') as f:
        f.write("waypoints = [\n")
        for p in final_path_points:
            f.write(f"    [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}, {p[3]}],\n")
        f.write("]\n")
