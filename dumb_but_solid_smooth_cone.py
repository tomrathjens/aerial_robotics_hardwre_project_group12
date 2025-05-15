# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Crazyflie Nano Quadcopter Client
#
#  Adapté pour génération de trajectoires non-jerk avec contrainte de cône d'approche

import logging
import time
import threading
import numpy as np
from pynput import keyboard
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

# CHANGE THIS URI TO YOUR CRAZYFLIE & RADIO CHANNEL
uri = uri_helper.uri_from_env(default='radio://0/20/2M/E7E7E7E712')
logging.basicConfig(level=logging.ERROR)

# ===== Paramètres utilisateur =====
nb_laps = 1                # Nombre de tours
nb_points = 100            # Nombre de points par segment minimal-jerk
time_step = 0.08           # Temps entre envois de consigne (s)
cone_angle_deg = 45.0      # Ouverture du cône à l'apex (°)
cone_length = 1.0          # Hauteur du cône (distance centre->base, m)

# Définition des portes [x, y, z, yaw(rad)]
gate1 = [1.15, -0.54, 0.79, np.deg2rad(30)]
gate2 = [2.16,  0.34, 1.15, np.deg2rad(-45)]
gate3 = [0.69,  1.14, 1.55, np.deg2rad(90)]
gate4 = [-0.70, 0.61, 1.65, np.deg2rad(15)]
gates = [gate1, gate2, gate3, gate4]
after_takeoff = [0.0, 0.0, 0.4, 0.0]

# Répéter laps
gates_in_order = gates * nb_laps
# Ajouter décollage et retour
gates_sequence = [after_takeoff] + gates_in_order + [after_takeoff]

# Paramètres cône
cone_angle = np.deg2rad(cone_angle_deg)
cone_radius = cone_length * np.tan(cone_angle)

# Fonction minimal-jerk entre p0 et p1
def minimal_jerk_segment(p0, p1, yaw0, yaw1, n_pts):
    u = np.linspace(0, 1, n_pts)
    poly = 3*u**2 - 2*u**3
    pos = p0 + (p1 - p0) * poly[:, None]
    yaw = yaw0 + (yaw1 - yaw0) * u
    return [[*pos[i], yaw[i]] for i in range(n_pts)]

# Vérifie si une trajectoire passe par le disque/base du cône
def passes_through_base(traj, center, normal):
    # Distance plan: (p-center)·normal + cone_length
    d = np.dot(traj - center, normal) + cone_length
    # Indices proches de zéro
    idx = np.where(np.isclose(d, 0, atol=1e-2))[0]
    for i in idx:
        # projection dans plan
        v = traj[i] - center - np.dot(traj[i]-center, normal)*normal
        if np.linalg.norm(v) <= cone_radius:
            return True
    return False

# Calcul du point via sur le cercle de base
def compute_via_point(traj, center, normal):
    d = np.dot(traj - center, normal) + cone_length
    i_min = np.argmin(np.abs(d))
    p = traj[i_min]
    # proj sur plan
    p_proj = p - (np.dot(p-center, normal) + cone_length)*normal
    # vecteur radial
    v = p_proj - center
    v_rad = v - np.dot(v, normal)*normal
    dir_rad = v_rad / np.linalg.norm(v_rad)
    via = center - normal*cone_length + dir_rad*cone_radius
    return via

# Construction de la liste de waypoints respectant chaque cône
def build_waypoints(seq):
    waypts = []
    prev_pt = np.array(seq[0][:3])
    prev_yaw = seq[0][3]
    for nxt in seq[1:]:
        center = np.array(nxt[:3])
        yaw = nxt[3]
        # Traj basique
        base_traj = np.array([p[:3] for p in minimal_jerk_segment(prev_pt, center, prev_yaw, yaw, nb_points)])
        # Si segment entre deux gates (pas décollage/atterrissage)
        if not np.all(prev_pt == seq[0]) and not np.all(center == seq[-1][:3]):
            normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            if passes_through_base(base_traj, center, normal):
                seg = minimal_jerk_segment(prev_pt, center, prev_yaw, yaw, nb_points)
            else:
                via = compute_via_point(base_traj, center, normal)
                seg1 = minimal_jerk_segment(prev_pt, via, prev_yaw, yaw, nb_points//2)
                seg2 = minimal_jerk_segment(via, center, yaw, yaw, nb_points//2)
                seg = seg1 + seg2
        else:
            seg = minimal_jerk_segment(prev_pt, center, prev_yaw, yaw, nb_points)
        waypts.extend(seg)
        prev_pt = center
        prev_yaw = yaw
    return waypts

# Génération des waypoints
waypoints = build_waypoints(gates_sequence)
# Debug: moyenne distance
if len(waypoints) > 1:
    ds = [np.linalg.norm(np.array(waypoints[i+1][:3]) - np.array(waypoints[i][:3]))
          for i in range(len(waypoints)-1)]
    print(f"Distance moyenne entre waypoints: {np.mean(ds):.3f} m")

# Classe de logging et gestion vol
class LoggingExample:
    def __init__(self, link_uri):
        self._cf = Crazyflie(rw_cache='./cache')
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)
        print(f"Connecting to {link_uri}")
        self._cf.open_link(link_uri)
        self.is_connected = True
    def _connected(self, link_uri):
        print(f"Connected to {link_uri}")
        lg = LogConfig(name='Stabilizer', period_in_ms=50)
        for var in ['stateEstimate.x','stateEstimate.y','stateEstimate.z','stabilizer.yaw']:
            lg.add_variable(var, 'float')
        try:
            self._cf.log.add_config(lg)
            lg.data_received_cb.add_callback(lambda *a: None)
            lg.error_cb.add_callback(lambda *a: print("Log error"))
            lg.start()
        except Exception as e:
            print("Log config error", e)
        Timer(50, self._cf.close_link).start()
    def _stab_log_error(self, *args): pass
    def _stab_log_data(self, *args): pass
    def _connection_failed(self, uri, msg): print(msg); self.is_connected=False
    def _connection_lost(self, uri, msg): print(msg); self.is_connected=False
    def _disconnected(self, uri): print("Disconnected"); self.is_connected=False

def emergency_stop_callback(cf):
    def on_press(key):
        try:
            if key.char == 'q':
                print("Emergency stop triggered!")
                cf.commander.send_stop_setpoint()
                cf.close_link()
                return False
        except AttributeError:
            pass
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == '__main__':
    cflib.crtp.init_drivers()
    le = LoggingExample(uri)
    cf = le._cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)
    threading.Thread(target=emergency_stop_callback, args=(cf,)).start()
    print("Starting control")
    while le.is_connected:
        time.sleep(0.01)
        # Takeoff
        for z in np.linspace(0, 0.4, 30):
            cf.commander.send_hover_setpoint(0, 0, 0, z)
            time.sleep(0.1)
        # Envoi des waypoints
        for pt in waypoints:
            cf.commander.send_position_setpoint(*pt)
            time.sleep(time_step)
        # Landing
        for z in np.linspace(0.4, 0, 30):
            cf.commander.send_hover_setpoint(0, 0, 0, z)
            time.sleep(0.1)
        cf.commander.send_stop_setpoint()
        break
