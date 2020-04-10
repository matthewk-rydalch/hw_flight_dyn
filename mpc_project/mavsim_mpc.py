"""
mavsim_python
    - MPC for final project
    - Last Update:
        April 2020
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap3.data_viewer import data_viewer
from chap4.wind_simulation import wind_simulation
from chap6.autopilot import autopilot
from chap7.mav_dynamics import mav_dynamics
from chap8.observer import observer
from chap10.path_follower import path_follower
from mpc_project.waypoint_viewer import waypoint_viewer
from mpc_project.mpc_manager import mpc_manager
from mpc_project.target_manager import target_manager

# initialize the visualization
waypoint_view = waypoint_viewer()  # initialize the viewer
data_view = data_viewer()  # initialize view of data plots

# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)
ctrl = autopilot(SIM.ts_simulation)
obsv = observer(SIM.ts_simulation)
path_follow = path_follower()

#_____mpc parameters______
Ts_mpc = 1.0
time_horizon = 25 #in units of Ts_mpc
mpc = mpc_manager(Ts_mpc, time_horizon)

#_____target intial pose and velocity________#
target_x = 1000.0
target_y = 1000.0
target_pd = 0.0
target_chi = -np.pi/2.0
target_Vg = 25.0
target_posVel = np.array([[target_x, target_y, target_pd, target_chi, target_Vg]]).T
target = target_manager(SIM.ts_simulation, target_posVel)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    #-------observer-------------
    mav.update_sensors() #I added, seems to be necessary
    measurements = mav._sensors  # get sensor measurements
    estimated_state = obsv.update(measurements)  # estimate states from measurements

    #-------MPC-------------------
    target.update()
    if round(sim_time, 3) % Ts_mpc == 0: #this is to reduce the frequency of running mpc.  It works better.
        waypoints, path = mpc.update(estimated_state, target)

    #-------path follower-------------
    autopilot_commands = path_follow.update(path, estimated_state)

    #-------controller-------------
    delta, commanded_state = ctrl.update(autopilot_commands, estimated_state)

    #-------physical system-------------
    # current_wind = wind.update()  # get the new wind vector
    current_wind = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T #wind messes up the velocity matching, so leave it out.
    mav.update_state(delta, current_wind)  # propagate the MAV dynamics

    #-------update viewer-------------
    waypoint_view.update(waypoints, path, mav.msg_true_state, target.state)  # plot path and MAV
    data_view.update(mav.msg_true_state, # true states
                     estimated_state, # estimated states
                     commanded_state, # commanded states
                     SIM.ts_simulation)

    #-------increment time-------------
    sim_time += SIM.ts_simulation
