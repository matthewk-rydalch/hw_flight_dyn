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
from chap10.path_viewer import path_viewer

# initialize the visualization
path_view = path_viewer()  # initialize the viewer
data_view = data_viewer()  # initialize view of data plots

# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)
ctrl = autopilot(SIM.ts_simulation)
obsv = observer(SIM.ts_simulation)
path_follow = path_follower()

# path definition
from message_types.msg_path import msg_path
path = msg_path()
path.flag = 'line'
path.line_origin = np.array([[0.0, 0.0, -100.0]]).T
path.line_direction = np.array([[0.5, 1.0, 0.0]]).T
path.line_direction = path.line_direction / np.linalg.norm(path.line_direction)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    #-------observer-------------
    mav.update_sensors() #I added, seems to be necessary
    measurements = mav._sensors  # get sensor measurements
    estimated_state = obsv.update(measurements)  # estimate states from measurements

    #-------path follower-------------
    autopilot_commands = path_follow.update(path, estimated_state)
    # autopilot_commands = path_follow.update(path, mav.msg_true_state) #for debugging

    #-------controller-------------
    delta, commanded_state = ctrl.update(autopilot_commands, estimated_state)

    #-------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    # current_wind = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    mav.update_state(delta, current_wind)  # propagate the MAV dynamics

    #-------update viewer-------------
    path_view.update(path, mav.msg_true_state)  # plot path and MAV
    data_view.update(mav.msg_true_state, # true states
                     estimated_state, # estimated states
                     commanded_state, # commanded states
                     SIM.ts_simulation)

    #-------increment time-------------
    sim_time += SIM.ts_simulation
