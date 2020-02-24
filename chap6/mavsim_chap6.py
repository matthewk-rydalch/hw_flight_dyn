"""
mavsim_python
    - Chapter 6 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/5/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap2.spacecraft_viewer import spacecraft_viewer
from chap3.data_viewer import data_viewer
from chap4.mav_dynamics import mav_dynamics
from chap4.wind_simulation import wind_simulation
from chap6.autopilot import autopilot
from tools.signals import signals
from chap5.trim import compute_trim


#TODO: To pick up from Tuesday, figure out why chi is behaving so strangly.  I have made some adjustments to fix it in mav_dynamics, but that didn't work.

# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = spacecraft_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots
if VIDEO == True:
    from chap2.video_writer import video_writer
    video = video_writer(video_name="chap6_video.avi",
                         bounding_box=(0, 0, 1000, 1000),
                         output_rate=SIM.ts_video)

# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)
ctrl = autopilot(SIM.ts_simulation)

# autopilot commands
from message_types.msg_autopilot import msg_autopilot
commands = msg_autopilot()
Va_command = signals(dc_offset=25.0, amplitude=3.0, start_time=2.0, frequency = 0.01)
h_command = signals(dc_offset=100.0, amplitude=10.0, start_time=0.0, frequency = 0.02)
chi_command = signals(dc_offset=np.radians(0), amplitude=np.radians(45), start_time=5.0, frequency = 0.015)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
trim_state, trim_input = compute_trim(mav, 25.0, 0.0) #git rid of this
while sim_time < SIM.end_time:

    #-------controller-------------
    estimated_state = mav.msg_true_state  # uses true states in the control
    commands.airspeed_command = Va_command.square(sim_time)
    commands.course_command = chi_command.square(sim_time)
    commands.altitude_command = h_command.square(sim_time)
    delta, commanded_state = ctrl.update(commands, estimated_state)
    #do I need to include a command state for roll feed forward?

    #-------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    current_wind = np.zeros((6, 1)) #get rid of this
    input = np.array([[delta.item(0), delta.item(1), trim_input.item(2), delta.item(3)]]).T #chi is off for anything but trim
    mav.update_state(input, current_wind)  # propagate the MAV dynamics

    #-------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV
    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     commanded_state, # commanded states
                     SIM.ts_simulation)
    if VIDEO == True: video.update(sim_time)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO == True: video.close()




