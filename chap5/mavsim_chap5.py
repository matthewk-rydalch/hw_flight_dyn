"""
mavSimPy 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        1/1/2019 - RWB
        1/29/2019 - RWB
        2/2/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap2.spacecraft_viewer import spacecraft_viewer
from chap2.video_writer import video_writer
from chap3.data_viewer import data_viewer
from chap4.mav_dynamics import mav_dynamics
from chap4.wind_simulation import wind_simulation
from chap5.trim import compute_trim
from chap5.compute_models import Compute_Models

# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = spacecraft_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots
models = Compute_Models()
if VIDEO == True:
    video = video_writer(video_name="chap5_video.avi",
                         bounding_box=(0, 0, 1000, 1000),
                         output_rate=SIM.ts_video)

# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)

# use compute_trim function to compute trim state and trim input
Va = 25.
gamma = 0.0*np.pi/180.
trim_state, trim_input = compute_trim(mav, Va, gamma)
mav._state = trim_state  # set the initial state of the mav to the trim state
delta = trim_input  # set input to constant constant trim input
long_impulse = trim_input + np.array([[0.0, 5.0, 0.0, 0.0]]).T
aleron_impulse = np.array([[15.0, 0.0, 0.0, 0.0]]).T
lat_impulse1 = trim_input + aleron_impulse
lat_impulse2 = trim_input - aleron_impulse
long_mode = False
lat_mode = False
if lat_mode:
    doublet = True
else:
    doublet = False

# # compute the state space model linearized about trim
# A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r \
    = models.compute_tf_model(mav, trim_state, trim_input)


# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    #-------physical system-------------
    #current_wind = wind.update()  # get the new wind vector
    current_wind = np.zeros((6,1))
    if long_mode:
        mav.update_state(long_impulse, current_wind)  # propagate the MAV dynamics
    elif lat_mode:
        if doublet:
            mav.update_state(lat_impulse1, current_wind)  # propagate the MAV dynamics
        else:
            mav.update_state(lat_impulse2, current_wind)  # propagate the MAV dynamics
            lat_mode = False
    else:
        mav.update_state(delta, current_wind)  # propagate the MAV dynamics
    long_mode = False
    doublet = False
    #-------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV
    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     mav.msg_true_state, # commanded states
                     SIM.ts_simulation)
    if VIDEO == True:
        video.update(sim_time)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO == True:
    video.close()




