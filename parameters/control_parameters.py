import sys
sys.path.append('..')
import numpy as np
from chap5.trim import compute_trim
from chap5.compute_models import Compute_Models
import parameters.aerosonde_parameters as MAV
from chap4.mav_dynamics import mav_dynamics
import parameters.simulation_parameters as SIM

#instantiate classes
TF = Compute_Models()
mav = mav_dynamics(SIM.ts_simulation)

#need a_th, a_phi and such values from TF
Va = 25.
gamma = 0.0*np.pi/180.
trim_state, trim_input = compute_trim(mav, Va, gamma)
T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r \
    = TF.compute_tf_model(mav, trim_state, trim_input)

#Tuning parameters
e_phi_max = 15 #degrees
xsi_phi = 0.707
tr_chi = 1
xsi_chi = 0.707
e_beta_max = 15 #degrees
xsi_B = 0.707
tr_th = 0.102
xsi_th = 0.3
Wh = 20
xsi_h = 0.707
Wv2 = 10
xsi_v = 0.707
tr_v = 1
tau_r_yaw_damper = 0.5 #0.5 is from the root locus in the supplimental material
kr_yaw_damper = 0.3 #I completely guessed on this

#parameters for solving kp, kd, ki
g = MAV.gravity
# sigma =
Va0 = 10 #m/s
delta_a_max = 45 #degrees
delta_r_max = 45 #degrees, I made this up
wn_phi = np.sqrt(abs(TF.a_phi_2)*delta_a_max/e_phi_max)
wn_chi = 2.2/tr_chi
Vg = mav._Vg
delta_e_max = 45 #degrees
wn_th = 2.2/tr_th
wn_h = 1/Wh*wn_th
wn_v = 2.2/tr_v


#----------roll loop-------------
roll_kp = delta_a_max/e_phi_max*np.sign(TF.a_phi_2)
roll_kd = (2.0*xsi_phi*wn_phi-TF.a_phi_1)/TF.a_phi_2

#----------course loop-------------
course_kp = 2*xsi_chi*wn_chi*Vg/g
course_ki = wn_chi**2*Vg/g

#We are doing yaw damper instead of sideslip loop
# #----------sideslip loop-------------
# sideslip_kp = delta_r_max/e_beta_max*np.sign(TF.a_B_2)
# sideslip_ki = 1/TF.a_B_2*((TF.a_B_1+TF.a_B_2*sideslip_kp)/2*xsi_B)**2


#----------yaw damper-------------
yaw_damper_tau_r = tau_r_yaw_damper
yaw_damper_kp = kr_yaw_damper #This is supposed to be kr?

#----------pitch loop-------------
pitch_kp = (wn_th**2-TF.a_th_2)/TF.a_th_3
pitch_kd = (2*xsi_th*wn_th-TF.a_th_1)/TF.a_th_3
K_theta_DC = pitch_kp*TF.a_th_3/(TF.a_th_2+pitch_kp*TF.a_th_3)

#----------altitude loop-------------
altitude_kp = 2*xsi_h*wn_h/(K_theta_DC*Va0)
altitude_ki = wn_h**2/(K_theta_DC*Va0)
altitude_zone = 10.0

#---------airspeed hold using throttle---------------
airspeed_throttle_kp = (2.0*xsi_v*wn_v*TF.a_v_1)/TF.a_v_2
# airspeed_throttle_ki = wn_v**2/TF.a_v_2
airspeed_throttle_ki = 5.0