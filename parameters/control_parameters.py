import sys
sys.path.append('..')
import numpy as np
import chap5.compute_models as TF
import parameters.aerosonde_parameters as MAV
import chap4.mav_dynamics as mav

#Tuning parameters
e_phi_max = 15 #degrees
xsi_phi = 0.707
tr_chi = 1
xsi_chi = 0.707
e_beta_max = 15 #degrees
xsi_B = 0.707
e_th_max = 10 #degrees
xsi_th = 0.707
Wh = 10
xsi_h = 0.707
Wv2 = 10
xsi_v = 0.707
tr_v = 1

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
wn_th = np.sqrt(TF.a_th_2+delta_e_max/e_th_max*abs(TF.a_th_3))
wn_h = 1/Wh*wn_th
wn_v = 2.2/tr_v


#----------roll loop-------------
roll_kp = delta_a_max/e_phi_max*np.sign(TF.a_phi_2)
roll_kd = (2.0*xsi_phi*wn_phi-TF.a_phi_1)/TF.a_phi_2

#----------course loop-------------
course_kp = 2*xsi_chi*wn_chi*Vg/g
course_ki = wn_chi**2*Vg/g

#----------sideslip loop-------------
sideslip_kp = delta_r_max/e_beta_max*np.sign(TF.a_B_2)
sideslip_ki = 1/TF.a_B_2*((TF.a_B_1+TF.a_B_2*sideslip_kp)/2*xsi_B)**2


# #----------yaw damper-------------
# yaw_damper_tau_r =
# yaw_damper_kp =

#----------pitch loop-------------
pitch_kp = delta_e_max/e_th_max*np.sign(TF.a_th_3)
pitch_kd = (2*xsi_th*wn_th-TF.a_th_1)/TF.a_th_3
K_theta_DC = pitch_kp*TF.a_th_3/(TF.a_th_2+pitch_kp*TF.a_th_3)

#----------altitude loop-------------
altitude_kp = 2*xsi_h*wn_h/(K_theta_DC*Va0)
altitude_ki = wn_h**2/(K_theta_DC*Va0)
# altitude_zone =

#---------airspeed hold using throttle---------------
airspeed_throttle_kp = (2.0*xsi_v*wn_v*TF.a_v1)/TF.a_v2
airspeed_throttle_ki = wn_v**2/TF.a_v2
