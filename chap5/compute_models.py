"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion, Quaternion2Euler
from tools.transfer_function import transfer_function
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts


def a_variables(mav):
    # parameters
    rho = MAV.rho
    S = MAV.S_wing
    b = MAV.b
    C_p_p = MAV.C_p_p
    Va = mav._Va
    C_p_delta_a = MAV.C_p_delta_a
    g = MAV.gravity
    Vg = mav._Vg  # this is in the body frame.  Is that right? (its a magnitude so it shouldn't matter right?)
    c = MAV.c
    Jy = MAV.Jy
    C_m_q = MAV.C_m_q
    C_m_a = MAV.C_m_alpha  # is this the right variable?
    C_m_delta_e = MAV.C_m_delta_e
    Va_star = Va  # is this right since it is right after the trim function?
    mass = MAV.mass
    C_D_0 = MAV.C_D_0
    C_D_al = MAV.C_D_alpha
    al_star = mav._alpha
    C_D_delta_e = MAV.C_D_delta_e
    S_prop = MAV.S_prop
    C_prop = MAV.C_prop
    k_motor = MAV.k_motor
    chi_star = mav.msg_true_state.chi
    C_Y_B = MAV.C_Y_beta
    C_Y_delta_r = MAV.C_Y_delta_r

    a_phi_1 = -0.5 * rho * Va ** 2 * S * b * C_p_p * b / (2.0 * Va)
    a_phi_2 = 0.5 * rho * Va ** 2 * S * b * C_p_delta_a

    a_th_1 = -rho * Va ** 2 * c * S / (2.0 * Jy) * C_m_q * c / (2.0 * Va)
    a_th_2 = -rho * Va ** 2 * c * S / (2.0 * Jy) * C_m_a
    a_th_3 = rho * Va ** 2 * c * S / (2.0 * Jy) * C_m_delta_e

    a_v_1 = rho * Va_star * S / mass * (
                C_D_0 + C_D_al * al_star + C_D_delta_e * delta_e_star) + rho * S_prop / mass * C_prop * Va_star
    a_v_2 = rho * S_prop / mass * C_prop * k_motor ** 2 * delta_t_star
    a_v_3 = g * np.cos(th_star - chi_star)

    a_B_1 = -rho * Va * S / (2.0 * mass) * C_Y_B
    a_B_2 = rho * Va * S / (2.0 * mass) * C_Y_delta_r

    return a_phi_1, a_phi_2, a_th_1, a_th_2, a_th_3, a_v_1, a_v_2, a_v_3, a_B_1, a_B_2


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    u_star = trim_state[3]
    v_star = trim_state[4]
    w_star = trim_state[5]
    e_vec = trim_state[6:10]
    [phi, theta, psi] = Quaternion2Euler(e_vec)
    delta_e_star = trim_input[1][0]
    delta_t_star = trim_input[3][0]
    th_star = theta #is this right?

    #parameters
    rho = MAV.rho
    S = MAV.S_wing
    b = MAV.b
    C_p_p = MAV.C_p_p
    Va = mav._Va
    C_p_delta_a = MAV.C_p_delta_a
    g = MAV.gravity
    Vg = mav._Vg #this is in the body frame.  Is that right? (its a magnitude so it shouldn't matter right?)
    c = MAV.c
    Jy = MAV.Jy
    C_m_q = MAV.C_m_q
    C_m_a = MAV.C_m_alpha #is this the right variable?
    C_m_delta_e = MAV.C_m_delta_e
    Va_star = Va #is this right since it is right after the trim function?
    mass = MAV.mass
    C_D_0 = MAV.C_D_0
    C_D_al = MAV.C_D_alpha
    al_star = mav._alpha
    C_D_delta_e = MAV.C_D_delta_e
    S_prop = MAV.S_prop
    C_prop = MAV.C_prop
    k_motor = MAV.k_motor
    chi_star = mav.msg_true_state.chi
    C_Y_B = MAV.C_Y_beta
    C_Y_delta_r = MAV.C_Y_delta_r

    [a_phi_1, a_phi_2, a_th_1, a_th_2, a_th_3, a_v_1, a_v_2, a_v_3, a_B_1, a_B_2] = a_variables(mav)


    #phi to delta_a
    num = np.array([[a_phi_2]])
    den = np.array([[1, a_phi_1, 0]])
    T_phi_delta_a = transfer_function(num, den, Ts)

    #chi to phi
    num = np.array([[g/Vg]])
    den = np.array([[1, 0]])
    T_chi_phi = transfer_function(num, den, Ts)

    #theta to delta_e
    num = np.array([[a_th_3, 0.0]])
    den = np.array([[1, a_th_1, a_th_2]])
    T_theta_delta_e = transfer_function(num, den, Ts)

    #h to theta
    num = np.array([[Va]])
    den = np.array([[1, 0]])
    T_h_theta = transfer_function(num, den, Ts)

    #h to Va
    num = np.array([[theta]])
    den = np.array([[1, 0]])
    T_h_Va = transfer_function(num, den, Ts)

    #Va to delta_t
    num = np.array([[a_v_2]])
    den = np.array([[1, a_v_1]])
    T_Va_delta_t = transfer_function(num, den, Ts)

    #Va to theta
    num = np.array([[a_v_3]])
    den = np.array([[1, a_v_1]])
    T_Va_theta = transfer_function(num, den, Ts)

    #beta to delta_r
    num = np.array([[a_B_2]])
    den = np.array([[1, a_B_1]])
    T_beta_delta_r = transfer_function(num, den, Ts)

    return T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r


def compute_ss_model(mav, trim_state, trim_input):

    #NOT FINISHED
    # #retrieve parameters from trim
    # e_vec = trim_state[6:10]
    # [phi_star, th_star, psi_star] = Quaternion2Euler(e_vec)
    # u_star = trim_state[3]
    # w_star = trim_state[5]
    #
    # #retrieve parameters from aerosonde_parameters
    # g = MAV.gravity
    #
    # #define coefficients
    # Xu = u_star*rho*S/m*(C_X_0+C_X_alpha*alpha_star+C_X_delta_e*delta_e) - rho*S*w_star*C_X_alpha/(2.0*m) + rho*S*c*C_X_q*u_star*q_star/(4.0*m*Va_star) - rho*S_prop*C_prop*w_star/m
    # Xw = -q_star + w_star*rho*S/m*(C_X_0+C_X_alpha*alpha_star+C_X_delta_e*delta_e) + rho*S*c*C_X_q*w_star*q_star/(4.0*m*Va_star) + rho*S*C_X_alpha*u_star/(2.0*m) - rho*S_prop*C_prop*w_star/m
    # Xq = -w_star + rho*Va_star*S_C_X_q*c/(4.0*m)
    # X_delta_e = rho*Va_star**2*S*C_x_delta_e/(2.0*m)
    # X_delta_t = rho*S_prop*C_prop*k**2*delta_t_star/m
    # Zu = q_star + u_star*rho*S/m*(C_Z_0+C_Z_alpha*alpha_star+C_Z_delta_e*delta_e_star) - rho*S*C_Z_alpha*w_star/(2.0*m)
    #
    # A_lon = np.array([[Xu, Xw, Xq, -g*np.cos(th_star), 0.0],
    #                   [Zu, Zw, Zq, -g*np.sin(th_star), 0.0],
    #                   [Mu, Mw, Mq, 0.0, 0.0],
    #                   [0.0, 0.0, 1.0, 0.0, 0.0],
    #                   [np.sin(th_star), -np.cos(th_star), 0.0, u_star*np.cos(th_star)+w_star*np.sin(th_star), 0.0]])

     return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
     return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    return x_quat

def f_euler(mav, x_euler, input):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state
    return f_euler_

def df_dx(mav, x_euler, input):
    # take partial of f_euler with respect to x_euler
    return A

def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to delta
    return B

def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    return dThrust

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    return dThrust