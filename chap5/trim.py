"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/5/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion

def compute_trim(mav, Va, gamma):

    # define initial state and input
    pn0 = mav._state[0][0]
    pe0 = mav._state[1][0]
    pd0 = mav._state[2][0]
    u0 = mav._state[3][0]
    v0 = mav._state[4][0]
    w0 = mav._state[5][0]
    e0_0 = mav._state[6][0]
    e1_0 = mav._state[7][0]
    e2_0 = mav._state[8][0]
    e3_0 = mav._state[9][0]
    p0 = mav._state[10][0]
    q0 = mav._state[11][0]
    r0 = mav._state[12][0]
    # delta_a, delta_e, delta_r, delta_t
    delta_a0 = 0.
    delta_e0 = 0.
    delta_r0 = 0.
    delta_t0 = 0.5
    state0 = np.array([[pn0, pe0, pd0, u0, v0, w0, e0_0, e1_0, e2_0, e3_0, p0, q0, r0]]).T
    delta0 = np.array([[delta_a0, delta_e0, delta_r0, delta_t0]]).T
    x0 = np.concatenate((state0, delta0), axis=0)
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7], # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9], # e3=0
                                x[10], # p=0  - angular rates should all be zero
                                x[11], # q=0
                                x[12], # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs
    res = minimize(trim_objective, x0, method='SLSQP', args = (mav, Va, gamma),
                   constraints=cons, options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = np.array([res.x[13:17]]).T
    return trim_state, trim_input

# objective function to be minimized
def trim_objective(x, mav, Va, gamma):

    # define initial state and input
    pd_dot = -Va*np.sin(gamma)
    u_dot = 0.0
    v_dot = 0.0
    w_dot = 0.0
    phi_dot = 0.0
    theta_dot = 0.0
    psi_dot = 0.0#Va/R*np.cos(gamma)
    p_dot = 0.0
    q_dot = 0.0
    r_dot = 0.0

    mav._state = np.array([x[0:13]]).T
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(np.array([x[13:17]]).T)
    # print('forces & moments = ', forces_moments)
    xdot = mav._derivatives(mav._state, forces_moments)
    # print('xdot = ', xdot)
    # print('mav state = ', mav._state)

    J_vec = np.zeros((11,1))
    J_vec[0] = pd_dot - xdot[2][0]
    J_vec[1] = u_dot - xdot[3][0]
    J_vec[2] = v_dot - xdot[4][0]
    J_vec[3] = w_dot - xdot[5][0]
    J_vec[4] = 0.0 - xdot[6][0] #since the R value is really big/infinite, these dot values are 0, 0, 0, 0.
    J_vec[5] = 0.0 - xdot[7][0]
    J_vec[6] = 0.0 - xdot[8][0]
    J_vec[7] = 0.0 - xdot[9][0]
    J_vec[8] = p_dot - xdot[10][0]
    J_vec[9] = q_dot - xdot[11][0]
    J_vec[10] = r_dot - xdot[12][0]

    J = np.linalg.norm(J_vec)**2

    return J

