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
    pn0 = 0.0
    pe0 = 0.0
    h0 = 20.0
    u0 = 20.0
    v0 = 0.0
    w0 = 0.0
    e0_0 = 0.0
    e1_0 = 0.0
    e2_0 = 0.0
    e3_0 = 0.0
    p0 = 0.0
    q0 = 0.0
    r0 = 0.0
    delta_e0 = -0.1
    delta_t0 = 0.8
    delta_a0 = 0.03
    delta_r0 = 0.0
    state0 = np.array([[pn0, pe0, h0, u0, v0, w0, e0_0, e1_0, e2_0, e3_0, p0, q0, r0]]).T
    delta0 = np.array([[delta_e0, delta_t0, delta_a0, delta_r0]]).T
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
    pn_dot = 0.0
    pe_dot = 0.0
    h_dot = Va*np.sin(gamma)
    u_dot = 0.0
    v_dot = 0.0
    w_dot = 0.0
    phi_dot = 0.0
    theta_dot = 0.0
    psi_dot = 0.0#Va/R*np.cos(gamma)
    p_dot = 0.0
    q_dot = 0.0
    r_dot = 0.0

    J_vec = np.zeros((13,1))
    J_vec[0] = pn_dot - mav._state[0]
    J_vec[1] = pe_dot - mav._state[1]
    J_vec[2] = h_dot - mav._state[2]
    J_vec[3] = u_dot - mav._state[3]
    J_vec[4] = v_dot - mav._state[4]
    J_vec[5] = w_dot - mav._state[5]
    J_vec[6] = 0.0 - mav._state[6] #since the R value is really big/infinite, these dot values are 0.
    J_vec[7] = 0.0 - mav._state[7]
    J_vec[8] = 0.0 - mav._state[8]
    J_vec[9] = 0.0 - mav._state[9]
    J_vec[10] = p_dot - mav._state[10]
    J_vec[11] = q_dot - mav._state[11]
    J_vec[12] = r_dot - mav._state[12]

    J = np.linalg.norm(J_vec)

    return J

