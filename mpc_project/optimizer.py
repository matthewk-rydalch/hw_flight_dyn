import sys #TODO what is this and the  next line for?

sys.path.append('..')
import numpy as np
from scipy.optimize import minimize

# from tools.tools import Euler2Quaternion
from mpc_project.dynamic_model import dynamic_model

class optimizer():
    def __init__(self, Ts):
        self.Ts = Ts
        self.predict_dynamics = dynamic_model(Ts)
        self.u = 0.0
        self.u_bound = np.pi/6.0 #plus or minus #TODO get the actual value for this

    def update(self, xhat, target_hat):

        u = self.optimize_horizon(xhat, target_hat)
        ###should be part of the optimizer
        waypoint = self.predict_dynamics.update(xhat, u)
        # waypoint = np.array([[target_hat.item(0), target_hat.item(1), -100.0]]).T #TODO this is for debugging
        ###

        # waypoint = np.array([[100.0, 0.0, -100.0]]).T + np.array([[xhat.item(0), 0.0, 0.0]]).T

        return waypoint

    def optimize_horizon(self, xhat, target_hat):
        #TODO need to change this for a horizon greater than 1

        # define initial state and input
        xm = xhat.item(0)
        ym = xhat.item(1)
        thm = xhat.item(3) #chi
        Vg = xhat.item(4)
        target_xm = target_hat.item(0)
        target_ym = target_hat.item(1)
        target_thm = target_hat.item(3)
        target_Vg = target_hat.item(4)
        #assume target trajectory and velocity won't change, so there is no input
        #initial mav input is global, self.u
        xm = np.array([[xm, ym, thm, target_xm, target_ym, target_thm]]).T

        #bounds
        bds = np.array([(-self.u_bound, self.u_bound)])

        # # define equality constraints
        # cons = ({'type': 'eq',
        #          'fun': lambda x: np.array([
        #              #TODO figure out where input constraints are set.  Perhaps here?
        #              # x[3] ** 2 + x[4] ** 2 + x[5] ** 2 - Va ** 2,  # magnitude of velocity vector is Va
        #              # x[4],  # v=0, force side velocity to be zero
        #              # x[6] ** 2 + x[7] ** 2 + x[8] ** 2 + x[9] ** 2 - 1.,  # force quaternion to be unit length
        #              # x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
        #              # x[9],  # e3=0
        #              # x[10],  # p=0  - angular rates should all be zero
        #              # x[11],  # q=0
        #              # x[12],  # r=0
        #          ]),
        #          #TODO figure out what the jacobian is of
        #          'jac': lambda x: np.array([
        #              [0., 0., 0., 2 * x[3], 2 * x[4], 2 * x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #              [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #              [0., 0., 0., 0., 0., 0., 2 * x[6], 2 * x[7], 2 * x[8], 2 * x[9], 0., 0., 0., 0., 0., 0., 0.],
        #              [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #              [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        #              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        #              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        #              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        #          ])
        #          })
        # solve the minimization problem to find the trim states and inputs
        res = minimize(self.mpc_objective, self.u, method='SLSQP', args=(xm, Vg, target_Vg),
                       bounds=bds, options={'ftol': 1e-10, 'disp': False})

        # state and input and return
        optimized_input = res.x[0]
        # print('optimized input chi_dot', optimized_input)
        return optimized_input

    # objective function to be minimized
    def mpc_objective(self, u, x, Vg, target_Vg):
        # define initial state and input
        xm = x.item(0)
        ym = x.item(1)
        thm = x.item(2) #chi
        target_xm = x.item(3)
        target_ym = x.item(4)
        target_thm = x.item(5)

        thp = thm + u * self.Ts
        xp = xm + Vg * np.cos(thp) * self.Ts
        yp = ym + Vg * np.sin(thp) * self.Ts

        target_thp = target_thm
        target_xp = target_xm + target_Vg * np.cos(target_thp) * self.Ts
        target_yp = target_ym + target_Vg * np.sin(target_thp) * self.Ts

        # pd_dot = -Va * np.sin(gamma)
        # u_dot = 0.0
        # v_dot = 0.0
        # w_dot = 0.0
        # phi_dot = 0.0
        # theta_dot = 0.0
        # psi_dot = 0.0  # Va/R*np.cos(gamma)
        # p_dot = 0.0
        # q_dot = 0.0
        # r_dot = 0.0
        #
        # mav._state = np.array([x[0:13]]).T
        # mav._update_velocity_data()
        # forces_moments = mav._forces_moments(np.array([x[13:17]]).T)
        # xdot = mav._derivatives(mav._state, forces_moments)

        J_vec = np.zeros((3, 1))
        J_vec[0] = xp - target_xp
        J_vec[1] = yp - target_yp
        J_vec[2] = thp - target_thp

        J = np.linalg.norm(J_vec) ** 2
        return J