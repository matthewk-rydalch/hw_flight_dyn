"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state

"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msg_state

import parameters.aerosonde_parameters as MAV
from tools.tools import Euler2Quaternion, Quaternion2Euler

class mav_dynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.pn0],  # (0)
                               [MAV.pe0],   # (1)
                               [MAV.pd0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0]])   # (12)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self.Va_b = np.array([[0.], [0.], [0.]])
        self._alpha = 0
        self._beta = 0
        # initialize true_state message
        self.msg_true_state = msg_state()


    ###################################
    # public functions
    def update_state(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_msg_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # redefine parameters from MAV
        mass = MAV.mass
        Jx = MAV.Jx
        Jy = MAV.Jy
        Jz = MAV.Jz
        Jxz = MAV.Jxz

        # defining r1, r2, r3, r4 (have to do with inertia)
        # r is captial gamma (the symbol with a right angle)
        r0 = Jx * Jz - Jxz ** 2  # this is the r value with no subscript, but the variable r was already used
        r1 = (Jxz * (Jx - Jy + Jz)) / r0
        r2 = (Jz * (Jz - Jy) + Jxz ** 2) / r0
        r3 = Jz / r0
        r4 = Jxz / r0
        r5 = (Jz - Jx) / Jy
        r6 = Jxz / Jy
        r7 = ((Jx - Jy) * Jx + Jxz ** 2) / r0
        r8 = Jx / r0

        # position kinematics eq B.1
        pn_dot = (e1 ** 2 + e0 ** 2 - e2 ** 2 - e3 ** 2) * u + 2 * (e1 * e2 - e3 * e0) * v + 2 * (e1 * e3 + e2 * e0) * w
        pe_dot = 2 * (e1 * e2 + e3 * e0) * u + (e2 ** 2 + e0 ** 2 - e1 ** 2 - e3 ** 2) * v + 2 * (e2 * e3 - e1 * e0) * w
        pd_dot = 2 * (e1 * e3 - e2 * e0) * u + 2 * (e2 * e3 + e1 * e0) * v + (e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2) * w

        # position dynamics eq B.2
        u_dot = (r * v - q * w) + fx / mass
        v_dot = (p * w - r * u) + fy / mass
        w_dot = (q * u - p * v) + fz / mass

        # rotational kinematics eq B.3
        e0_dot = 1 / 2 * (-p * e1 - q * e2 - r * e3)
        e1_dot = 1 / 2 * (p * e0 + r * e2 - q * e3)
        e2_dot = 1 / 2 * (q * e0 - r * e1 + p * e3)
        e3_dot = 1 / 2 * (r * e0 + q * e1 - p * e2)

        # This block is used to normalize the quaternions, but that has already been done in the update state function
        # # rotational kinematics eq B.3 modified
        # lmb = 1000 #lambda, best value was given on page 257
        # normE = np.sqrt(e0**2+e1**2+e2**2+e3**2) #also defined the same way in update state function
        # e0_dot = 1/2*(lmb*(1.0-normE**2)*e0 - p*e1 - q*e2 - r*e3)
        # e1_dot = 1/2*(p*e0 + lmb*(1.0-normE**2)*e1 + r*e2 - q*e3)
        # e2_dot = 1/2*(q*e0 - r*e1 + lmb*(1.0-normE**2)*e2 + p*e3)
        # e3_dot = 1/2*(r*e0 + q*e1 - p*e2 + lmb*(1.0-normE**2)*e3)

        # rotatonal dynamics eq. B.4
        p_dot = (r1 * p * q - r2 * q * r) + (r3 * l + r4 * n)
        q_dot = (r5 * p * r - r6 * (p ** 2 - r ** 2)) + (1 / Jy * m)
        r_dot = (r7 * p * q - r1 * q * r) + (r4 * l + r8 * n)

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot


    def _update_velocity_data(self, wind=np.zeros((6,1))):
        # compute airspeed
        self._wind = np.array([[wind[0][0]+wind[3][0]],
                              [wind[1][0]+wind[4][0]],
                              [wind[2][0]+wind[5][0]]])
        Va_b = np.array([[self._state[3][0]],#-self._wind[0][0]],
                         [self._state[4][0]],#-self._wind[1][0]],
                         [self._state[5][0]]])#-self._wind[2][0]]])

        self._Va = np.linalg.norm(Va_b)

        if self._Va == 0.0:
            self._alpha = 0.0
            self._beta = 0.0
        else:
            # compute angle of attack
            self._alpha = np.arctan2(Va_b[2], Va_b[0])[0]

            # compute sideslip angle
            self._beta = np.arcsin(Va_b[1][0]/self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        print('e, t, a , r = ', delta)
        #longitudinal coefficients
        C_L_0 = MAV.C_L_0
        C_L_alpha = MAV.C_L_alpha
        C_L_q = MAV.C_L_q
        C_L_delta_e = MAV.C_L_delta_e
        C_D_0 = MAV.C_D_0
        C_D_alpha = MAV.C_D_alpha
        C_D_p = MAV.C_D_p
        C_D_q = MAV.C_D_q
        C_D_delta_e = MAV.C_D_delta_e
        C_m_0 = MAV.C_m_0
        C_m_alpha = MAV.C_m_alpha
        C_m_q = MAV.C_m_q
        C_m_delta_e = MAV.C_m_delta_e
        C_prop = MAV.C_prop
        M = MAV.M
        alpha0 = MAV.alpha0
        epsilon = MAV.epsilon

        #lateral coefficients
        C_Y_0 = MAV.C_Y_0
        C_Y_beta = MAV.C_Y_beta
        C_Y_p = MAV.C_Y_p
        C_Y_r = MAV.C_Y_r
        C_Y_delta_a = MAV.C_Y_delta_a
        C_Y_delta_r = MAV.C_Y_delta_r
        C_ell_0 = MAV.C_ell_0
        C_ell_beta = MAV.C_ell_beta
        C_ell_p = MAV.C_ell_p
        C_ell_r = MAV.C_ell_r
        C_ell_delta_a = MAV.C_ell_delta_a
        C_ell_delta_r = MAV.C_ell_delta_r
        C_n_0 = MAV.C_n_0
        C_n_beta = MAV.C_n_beta
        C_n_p = MAV.C_n_p
        C_n_r = MAV.C_n_r
        C_n_delta_a = MAV.C_n_delta_a
        C_n_delta_r = MAV.C_n_delta_r

        #common terms
        al = self._alpha
        beta = self._beta
        Va = self._Va
        s_al = np.sin(al)
        c_al = np.cos(al)
        [th, phi, psi] = Quaternion2Euler(self._state[6:10])
        p = self._state[10][0]
        q = self._state[11][0]
        r = self._state[12][0]
        delta_e = delta[0][0]
        delta_t = delta[1][0]
        delta_a = delta[2][0]
        delta_r = delta[3][0]

        #coefficients
        CL = C_L_0+C_L_alpha*al
        CD = C_D_0+C_D_alpha*al
        Cx_a = -CD*c_al+CL*s_al
        Cxq_a = -C_D_q*c_al+C_L_q*s_al
        Cx_de_a = -C_D_delta_e*c_al+C_L_delta_e*s_al
        Cz_a = -CD*s_al-CL*c_al
        Czq_a = -C_D_q*s_al-C_L_q*c_al
        Cz_de_a = -C_D_delta_e*s_al

        # compute t h r u s t and torque due to p r o p ell e r ( See addendum by McLain)
        # map delta_t throttle command(0 t o 1) in to motor input voltage
        V_in = MAV.V_max*delta_t
        # Quadratic formula to solve for motor speed
        a = MAV.C_Q0*MAV.rho*np.power(MAV.D_prop,5) \
        / ((2.*np.pi)**2)
        b = (MAV.C_Q1*MAV.rho*np.power(MAV.D_prop,4) \
        / (2.*np.pi))*Va+MAV.KQ**2/MAV.R_motor
        c = MAV.C_Q2*MAV.rho*np.power(MAV.D_prop,3)*Va**2-(MAV.KQ/MAV.R_motor)*V_in+MAV.KQ*MAV.i0
        # Consider only positive root
        Omega_op = (-b+np.sqrt(b**2-4*a*c))/(2.*a)
        # compute advance ratio
        J_op = 2*np.pi*Va/(Omega_op*MAV.D_prop)
        # compute nondimens ionalized coefficients of thrust and torque
        C_T = MAV.C_T2*J_op**2+MAV.C_T1*J_op+MAV.C_T0
        C_Q = MAV.C_Q2*J_op**2+MAV.C_Q1*J_op+MAV.C_Q0
        # add thrust and torque due to propeller
        n = Omega_op/(2*np.pi)
        #Tp = MAV.rho*n**2*np.power(MAV.D_prop,4)*C_T
        Tp = C_T*MAV.rho*Omega_op**2*MAV.D_prop**4/(2*np.pi)**2
        Qm = -MAV.rho*n**2*np.power(MAV.D_prop,5)*C_Q
        # Tp = C_T*MAV.rho*Omega_op**2*MAV.D_prop**4/(2.0*np.pi)**2 #tried these, they match the new material
        Qp = MAV.KQ*(1/MAV.R_motor*(V_in-MAV.K_V*Omega_op)-MAV.i0)


        #forces
        fg = np.array([[-MAV.mass*MAV.gravity*np.sin(th)],
                       [MAV.mass*MAV.gravity*np.cos(th)*np.sin(phi)],
                       [MAV.mass*MAV.gravity*np.cos(th)*np.cos(phi)]])

        if Va == 0.0:
            fa = np.array([[0.0],[0.0],[0.0]])
        else:
            fa = 1/2*MAV.rho*Va**2*MAV.S_wing*np.array([[Cx_a+Cxq_a*MAV.c/(2*Va)*q+Cx_de_a*delta_e],
                                                        [C_Y_0+C_Y_beta*beta+C_Y_p*MAV.b/(2.0*Va)*p+C_Y_r*MAV.b/(22.0*Va)*r+C_Y_delta_a*delta_a+C_Y_delta_r*delta_r],
                                                        [Cz_a+Czq_a*MAV.c/(2*Va)*q+Cz_de_a*delta_e]])

        fp = np.array([[Tp],
                       [0.0],
                       [0.0]])

        if Va == 0.0:
            Ma = np.array([[0.0],[0.0],[0.0]])
        else:
            Ma = 1/2*MAV.rho*Va**2*MAV.S_wing*np.array([[MAV.b*(C_ell_0+C_ell_beta*beta+C_ell_p*MAV.b/(2*Va)*p+C_ell_r*MAV.b/(2*Va)*r+C_ell_delta_a*delta_a+C_ell_delta_r*delta_r)],
                                                        [MAV.c*(C_m_0+C_m_alpha*al+C_m_q*MAV.c/(2*Va)*q+C_m_delta_e*delta_e)],
                                                        [MAV.b*(C_n_0+C_n_beta*beta+C_n_p*MAV.b/(2*Va)*p+C_n_r*MAV.b/(2*Va)*r+C_n_delta_a*delta_a+C_n_delta_r*delta_r)]])

        Mt = np.array([[Qm],
                       [0.0],
                       [0.0]])

        fx = fg[0][0]+fa[0][0]+fp[0][0]
        fy = fg[1][0]+fa[1][0]+fp[1][0]
        fz = fa[2][0]+fp[2][0]+fg[2][0]
        Mx = Ma[0][0]+Mt[0][0]
        My = Ma[1][0]+Mt[1][0]
        Mz = Ma[2][0]+Mt[2][0]

        # fx = 0.0
        # fy = 0.0
        # fz = 0.0
        # Mx = 0.0
        # My = 0.0
        # Mz = 0.0

        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz

        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    def _update_msg_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.Va = self._Va
        self.msg_true_state.alpha = self._alpha
        self.msg_true_state.beta = self._beta
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        # Vg_b = self.Va_b+self._wind
        # self.msg_true_state.Vg = np.linalg.norm(Vg_b)
        # self.msg_true_state.gamma = np.arctan2(Vg_b[2], Vg_b[0])
        # self.msg_true_state.chi = np.arctan2(Vg_b[1], Vg_b[0])
        Vg_b = self.Va_b+self._wind
        R = self._Euler2Rotation(phi, theta, psi)
        Vg_i = R.T@Vg_b
        self.msg_true_state.Vg = np.linalg.norm(Vg_b)
        self.msg_true_state.gamma = np.arctan2(Vg_i[2], np.sqrt(Vg_i[0]**2+Vg_i[1]**2))
        self.msg_true_state.chi = np.arctan2(Vg_i[1], Vg_i[0])
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)
        self.msg_true_state.wn = self._wind.item(0)
        self.msg_true_state.we = self._wind.item(1)

    def _Euler2Rotation(self, phi, theta, psi):
        """
        Converts euler angles to rotation matrix (R_b^i, i.e., body to inertial)
        """
        # only call sin and cos once for each angle to speed up rendering
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        R_roll = np.array([[1, 0, 0],
                           [0, c_phi, s_phi],
                           [0, -s_phi, c_phi]])
        R_pitch = np.array([[c_theta, 0, -s_theta],
                            [0, 1, 0],
                            [s_theta, 0, c_theta]])
        R_yaw = np.array([[c_psi, s_psi, 0],
                          [-s_psi, c_psi, 0],
                          [0, 0, 1]])
        R = R_roll @ R_pitch @ R_yaw  # inertial to body (Equation 2.4 in book)
        return R.T  # transpose to return body to inertial
