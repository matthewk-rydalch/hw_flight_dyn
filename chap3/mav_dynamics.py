"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
part of mavsimPy
    - Beard & McLain, PUP, 2012
    - Update history:  
        12/17/2018 - RWB
        1/14/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msg_state

import parameters.aerosonde_parameters as MAV
from tools.tools import Quaternion2Euler

class mav_dynamics:
    def __init__(self, Ts):
        self.ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        self._state = np.array([[MAV.pn0, MAV.pe0, MAV.pd0, MAV.u0, MAV.v0, MAV.w0, MAV.e0, MAV.e1, MAV.e2, MAV.e3, MAV.p0, MAV.q0, MAV.r0 #the quaternion values come from the initial states phi0, th0, psi0
                                ]]).T
        self.msg_true_state = msg_state()

    ###################################
    # public functions
    def update_state(self, forces_moments):
        '''

            Integrate the differential equations defining dynamics. 
            Inputs are the forces and moments on the aircraft.
            Ts is the time step between function calls.
        '''

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self.ts_simulation
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

        #redefine parameters from MAV
        mass = MAV.mass
        Jx = MAV.Jx
        Jy = MAV.Jy
        Jz = MAV.Jz
        Jxz = MAV.Jxz     

        # defining r1, r2, r3, r4 (have to do with inertia)
        # r is captial gamma (the symbol with a right angle)
        r0 = Jx*Jz-Jxz**2 #this is the r value with no subscript, but the variable r was already used
        r1 = (Jxz*(Jx-Jy+Jz))/r0
        r2 = (Jz*(Jz-Jy)+Jxz**2)/r0
        r3 = Jz/r0
        r4 = Jxz/r0
        r5 = (Jz-Jx)/Jy
        r6 = Jxz/Jy
        r7 = ((Jx-Jy)*Jx+Jxz**2)/r0
        r8 = Jx/r0

        # position kinematics eq B.1
        pn_dot = (e1**2+e0**2-e2**2-e3**2)*u + 2*(e1*e2-e3*e0)*v + 2*(e1*e3+e2*e0)*w
        pe_dot = 2*(e1*e2+e3*e0)*u + (e2**2+e0**2-e1**2-e3**2)*v + 2*(e2*e3-e1*e0)*w
        pd_dot = 2*(e1*e3-e2*e0)*u + 2*(e2*e3+e1*e0)*v + (e3**2+e0**2-e1**2-e2**2)*w

        # position dynamics eq B.2
        u_dot = (r*v-q*w) + fx/mass
        v_dot = (p*w-r*u) + fy/mass
        w_dot = (q*u-p*v) + fz/mass

        # rotational kinematics eq B.3
        e0_dot = 1/2*(-p*e1 - q*e2 - r*e3)
        e1_dot = 1/2*(p*e0 + r*e2 - q*e3)
        e2_dot = 1/2*(q*e0 - r*e1 + p*e3)
        e3_dot = 1/2*(r*e0 + q*e1 - p*e2)

        #This block is used to normalize the quaternions, but that has already been done in the update state function
        # # rotational kinematics eq B.3 modified
        # lmb = 1000 #lambda, best value was given on page 257
        # normE = np.sqrt(e0**2+e1**2+e2**2+e3**2) #also defined the same way in update state function
        # e0_dot = 1/2*(lmb*(1.0-normE**2)*e0 - p*e1 - q*e2 - r*e3)
        # e1_dot = 1/2*(p*e0 + lmb*(1.0-normE**2)*e1 + r*e2 - q*e3)
        # e2_dot = 1/2*(q*e0 - r*e1 + lmb*(1.0-normE**2)*e2 + p*e3)
        # e3_dot = 1/2*(r*e0 + q*e1 - p*e2 + lmb*(1.0-normE**2)*e3)

        # rotatonal dynamics eq. B.4
        p_dot = (r1*p*q-r2*q*r) + (r3*l+r4*n)
        q_dot = (r5*p*r-r6*(p**2-r**2)) + (1/Jy*m)
        r_dot = (r7*p*q-r1*q*r) + (r4*l+r8*n)

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_msg_true_state(self):
        # update the true state message:
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)

