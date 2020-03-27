import numpy as np
from math import sin, cos, atan, atan2
import sys

sys.path.append('..')
from message_types.msg_autopilot import msg_autopilot

class path_follower:
    def __init__(self):
        ## tuning parameters?
        self.chi_inf = 1.2  # approach angle for large distance from straight-line path
        self.k_path = 0.3  # proportional gain for straight-line path following
        self.k_orbit = 15.0  # proportional gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = msg_autopilot()  # message sent to autopilot

    def update(self, path, state):
        if path.flag=='line':
            self._follow_straight_line(path, state)
        elif path.flag=='orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):

        #extract needed parameters
        r = path.line_origin
        q = path.line_direction
        qn = q[0][0]
        qe = q[1][0]
        qd = q[2][0]
        p = np.array([[state.pn, state.pe, -state.h]]).T

        #calculate intermediate values
        chi_q = self._wrap(atan2(qe, qn), state.chi)
        Rp = np.array([[cos(chi_q), sin(chi_q), 0.0], \
                       [-sin(chi_q), cos(chi_q), 0.0], \
                       [0.0, 0.0, 1.0]])
        epi = (p-r)
        ep = Rp@epi
        epy = ep[1][0]
        rd = r[2][0]
        k = np.array([0.0, 0.0, 1.0])
        n = np.cross(q.T[0],k)/np.linalg.norm(np.cross(q.T[0],k)) #np.cross requires a 1 d array
        si = epi.T[0]-np.dot(epi.T[0],n)*n #reduced dimension of epi to match n's dimension
        sn = si[0]
        se = si[1]

        #calulate command outputs
        chi_c = chi_q - self.chi_inf*2.0/np.pi*atan(self.k_path*epy) #may need to change as shown on pg 181
        h_c = -rd + np.sqrt(sn**2 + se**2)*(qd/np.sqrt(qn**2+qe**2))

        #package command outputs into message
        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = chi_c
        self.autopilot_commands.altitude_command = h_c
        self.autopilot_commands.phi_feedforward = 0.0 #for straight line

    def _follow_orbit(self, path, state):

        #extract needed parameters
        pn = state.pn
        pe = state.pe
        pd = -state.h
        c = path.orbit_center
        cn = c[0][0]
        ce = c[1][0]
        cd = c[2][0]
        rho = path.orbit_radius
        if path.orbit_direction == 'CW':
            L = 1
        else:
            L = -1
        g = 9.81
        phi_ff = 0.0 #can be changed below in command outputs if d-rho = 0

        #calculate intermediate values
        d = np.sqrt((pn-cn)**2+(pe-ce)**2)
        var_phi = self._wrap(atan2(pe-ce, pn-cn), state.chi)
        orbit_phi = L*atan2(state.Va**2, g*rho) #no wind case

        #calulate command outputs
        chi_c = var_phi + L*(np.pi/2.0 + atan(self.k_orbit*(d-rho)/rho))  #may need to change as shown on pg 183
        h_c = -cd
        if d == 0.0: #Should this be 0 or smaller than a threshold?
            phi_ff = orbit_phi
        else:
            phi_ff = 0.0

        #package command outputs into message
        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = chi_c
        self.autopilot_commands.altitude_command = h_c
        self.autopilot_commands.phi_feedforward = phi_ff

    #TODO replace this with the wrap function in tools
    def _wrap(self, th1, th2):
        while th1-th2 > np.pi:
            th1 = th1 - 2.0 * np.pi
        while th1-th2 < -np.pi:
            th1 = th1 + 2.0 * np.pi
        return th1
