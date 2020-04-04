# dubins_parameters
#   - Dubins parameters that define path between two configurations
#
# mavsim_matlab 
#     - Beard & McLain, PUP, 2012
#     - Update history:  
#         3/26/2019 - RWB

import numpy as np
import sys
sys.path.append('..')


class dubins_parameters:
    def __init__(self):
        self.p_s = np.inf*np.ones((3,1))  # the start position in re^3
        self.chi_s = np.inf  # the start course angle
        self.p_e = np.inf*np.ones((3,1))  # the end position in re^3
        self.chi_e = np.inf  # the end course angle
        self.radius = np.inf  # turn radius
        self.length = np.inf  # length of the Dubins path
        self.center_s = np.inf*np.ones((3,1))  # center of the start circle
        self.dir_s = np.inf  # direction of the start circle
        self.center_e = np.inf*np.ones((3,1))  # center of the end circle
        self.dir_e = np.inf  # direction of the end circle
        self.r1 = np.inf*np.ones((3,1))  # vector in re^3 defining half plane H1
        self.r2 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H2
        self.r3 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H3
        self.n1 = np.inf*np.ones((3,1))  # unit vector in re^3 along straight line path
        self.n3 = np.inf*np.ones((3,1))  # unit vector defining direction of half plane H3

    def update(self, p_s, chi_s, p_e, chi_e, R):

        #check segment length
        ell = np.linalg.norm(p_s - p_e)
        assert(ell >= 2 * R) #TODO book says 3R?

        # decide what dubins path case to use RSR, RSL, exc.
        # get the centers of each case
        cr_s = np.array([p_s]).T + R * Rz(np.pi / 2.0) @ np.array([[np.cos(chi_s), np.sin(chi_s), 0]]).T
        cl_s = np.array([p_s]).T + R * Rz(-np.pi / 2.0) @ np.array([[np.cos(chi_s), np.sin(chi_s), 0]]).T
        cr_e = np.array([p_e]).T + R * Rz(np.pi / 2.0) @ np.array([[np.cos(chi_e), np.sin(chi_e), 0]]).T
        cl_e = np.array([p_e]).T + R * Rz(-np.pi / 2.0) @ np.array([[np.cos(chi_e), np.sin(chi_e), 0]]).T

        # determine minimum
        min_index, l, th1, th2, L = self.min_dubin(cr_s, cl_s, cr_e, cl_e, chi_s, chi_e, R)

        e1 = np.array([[1.0, 0.0, 0.0]]).T
        # R-S-R
        if min_index == 0:
            cs = cr_s
            Ls = 1
            ce = cr_e
            Le = 1

            q1 = (ce - cs) / l
            z1 = cs + R * Rz(-np.pi / 2.0) @ q1
            z2 = ce + R * Rz(-np.pi / 2.0) @ q1
        # R-S-L
        elif min_index == 1:
            cs = cr_s
            Ls = 1
            ce = cl_e
            Le = -1

            q1 = Rz(th2 + np.pi / 2.0) @ e1
            z1 = cs + R * Rz(th2) @ e1
            z2 = ce + R * Rz(th2 + np.pi) @ e1
        # L-S-R
        elif min_index == 2:
            cs = cl_s
            Ls = -1
            ce = cr_e
            Le = 1

            q1 = Rz(th1 + th2 - np.pi / 2.0) @ e1
            z1 = cs + R * Rz(th1 + th2) @ e1
            z2 = ce + R * Rz(th1 + th2 - np.pi) @ e1
        # L-S-L
        elif min_index == 3:
            cs = cl_s
            Ls = -1
            ce = cl_e
            Le = -1

            q1 = (ce - cs) / l
            z1 = cs + R * Rz(np.pi / 2.0) @ q1
            z2 = ce + R * Rz(np.pi / 2.0) @ q1
        else:
            print("invalid min index value")

        z3 = np.array([p_e]).T
        q3 = Rz(chi_e)@e1

        self.p_s = p_s
        self.chi_s = chi_s
        self.p_e = p_e
        self.chi_e = chi_e
        self.radius = R
        self.length = l
        self.center_s = cs
        self.dir_s = Ls
        self.center_e = ce
        self.dir_e = Le
        self.r1 = z1
        self.n1 = q1
        self.r2 = z2
        self.r3 = z3
        self.n3 = q3

    def min_dubin(self, cr_s, cl_s, cr_e, cl_e, chi_s, chi_e, R):

        #compute l theta and L for each case
        li = np.zeros(4)
        th1i = np.zeros(4)
        th2i = np.zeros(4)
        Li = np.zeros(4)

        # R-S-R case 0
        li[0] = np.linalg.norm(cr_s-cr_e)
        th1i[0] = np.arctan2(cr_e[1][0]-cr_s[1][0],cr_e[0][0]-cr_s[0][0])
        th2i[0] = th1i[0] - np.pi/2.0
        Li[0] = li[0]+R*wrap(2.0*np.pi+wrap(th2i[0])-wrap(chi_s-np.pi/2)) \
                + R*wrap(2.0*np.pi+wrap(chi_e-np.pi/2.0)-wrap(th2i[0]))

        #R-S-L case 1
        li[1] = np.linalg.norm(cr_s - cl_e)
        th1i[1] = np.arctan2(cl_e[1][0]-cr_s[1][0],cl_e[0][0]-cr_s[0][0])
        th2i[1] = th1i[1] - np.pi/2.0 + np.arcsin(2.0*R/li[1])
        Li[1] = np.sqrt(li[1]**2-4.0*R**2)+R*wrap(2.0*np.pi+wrap(th2i[1])-wrap(chi_s-np.pi/2.0)) \
                +R*wrap(2.0*np.pi+wrap(th2i[1]+np.pi)-wrap(chi_e+np.pi/2.0))

        #L-S-R case 2
        li[2] = np.linalg.norm(cl_s - cr_e)
        th1i[2] = np.arctan2(cr_e[1][0]-cl_s[1][0],cr_e[0][0]-cl_s[0][0])
        th2i[2] = np.arccos(2.0*R/li[2])
        Li[2] = np.sqrt(li[2]**2-4.0*R**2)+R*wrap(2.0*np.pi+wrap(chi_s+np.pi/2.0)-wrap(th1i[2]+th2i[2])) \
                +R*wrap(2.0*np.pi+wrap(chi_e-np.pi/2.0)-wrap(th1i[2]+th2i[2]-np.pi))

        #L-S-L case 3
        li[3] = np.linalg.norm(cl_s-cl_e)
        th1i[3] = np.arctan2(cl_e[1][0]-cl_s[1][0],cl_e[0][0]-cl_s[0][0])
        th2i[3] = th1i[3] + np.pi/2.0
        Li[3] = li[3] + R*wrap(2.0*np.pi+wrap(chi_s+np.pi/2.0)-wrap(th2i[3])) \
                + R*wrap(2.0*np.pi+wrap(th2i[3])-wrap(chi_e+np.pi/2.0))

        min_index = np.argmin(Li)
        l = li[min_index]
        th1 = th1i[min_index]
        th2 = th2i[min_index]
        L = Li[min_index]

        return min_index, l, th1, th2, L

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0]])


def wrap(x):
    # make x between 0 and 2*pi
    th = 2*np.pi
    th2 = th - 2*np.pi
    while x < th2:
        x += th
    while x > th:
        x -= th
    return x


