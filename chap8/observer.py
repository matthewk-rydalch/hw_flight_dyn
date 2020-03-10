"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
import parameters.aerosonde_parameters as MAV
from tools.rotations import Euler2Rotation
from tools.wrap import wrap

from message_types.msg_state import msg_state

class observer:
    def __init__(self, ts_control):
        # initialized estimated state message
        self.estimated_state = msg_state()
        # use alpha filters to low pass filter gyros and accels
        self.lpf_gyro_x = alpha_filter(alpha=0.5)
        self.lpf_gyro_y = alpha_filter(alpha=0.5)
        self.lpf_gyro_z = alpha_filter(alpha=0.5)
        self.lpf_accel_x = alpha_filter(alpha=0.5)
        self.lpf_accel_y = alpha_filter(alpha=0.5)
        self.lpf_accel_z = alpha_filter(alpha=0.5)
        # use alpha filters to low pass filter static and differential pressure
        self.lpf_static = alpha_filter(alpha=0.9)
        self.lpf_diff = alpha_filter(alpha=0.5)
        # ekf for phi and theta
        self.attitude_ekf = ekf_attitude(self.estimated_state)
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = ekf_position()

    def update(self, measurements):

        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurements.gyro_x) - SENSOR.gyro_x_bias
        self.estimated_state.q = self.lpf_gyro_y.update(measurements.gyro_y) - SENSOR.gyro_y_bias
        self.estimated_state.r = self.lpf_gyro_z.update(measurements.gyro_z) - SENSOR.gyro_z_bias

        # invert sensor model to get altitude and airspeed
        self.estimated_state.h = self.lpf_static.update(measurements.static_pressure)/(MAV.rho*MAV.gravity)
        self.estimated_state.Va = np.sqrt(1/MAV.rho*self.lpf_static.update(measurements.diff_pressure))

        # estimate phi and theta with simple ekf
        # self.attitude_ekf.update(self.estimated_state, measurements)

        # estimate pn, pe, Vg, chi, wn, we, psi
        # self.position_ekf.update(self.estimated_state, measurements)

        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state

class alpha_filter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y = self.alpha*self.y+(1-self.alpha)*u
        return self.y

class ekf_attitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self, initial_state):
        Q_tune = 1 #TODO tune this
        self.Q = Q_tune*np.identity(2)
        self.Q_gyro = SENSOR.gyro_sigma**2*np.identity(3)
        self.R_accel = SENSOR.accel_sigma**2*np.identity(3)
        self.N = 5  #TODO get the right number of prediction step per sample
        self.xhat =  np.array([[initial_state.phi, initial_state.theta]]).T# initial state: phi, theta
        self.P = np.identity(2)
        self.Ts = SIM.ts_control/self.N

    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u), but this one doesn't actually use u
        p = state.p
        q = state.q
        r = state.r
        phi = x[0][0]
        theta = x[1][0]

        _f = np.array([[p+q*np.sin(phi)*np.tan(theta)+r*np.cos(phi)*np.tan(theta), \
                        q*np.cos(phi)-r*np.sin(phi)]])
        return _f

    def h(self, x, state):
        # measurement model y
        p = state.p
        q = state.q
        r = state.r
        Va = state.Va
        phi = x[0][0]
        theta = x[1][0]
        g = MAV.gravity

        _h = np.array([[q*Va*np.sin(theta)+g*np.sin(theta)],
                       [r*Va*np.cos(theta)-p*Va*np.sin(theta)-g*np.cos(theta)*np.sin(phi)],
                       [-q*Va*np.cos(theta)-g*np.cos(theta)*np.cos(phi)]])
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            Tp = self.Ts
            self.xhat = self.xhat +Tp*self.f(self.xhat, state) #This one doesn't actually use u
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # compute G matrix for gyro noise
            G = np.array([[1.0, np.sin(state.phi)*np.tan(state.theta), np.cos(state.phi)*np.tan(state.theta), 0],\
                           0.0, np.cos(state.phi), -np.sin(state.phi), 0.0])
            # update P with continuous time model
            self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            ## convert to discrete time models
            A_d = np.identity(2) + A*Tp + A@A*Tp**2
            #TODO figure out how to use G_d
            # G_d =
            # update P with discrete time model
            self.P = A_d@self.P@A_d.T+Tp**2@self.Q

    def measurement_update(self, state, measurement):
        # measurement updates
        threshold = 2.0
        h = self.h(self.xhat, state)
        C = jacobian(self.h, self.xhat, state)
        y = np.array([measurement.accel_x, measurement.accel_y, measurement.accel_z])
        for i in range(0, 3):
            if np.abs(y[i]-h[i,0]) < threshold:
                Ci = C[i]
                L = self.P@Ci.T@np.linalg.inv(self.R_accel[i]+Ci@self.P@Ci.T)
                self.P = (np.identity(2)-L@Ci)@self.P@(np.identity(2)-L@Ci).T + L@self.R[i]@L.T
                self.xhat = self.xhat+L@(y[i]-h[i])

class ekf_position:
    # implement continous-discrete EKF to estimate pn, pe, chi, Vg
    def __init__(self):
        Q_tune = 1 #TODO need to tune this
        self.Q = Q_tune*np.identity(4)
        self.R = SENSOR.accel_sigma**2*np.identity(3) #is this the right sensor?
        self.N =  5 #TODO need to find the right value for this # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        wn0 = 0.0
        we0 = 0.0
        self.xhat = np.array([[MAV.pn0, MAV.pe0, MAV.Va0, MAV.psi0, wn0, we0, MAV.psi0]])
        self.P = np.identity(2)
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999


    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.pn = self.xhat.item(0)
        state.pe = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        Vg = x[2][0]
        chi = x[3][0]
        wn = x[4][0]
        we = x[5][0]
        psi = x[6][0]
        Va = state.Va
        r = state.r #is r psi dot?  The equation actually uses psi dot
        g = MAV.gravity
        phi = state.phi
        theta = state.theta
        q = state.q


        _f = np.array([[Vg*np.cos(chi), \
                        Vg*np.sin(chi), \
                        ((Va*np.cos(psi)+wn)*(-Va*r*np.sin(psi))+(Va*np.sin(psi)+we)*(Va*r*np.cos(psi)))/Vg, \
                        g/Vg*np.tan(phi)*np.cos(chi-psi), \
                        0.0, \
                        0.0, \
                        q*np.sin(phi)/np.cos(theta)+r*np.cos(phi)/np.cos(theta)]]).T
        return _f

    def h_gps(self, x, state):
        # measurement model for gps measurements
        #TODO I don't think I implemented the two h functions correctly
        pn = x[0][0]
        pe = x[1][0]
        Vg = x[2][0]
        chi = x[3][0]
        wn = x[4][0]
        we = x[5][0]
        psi = x[6][0]
        Va = state.Va

        _h = np.array([[pn,
                        pe,
                        Vg,
                        chi,
                        Va*np.cos(psi) + wn - Vg*np.cos(chi),
                        Va*np.sin(psi) + we - Vg*np.sin(chi)]]).T
        return _h

    def h_pseudo(self, x, state):
        # measurement model for wind triangle pseudo measurement
        pn = x[0][0]
        pe = x[1][0]
        Vg = x[2][0]
        chi = x[3][0]
        wn = x[4][0]
        we = x[5][0]
        psi = x[6][0]
        Va = state.Va

        _h = np.array([[pn,
                        pe,
                        Vg,
                        chi,
                        Va * np.cos(psi) + wn - Vg * np.cos(chi),
                        Va * np.sin(psi) + we - Vg * np.sin(chi)]]).T
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            Tp = self.Ts
            self.xhat = self.xhat + Tp*self.f(self.xhat, state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q @ G.T) #TODO need to implement this function as it has G
            # convert to discrete time models
            A_d = np.identity(2) + A*Tp + A@A*Tp**2
            # update P with discrete time model
            self.P = A_d@self.P@A_d.T + Tp**2@self.Q

    def measurement_update(self, state, measurement):
        # always update based on wind triangle pseudu measurement
        h = self.h_pseudo(self.xhat, state)
        C = jacobian(self.h_pseudo, self.xhat, state)
        y = np.array([0, 0])
        for i in range(0, 2):
            Ci = C[i]
            L = self.P@Ci.T@np.linalg.inv(self.R[i] + Ci@self.P@Ci.T)
            self.P = (np.identity(2)-L@Ci)@self.P@(np.identity(2)-L@Ci).T + L@self.R[i]@L.T
            self.xhat = self.xhat + L@(y[i]-h[i])

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, state)
            C = jacobian(self.h_gps, self.xhat, state)
            y = np.array([measurement.gps_n, measurement.gps_e, measurement.gps_Vg, measurement.gps_course])
            for i in range(0, 4):
                Ci = C[i]
                L = self.P @ Ci.T @ np.linalg.inv(self.R[i] + Ci @ self.P @ Ci.T)
                self.P = (np.identity(2) - L @ Ci) @ self.P @ (np.identity(2) - L @ Ci).T + L @ self.R[i] @ L.T
                self.xhat = self.xhat + L @ (y[i] - h[i])
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

def jacobian(fun, x, state):
    # compute jacobian of fun with respect to x
    f = fun(x, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.01  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J