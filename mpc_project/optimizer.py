import numpy as np
from scipy.optimize import minimize

from parameters.aerosonde_parameters import pd0

class optimizer():
    def __init__(self, Ts, time_horizon):
        self.Ts = Ts
        self.N = time_horizon  # in units of Ts
        self.u = np.zeros(time_horizon)
        self.u_bound = np.pi/14.0 #from experimental test
        self.r = 0.5 #gain for optimization
        self.D = np.zeros((self.N,self.N))
        for i in range(self.N-1):
            self.D[i][i] = 1
            self.D[i][i+1] = -1
        self.pd = pd0*np.ones(self.N)

    def update(self, xhat, target_hat):

        us = self.optimize_horizon(xhat, target_hat)

        u = us.item(0)

        return self.waypoints, u

    def optimize_horizon(self, xhat, target_hat):

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
        bds = np.ones((self.N,1))@np.array([[-self.u_bound, self.u_bound]])

        res = minimize(self.mpc_objective, self.u, method='SLSQP', args=(xm, Vg, target_Vg, self.r),
                       bounds=bds, options={'ftol': 1e-10, 'disp': False})

        # state and input and return
        optimized_inputs = res.x

        #set up the next time steps intial guess
        for i in range(self.N-1):
            self.u[i] = res.x[i+1]
        self.u[self.N-1] = res.x[self.N-1]

        return optimized_inputs

    # objective function to be minimized
    def mpc_objective(self, u, x, Vg, target_Vg, r):
        # define initial state and input
        xt = np.zeros(self.N)
        yt = np.zeros(self.N)
        tht = np.zeros(self.N)
        target_xt = np.zeros(self.N)
        target_yt = np.zeros(self.N)
        target_tht = np.zeros(self.N)
        error2 = 0.0

        tht[0] = x.item(2) + u.item(0) * self.Ts
        xt[0] = x.item(0) + Vg * np.cos(tht.item(0)) * self.Ts
        yt[0] = x.item(1) + Vg * np.sin(tht.item(0)) * self.Ts

        target_tht[0] = x.item(5) #no input on target
        target_xt[0] = x.item(3) + Vg * np.cos(target_tht.item(0)) * self.Ts
        target_yt[0] = x.item(4) + Vg * np.sin(target_tht.item(0)) * self.Ts

        for n in range(1,self.N,1):
            tht[n] = tht.item(n-1) + u.item(n) * self.Ts
            xt[n] = xt.item(n-1) + Vg * np.cos(tht.item(n)) * self.Ts
            yt[n] = yt.item(n-1) + Vg * np.sin(tht.item(n)) * self.Ts

            target_tht[n] = target_tht.item(n-1)
            target_xt[n] = target_xt.item(n-1) + target_Vg * np.cos(target_tht.item(n)) * self.Ts
            target_yt[n] = target_yt.item(n-1) + target_Vg * np.sin(target_tht.item(n)) * self.Ts

            #this is Y.T@Y
            error2 = error2 + (xt.item(n)-target_xt.item(n))**2+(yt.item(n)-target_yt.item(n))**2

        U = np.array([u]).T

        J = error2 + r * U.T @ self.D.T @ self.D @ U

        self.waypoints = np.array([xt, yt, self.pd]).T #save off the waypoints for plotting

        return J