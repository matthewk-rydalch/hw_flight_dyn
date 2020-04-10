import numpy as np

from parameters.aerosonde_parameters import pd0

class dynamic_model():
    def __init__(self, Ts):
        self.Ts = Ts
        self.pd = pd0
        self.xt = 0.0
        self.yt = 0.0
        self.tht = 0.0

    def update(self, xhat, u, N):

        predicted_waypoints = np.zeros((N,3))
        self.xt = xhat.item(0)
        self.yt = xhat.item(1)
        self.tht = xhat.item(3)
        Vg = xhat.item(4)
        for i in range(N):
            self.dubins_car(Vg, u[i])
            predicted_waypoints[i,:] = np.array([self.xt, self.yt, self.pd])

        return predicted_waypoints

    def dubins_car(self, Vg, u=0):

        self.tht = self.tht + u * self.Ts
        self.xt = self.xt + Vg * np.cos(self.tht) * self.Ts
        self.yt = self.yt + Vg * np.sin(self.tht) * self.Ts