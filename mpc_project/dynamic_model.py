import numpy as np

from parameters.aerosonde_parameters import pd0

class dynamic_model():
    def __init__(self, Ts):
        self.Ts = Ts
        self.pd = pd0

    def update(self, xhat, u): #xhat, target_hat, u):

        #TODO I think the way this is set up, the vehicle will never turn in one step
        pose2d = self.dubins_car(xhat, u)
        predicted_waypoint = np.array([[pose2d.item(0), pose2d.item(1), self.pd]]).T

        return predicted_waypoint

    def dubins_car(self, hat, u=0):
        xm = hat.item(0)
        ym = hat.item(1)
        thm = hat.item(3) #chi
        Vg = hat.item(4)

        thp = thm + u * self.Ts
        xp = xm + Vg * np.cos(thp) * self.Ts
        yp = ym + Vg * np.sin(thp) * self.Ts

        pose2d = np.array([[xp, yp, thp]])

        return pose2d