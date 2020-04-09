import numpy as np

class dynamic_model():
    def __init__(self):
        self.Ts = 0.1 #TODO get the right value here
        self.pd = -100 #TODO get this connected to the real down position

    def update(self, xhat, u): #xhat, target_hat, u):

        #TODO need xhat to include chi and Vg
        #TODO I think the way this is set up, the vehicle will never turn in one step
        pose2d = self.dubins_car(xhat, u)
        # target_predict = self.dubins_car(target_hat)

        # error_predict = x_predict - target_predict

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