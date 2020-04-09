import numpy as np

class target_manager():
    def __init__(self):
        self.Ts = 0.1 #TODO get the right start values for all of these
        self.x = 100
        self.y = 100
        self.d = -100
        self.chi = -np.pi/2.0
        self.Vg = 20

    def update(self):
        self.dubins_car()

    def estimate(self):

        target_hat = np.array([[self.x, self.y, self.d, self.chi, self.Vg]]).T

        return target_hat

    def dubins_car(self):

        self.chi = self.chi + self.u * self.Ts
        self.x = self.x + self.Vg * np.cos(self.chi) * self.Ts
        self.y = self.y + self.Vg * np.sin(self.chi) * self.Ts