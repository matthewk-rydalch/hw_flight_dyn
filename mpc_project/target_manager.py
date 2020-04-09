import numpy as np
from message_types.msg_state import msg_state

class target_manager():
    def __init__(self, Ts, posVel):
        self.Ts = Ts
        self.posVel = posVel
        #for plotting
        self.state = msg_state()
        self.state.pn = posVel.item(0)
        self.state.pe = posVel.item(1)
        self.state.h = -posVel.item(2)
        self.state.phi = 0.0
        self.state.theta = 0.0
        self.state.psi = 0.0

    def update(self):
        #TODO it looks like the velocity of the platform is not tied to the velocity line plotted
        self.dubins_car()

    def estimate(self):

        #TODO add noise?
        target_hat = self.posVel

        return target_hat

    def dubins_car(self, u = 0.0):
        x = self.posVel.item(0)
        y = self.posVel.item(1)
        chi = self.posVel.item(3)
        Vg = self.posVel.item(4)
        chi = chi + u * self.Ts
        x = x + Vg * np.cos(chi) * self.Ts
        y = y + Vg * np.sin(chi) * self.Ts

        self.posVel[0][0] = x
        self.posVel[1][0] = y
        self.posVel[3][0] = chi
        self.state.pn = x
        self.state.pe = y
