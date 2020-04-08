import numpy as np

from mpc_project.dynamic_model import dynamic_model

class optimizer():
    def __init__(self):
        self.predict_dynamics = dynamic_model()

    def update(self, xhat, target_hat):

        ###should be part of the optimizer
        a = self.predict_dynamics.update(xhat, target_hat)
        ###

        u = np.array([[100.0, 0.0, -100.0]]).T + np.array([[xhat.item(0), 0.0, 0.0]]).T

        return u