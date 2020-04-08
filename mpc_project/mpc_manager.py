import numpy as np
import sys
sys.path.append('..')
from chap11.dubins_parameters import dubins_parameters
from message_types.msg_path import msg_path
import tools.wrap

from mpc_project.optimizer import optimizer
from mpc_project.target_manager import target_manager

class mpc_manager():
    def __init__(self):
        # message sent to path follower
        self.path = msg_path()
        self.optimize = optimizer()
        self.target = target_manager()

    def update(self, state_estimates):

        #restrict flight to a 2D horizontal plane
        xhat = np.array([[state_estimates.pn, state_estimates.pe, -state_estimates.h]]).T
        target_hat = self.target.estimate()

        u = self.optimize.update(xhat, target_hat)
        direction = np.array([[u.item(0)-xhat.item(0),u.item(1)-xhat.item(1),0.0]]).T
        direction = direction/np.linalg.norm(direction)

        self.path.flag = 'line'
        self.path.line_origin = u
        self.path.line_direction = direction
        self.path.flag_path_changed

        return self.path