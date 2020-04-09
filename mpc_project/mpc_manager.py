import numpy as np
import sys
sys.path.append('..')
from chap11.dubins_parameters import dubins_parameters
from message_types.msg_path import msg_path
import tools.wrap

from mpc_project.optimizer import optimizer

class mpc_manager():
    def __init__(self, Ts):
        # message sent to path follower
        self.Ts = Ts
        self.path = msg_path()
        self.optimize = optimizer(Ts)

    def update(self, state_estimates, target):

        #restrict flight to a 2D horizontal plane
        xhat = np.array([[state_estimates.pn, state_estimates.pe, -state_estimates.h, state_estimates.chi, state_estimates.Vg]]).T
        target_hat = target.estimate()

        u = self.optimize.update(xhat, target_hat)
        vec = np.array([[u.item(0)-xhat.item(0),u.item(1)-xhat.item(1),0.0]]).T
        len = np.linalg.norm(vec)
        if len < 0.001:
            len = np.linalg.norm([xhat.item(0),xhat.item(1),0.0])
            if len < 0.001:
                direction = np.array([[1.0, 0.0, 0.0]]).T
            else:
                direction = np.array([[xhat.item(0), xhat.item(1), 0.0]]).T/len
        else:
            direction = vec/len

        self.path.flag = 'line'
        self.path.line_origin = u
        self.path.line_direction = direction
        self.path.flag_path_changed

        return self.path