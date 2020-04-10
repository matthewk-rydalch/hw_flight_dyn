import numpy as np
import sys
sys.path.append('..')
from chap11.dubins_parameters import dubins_parameters
from message_types.msg_path import msg_path
import tools.wrap

from mpc_project.optimizer import optimizer

class mpc_manager():
    def __init__(self, Ts, time_horizon):
        # message sent to path follower
        self.Ts = Ts
        self.path = msg_path()
        self.optimize = optimizer(Ts, time_horizon)

    def update(self, state_estimates, target):

        #restrict flight to a 2D horizontal plane
        xhat = np.array([[state_estimates.pn, state_estimates.pe, -state_estimates.h, state_estimates.chi, state_estimates.Vg]]).T
        target_hat = target.estimate()

        #TODO change variable name for u
        waypoint = self.optimize.update(xhat, target_hat)
        scale = 100 #this is to make the direction more accurate
        vec = scale*np.array([[waypoint.item(0)-xhat.item(0),waypoint.item(1)-xhat.item(1),0.0]]).T
        len = np.linalg.norm(vec)
        if len < 0.001:
            len = np.linalg.norm([xhat.item(0),xhat.item(1),0.0])
            if len < 0.001:
                direction = np.array([[1.0, 0.0, 0.0]]).T
                print('cant compute direction')
            else:
                direction = np.array([[xhat.item(0), xhat.item(1), 0.0]]).T/len
                print('cant compute direction')
        else:
            direction = vec/len
            # print('direction = ', direction)

        # direction = np.array([[1.0,0.0,0.0]]).T #TODO for debugging
        # vec = np.array([[target_hat.item(0) - xhat.item(0), target_hat.item(1) - xhat.item(1), 0.0]]).T #TODO for debugging
        # direction = vec/np.linalg.norm(vec)

        self.path.flag = 'line'
        self.path.line_origin = waypoint
        self.path.line_direction = direction
        self.path.flag_path_changed

        return self.path