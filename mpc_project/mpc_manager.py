import numpy as np
import sys
sys.path.append('..')
from chap11.dubins_parameters import dubins_parameters
from message_types.msg_path import msg_path
import tools.wrap

from mpc_project.dynamic_model import dynamic_model
from mpc_project.optimizer import optimizer
from mpc_project.target_manager import target_manager

class mpc_manager():
    def __init__(self):
        # message sent to path follower
        self.path = msg_path()
        self.predict_dynamics = dynamic_model()
        self.optimize = optimizer()
        self.target = target_manager()

    def update(self):
        self.path.flag = 'line'
        self.path.line_origin = np.array([[0.0, 0.0, -100.0]]).T
        self.path.line_direction = np.array([[0.5, 1.0, 0.0]]).T
        self.path.line_direction = self.path.line_direction / np.linalg.norm(self.path.line_direction)        


        return self.path