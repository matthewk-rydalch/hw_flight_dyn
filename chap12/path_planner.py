# path planner for mavsim_python
#
# mavsim_python
#     - Beard & McLain, PUP, 2012
#     - Last updated:
#         4/3/2019 - BGM
import numpy as np
import sys
sys.path.append('..')
from message_types.msg_waypoints import msg_waypoints
from chap12.planRRT import planRRT

class path_planner:
    def __init__(self):
        # waypoints definition
        self.waypoints = msg_waypoints()

    def update(self, map, state, planner_flag):

        self.rrt = planRRT(map)
        if planner_flag == 1:
            self.waypoints.type = 'fillet'
            self.waypoints.num_waypoints = 4
            Va = 25
            self.waypoints.ned \
                = np.array([[0, 0, -100],
                            [1000, 0, -100],
                            [0, 1000, -100],
                            [1000, 1000, -100]]).T
            self.waypoints.airspeed \
                = np.array([[Va, Va, Va, Va]])
        elif planner_flag == 2:
            self.waypoints.type = 'dubins'
            self.waypoints.num_waypoints = 4
            Va = 25
            self.waypoints.ned \
                = np.array([[0, 0, -100],
                            [1000, 0, -100],
                            [0, 1000, -100],
                            [1000, 1000, -100]]).T
            self.waypoints.airspeed[:, 0:self.waypoints.num_waypoints] \
                = np.array([[Va, Va, Va, Va]])
            self.waypoints.course \
                = np.array([[np.radians(0),
                             np.radians(45),
                             np.radians(45),
                             np.radians(-135)]]).T
        elif planner_flag == 3:
            self.waypoints.type = 'fillet'
            self.waypoints.num_waypoints = 0
            Va = 25
            # current configuration vector format: N, E, D, Va
            wpp_start = np.array([state.pn,
                                  state.pe,
                                  -state.h,
                                  state.Va])
            if np.linalg.norm(np.array([state.pn, state.pe, -state.h])-np.array([map.city_width, map.city_width, -state.h])) == 0:
                wpp_end = np.array([0,
                                    0,
                                    -state.h,
                                    Va])
            else:
                wpp_end = np.array([map.city_width,
                                    map.city_width,
                                    -state.h,
                                    Va])

            waypoints, course = self.rrt.planPath(wpp_start, wpp_end, map)
            self.waypoints.ned = waypoints.T
            self.waypoints.airspeed = Va
            self.waypoints.num_waypoints = len(waypoints)

        elif planner_flag == 4:
            self.waypoints.type = 'dubins'
            self.waypoints.num_waypoints = 0
            Va = 25
            # current configuration vector format: N, E, D, Va
            wpp_start = np.array([state.pn,
                                  state.pe,
                                  -state.h,
                                  state.Va])
            if np.linalg.norm(np.array([state.pn, state.pe, -state.h]) - np.array(
                    [map.city_width, map.city_width, -state.h])) == 0:
                wpp_end = np.array([0,
                                    0,
                                    -state.h,
                                    Va])
            else:
                wpp_end = np.array([map.city_width,
                                    map.city_width,
                                    -state.h,
                                    Va])

            waypoints, course = self.rrt.planPath(wpp_start, wpp_end, map)
            self.waypoints.ned = waypoints.T
            self.waypoints.course = course
            self.waypoints.airspeed = Va
            self.waypoints.num_waypoints = len(waypoints)
        else:
            print("Error in Path Planner: Undefined planner type.")

        return self.waypoints
