import numpy as np
import sys
sys.path.append('..')
from chap11.dubins_parameters import dubins_parameters
from message_types.msg_path import msg_path

class path_manager:
    def __init__(self):
        # message sent to path follower
        self.path = msg_path()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        # flag that request new waypoints from path planner
        self.flag_need_new_waypoints = True
        self.halfspace_n = np.inf * np.ones((3,1))
        self.halfspace_r = np.inf * np.ones((3,1))
        # state of the manager state machine
        self.manager_state = 1
        # dubins path parameters
        self.dubins_path = dubins_parameters()
        self.new_waypoint_path = True #I added
        self.flag_path_changed = False #I added

    # def update(self, waypoints, radius, state): #TODO Why is this a function in a funtion?
    def update(self, waypoints, radius, state):
        # this flag is set for one time step to signal a redraw in the viewer
        if self.path.flag_path_changed == True:
            self.path.flag_path_changed = False
        if waypoints.num_waypoints == 0:
            waypoints.flag_manager_requests_waypoints = True
        else:
            if waypoints.type == 'straight_line':
                self.line_manager(waypoints, state)
            elif waypoints.type == 'fillet':
                self.fillet_manager(waypoints, radius, state)
            elif waypoints.type == 'dubins':
                self.dubins_manager(waypoints, radius, state)
            else:
                print('Error in Path Manager: Undefined waypoint type.')

        return self.path

    def line_manager(self, waypoints, state):

        p = np.array([state.pn, state.pe, -state.h])
        w = waypoints.ned.T
        N = waypoints.num_waypoints
        assert (N >= 3),"Less than 3 waypoints!"

        if self.new_waypoint_path:
            self.initialize_pointers()
        
        i_p = self.ptr_previous
        i = self.ptr_current
        i_n = self.ptr_next

        self.halfspace_r = w[i] #TODO point on half plane?  Dr. Beard said it is a point on the line before wi, but I feel like this only makes sense how I have it.
        qi_p = (w[i]-w[i_p])/np.linalg.norm(w[i]-w[i_p])
        qi = (w[i_n]-w[i])/np.linalg.norm(w[i_n]-w[i])
        self.halfspace_n = (qi_p+qi)/np.linalg.norm(qi_p+qi) #normal to half plane

        r = self.halfspace_r
        q = qi_p
        self.path.line_origin = np.array([r]).T
        self.path.line_direction = np.array([q]).T

        #Tell waypoint viewer to replot the path
        if self.flag_path_changed:
            self.path.flag_path_changed = True
            self.flag_path_changed = False
        if self.inHalfSpace(p):
            print('in half space')
            self.increment_pointers(waypoints.num_waypoints)
            self.flag_path_changed = True #Flag to indicate the path will change on next iteration and needs to be plotted.

    def fillet_manager(self, waypoints, radius, state):
        #get variables
        p = np.array([state.pn, state.pe, -state.h])
        w = waypoints.ned.T
        R = radius
        N = waypoints.num_waypoints
        i_p = self.ptr_previous
        i = self.ptr_current
        i_n = self.ptr_next
        assert (N >= 3),"Less than 3 waypoints!"

        #check if there is a new path
        if self.new_waypoint_path:
            self.initialize_pointers()
            self.manager_state = 1

        #calculate the variables used in both states
        qi_p = (w[i] - w[i_p]) / np.linalg.norm(w[i] - w[i_p])
        qi = (w[i_n] - w[i]) / np.linalg.norm(w[i_n] - w[i])
        ro_var = np.arccos(-qi_p.T@qi)

        #straight line path
        if self.manager_state == 1:
            #get variables
            self.path.flag = 'line'
            r = w[i]
            q = qi_p
            z = w[i] - R/np.tan(ro_var/2.0)*qi_p
            self.halfspace_r = z
            self.halfspace_n = qi_p

            #set the path for path follower
            self.path.line_origin = np.array([r]).T
            self.path.line_direction = np.array([q]).T

            #check if in half space &
            # Tell waypoint viewer to replot the path
            if self.flag_path_changed:
                self.path.flag_path_changed = True
                self.flag_path_changed = False
            if self.inHalfSpace(p):
                print('in half space')
                self.manager_state = 2
                self.flag_path_changed = True  #Flag to indicate the path will change on next iteration and needs to be plotted.

        #Orbit path
        elif self.manager_state == 2:
            #get parameters
            self.path.flag = 'orbit'
            c = w[i] - R/np.sin(ro_var/2.0)*(qi_p-qi)/np.linalg.norm(qi_p-qi)
            ro = R

            #calculate variables
            L = np.sign(qi_p[0]*qi[1]-qi_p[1]*qi[0])
            z = w[i] + R/np.tan(ro_var/2.0)*qi
            self.halfspace_r = z
            self.halfspace_n = qi

            #set path parameters for path follower
            self.path.orbit_center = np.array([c]).T
            self.path.orbit_direction = L
            self.path.orbit_radius = ro

            #check if in half space &
            # Tell waypoint viewer to replot the path
            if self.flag_path_changed:
                self.path.flag_path_changed = True
                self.flag_path_changed = False
            if self.inHalfSpace(p):
                print('in half space')
                self.increment_pointers(waypoints.num_waypoints)
                self.manager_state = 1
                self.flag_path_changed = True  #Flag to indicate the path will change on next iteration and needs to be plotted.

    # def dubins_manager(self, waypoints, radius, state):

    def initialize_pointers(self):
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2

        self.new_waypoint_path = False

    def increment_pointers(self, num_waypoints):
        self.ptr_previous = self.ptr_current
        self.ptr_current = self.ptr_next
        self.ptr_next = self.ptr_next + 1

        if self.ptr_next == num_waypoints:
            self.ptr_next = 0

    def inHalfSpace(self, pos):
        if (pos-self.halfspace_r).T @ self.halfspace_n >= 0:
            return True
        else:
            return False

#TODO do I actually need these functions?            
    # def contruct_line(self):

    # def construct_circle(self):
