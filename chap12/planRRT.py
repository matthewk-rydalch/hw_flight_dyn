import numpy as np
import random
from message_types.msg_waypoints import msg_waypoints


class planRRT():
    def __init__(self, map):
        self.waypoints = msg_waypoints()
        self.segmentLength = 300 # standard length of path segments
        #keep in mind the size of the buildings.  Right now they are 360 wide
        self.min_distance = 250 # minimum distance from building center to path
        self.complete_path = np.array([[]])

    def planPath(self, wpp_start, wpp_end, map):

        # desired down position is down position of end node
        pd = wpp_end.item(2)

        # specify start and end nodes from wpp_start and wpp_end
        # format: N, E, D, cost, parentIndex, connectsToGoalFlag,
        start_node = np.array([wpp_start.item(0), wpp_start.item(1), pd, 0, 0, 0])
        end_node = np.array([wpp_end.item(0), wpp_end.item(1), pd, 0, 0, 0])

        # establish tree starting with the start node
        tree = np.array([start_node])

        # check to see if start_node connects directly to end_node
        if ((np.linalg.norm(start_node[0:3] - end_node[0:3]) < self.segmentLength ) and not self.feasiblePath(start_node, end_node, map)):
            self.waypoints.ned = end_node[0:3]
        else:
            numPaths = 0
            while numPaths < 3: #TODO only 3 complete paths required
                tree, flag = self.extendTree(tree, end_node, self.segmentLength, map, pd)
                numPaths = numPaths + flag


        # find path with minimum cost to end_node
        path = self.findMinimumPath(tree, end_node)
        return self.smoothPath(path, map)

    def extendTree(self, tree, end_node, segmentLength, map, pd, chi): #Todo should map be a global variable?

        connectsToGoalFlag = False
        D = segmentLength
        p = self.generateRandomNode(map, pd)
        v_star_index, v_star, v_star_cost = self.findClosestConfiguration(p, tree): #v_star includes just the position
        v_plus = D*(v_star-p)/np.linalg.norm(v_star-p) #step a distance D in direction of v_star to p
        if self.feasiblePath(p, v_plus, map):
            cost = v_star_cost + np.linalg.norm(v_plus-v_star) #cost is just set up to be the length of the entire path
            # format: N, E, D, cost, parentIndex, connectsToGoalFlag,
            new_node = np.array([[v_plus[0], v_plus[1], v_plus[2], cost, v_star_index, connectsToGoalFlag]])
            tree = np.concatenate((tree, new_node))
            if self.check_goal(end_node, segmentLength, v_plus, map):
                connectsToGoalFlag = True
                cost = cost + np.linalg.norm(end_node[0:3]-v_plus) #cost is just set up to be the length of the entire path
                # format: N, E, D, cost, parentIndex, connectsToGoalFlag,
                new_node = np.array([[end_node[0], end_node[1], end_node[2], cost, len(tree), connectsToGoalFlag]])
                tree = np.concatenate((tree, new_node))
                self.complete_path = np.concatenate((self.complete_path, new_node))

        return tree, connectsToGoalFlag

    def generateRandomNode(self, map, pd): #TODO add chi

        pn = random.uniform(0.0, map.city_width)
        pe = random.uniform(0.0, map.city_width) #width and length are the same

        p = np.array([[pn, pe, pd]]).T

        return p

    def findClosestConfiguration(self, p, tree): #tree variable only has position in this scope

        tree_positions = tree[:,0:3] #todo may need to exclude the end point and others within reach of the end point
        parent_candidate_len = [] #length from the parent candidate to the new node
        for parent_candidate in tree_positions:
            parent_candidate_len.append(np.linalg.norm(p-parent_candidate))

        parent_index = np.argmin(parent_candidate_len) #closest parent candidate to the new node
        parent = tree_positions[parent_index]
        parent_cost = tree[parent_index][3]

        return parent_index, parent, parent_cost

    def check_goal(self, end_node, segmentLength, v_plus, map):

        pe = end_node[0:3]
        if np.linalg.norm(pe - v_plus) <= segmentLength and self.feasiblePath(v_plus, end_node, map)
            return True
        else:
            return False

    def feasiblePath(self, start_node, end_node, map):
        #get parameters
        ps = np.array([start_node[0:3]]).T
        pe = np.array([end_node[0:3]]).T

        collision = self.collision(ps, pe, map)
        # # fliable = self.fliable(): #TODO add this function
        #
        # if fliable and not collision:
        #     return True
        # return False
        if not collision:
            return True
        return False


    def collision(self, ps, pe, map):

        #get parameters
        Del = self.min_distance #min distance between line and building (diameter from each segment point of the path)
        vec = pe - ps
        len = np.linalg.norm(vec)

        close_buildings = []
        for i in range(map.num_city_blocks):
            for j in range(map.num_city_blocks):
                building = np.array([[map.building_north[i], map.building_east[j], map.building_height[i][j]]]).T
                if np.linalg.norm(building - ps) <= len/2.0 or np.linalg.norm(building - pe) <= len/2.0:
                    close_buildings.append(building)

        # get points along path and check them
        collision = self.pointsAlongPath(ps, pe, Del, vec, len, close_buildings)

        return collision

    def pointsAlongPath(self, ps, pe, Del, vec, len, close_buildings):

        dir = vec/len

        #initialize first point
        i = 0
        point = ps + Del*dir
        while np.linalg.norm(point - ps) <= len:
            for building in close_buildings:
                if np.linalg.norm(building - point) <= Del:
                    return True

            point = point + Del*dir

        return False


    # def downAtNE(self, map, n, e): #TODO not sure what this is

    def findMinimumPath(self, tree, end_node):

        costs = self.complete_path[:,3]
        min_index = np.argmin(costs)

        #take the last node, check its parent index if it is 0
        #if not, get the tree node at that index and put it IN FRONT
        #repeat for that node
        #if the parent index is 0 end the while loop and add the first node to the front
        min_path = np.array([self.complete_path[min_index]])
        while min_path[0][4] ~= 0:
            min_path = np.concatenate((tree[min_path[0][4]], min_path))
        min_path = np.concatenate((tree[0], min_path))

        return min_path

    def smoothPath(self, path, map):

        waypoints = path[:,0:3]
        i = 0
        j = 1
        smooth_path = np.array([[waypoints[i]]])
        N = len(waypoints)

        while j < N:
            if not self.feasiblePath(waypoints[i], waypoints[j], map)
                j = j-1
                i = j
                smooth_path = np.concatenate((smooth_path, waypoints[i]))
            j = j+1
        smooth_path = np.concatenate((smooth_path, waypoints[j-1]))

        return smooth_path


