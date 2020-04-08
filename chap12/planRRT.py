import numpy as np
import random
from message_types.msg_waypoints import msg_waypoints


class planRRT():
    def __init__(self, map):
        self.waypoints = msg_waypoints()
        self.segmentLength = 300 # standard length of path segments
        #keep in mind the size of the buildings.  Right now they are 360 wide
        self.min_distance = 100 # minimum distance from building center to path
        self.complete_path = []
        self.reqPaths = 3

    def planPath(self, wpp_start, wpp_end, map):

        # desired down position is down position of end node
        pd = wpp_end.item(2)

        # specify start and end nodes from wpp_start and wpp_end
        # format: N, E, D, cost, parentIndex, connectsToGoalFlag,
        start_node = np.array([wpp_start.item(0), wpp_start.item(1), pd, 0, 0, 0])
        end_node = start_node + np.array([500.0, 500.0, 0, 0, 0, 0])
        # end_node = np.array([wpp_end.item(0), wpp_end.item(1), pd, 0, 0, 0])
        # establish tree starting with the start node
        tree = np.array([start_node])
        numPaths = 0

        # check to see if start_node connects directly to end_node
        if ((np.linalg.norm(start_node[0:3] - end_node[0:3]) < self.segmentLength ) and self.feasiblePath(start_node, end_node, map)):
            self.waypoints.ned = end_node[0:3]
        else:
            while numPaths < self.reqPaths: #TODO only 3 complete paths required
                tree, flag = self.extendTree(tree, end_node, map, pd)
                numPaths = numPaths + flag


        # find path with minimum cost to end_node
        path = self.findMinimumPath(tree, end_node)
        return self.smoothPath(path, map)

    def extendTree(self, tree, end_node, map, pd): #Todo should map be a global variable?

        connectsToGoalFlag = 0
        D = self.segmentLength
        p = self.generateRandomNode(map, pd)
        v_star_index, v_star, v_star_cost = self.findClosestConfiguration(p, tree) #v_star includes just the position
        v_plus = v_star + D*(p-v_star)/np.linalg.norm(p-v_star) #step a distance D in direction of v_star to p
        v_plus[2][0] = pd
        if self.feasiblePath(v_star, v_plus, map):
            cost = v_star_cost + np.linalg.norm(v_plus-v_star) #cost is just set up to be the length of the entire path
            # format: N, E, D, cost, parentIndex, connectsToGoalFlag,
            new_node = np.array([[v_plus.item(0), v_plus.item(1), v_plus.item(2), cost, v_star_index, connectsToGoalFlag]])
            tree = np.concatenate((tree, new_node))
            if self.check_goal(end_node, D, v_plus, map):
                connectsToGoalFlag = 1
                cost = cost + np.linalg.norm(end_node[0:3]-v_plus) #cost is just set up to be the length of the entire path
                # format: N, E, D, cost, parentIndex, connectsToGoalFlag,
                new_node = np.array([[end_node.item(0), end_node.item(1), end_node.item(2), cost, len(tree)-1, connectsToGoalFlag]])
                tree = np.concatenate((tree, new_node))
                self.complete_path.append(new_node)
                print('completed path')

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
            parent_candidate_len.append(np.linalg.norm(p-np.array([parent_candidate]).T))

        parent_index = np.argmin(parent_candidate_len) #closest parent candidate to the new node
        parent = np.array([tree_positions[parent_index]]).T
        parent_cost = tree[parent_index][3]

        return parent_index, parent, parent_cost

    def check_goal(self, end_node, segmentLength, v_plus, map):

        pe = np.array([end_node[0:3]]).T
        if np.linalg.norm(pe - v_plus) <= segmentLength and self.feasiblePath(v_plus, end_node, map):
            return True
        else:
            return False

    def feasiblePath(self, start_node, end_node, map):
        #get parameters
        ps = start_node[0:3]
        pe = end_node[0:3]

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
                # if np.linalg.norm(building[0:2] - ps[0:2]) <= Del or np.linalg.norm(building[0:2] - pe[0:2]) <= Del:
                if np.linalg.norm(building[0:2] - pe[0:2]) <= len and pe[2]-building[2] <= len:
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
                if np.linalg.norm(building[0:2] - point[0:2]) <= Del and point[2]-building[2] <= Del:
                    return True
            point = point + Del*dir
        return False


    # def downAtNE(self, map, n, e): #TODO not sure what this is

    def findMinimumPath(self, tree, end_node):

        if len(self.complete_path) == self.reqPaths:
            costs = np.squeeze(np.array(self.complete_path))[:,3]
            min_index = np.argmin(costs)

            #take the last node, check its parent index if it is 0
            #if not, get the tree node at that index and put it IN FRONT
            #repeat for that node
            #if the parent index is 0 end the while loop and add the first node to the front
            min_path = np.array(self.complete_path[min_index])
            while min_path.item(4) != 0:
                min_path = np.concatenate((np.array([tree[int(min_path.item(4))]]), min_path))
            min_path = np.concatenate((np.array([tree[0]]), min_path))
        else: #this is the case where the start and end nodes are withing a segment and are a feasible path
            min_path = np.concatenate((tree, np.array([end_node])))

        return min_path

    def smoothPath(self, path, map):

        waypoints = path[:,0:3]
        i = 0
        j = 1
        smooth_path = np.array([waypoints[i]])
        N = len(waypoints)

        while j < N:
            if not self.feasiblePath(waypoints[i], waypoints[j], map):
                j = j-1
                i = j
                smooth_path = np.concatenate((smooth_path, np.array([waypoints[i]])))
            j = j+1
        smooth_path = np.concatenate((smooth_path, np.array([waypoints[j-1]])))

        return smooth_path


