import numpy as np
from message_types.msg_waypoints import msg_waypoints


class planRRT():
    def __init__(self, map):
        self.waypoints = msg_waypoints()
        self.segmentLength = 300 # standard length of path segments
        #keep in mind the size of the buildings.  Right now they are 360 wide
        self.min_distance = 250 # minimum distance from building center to path

    def planPath(self, wpp_start, wpp_end, map):



        # desired down position is down position of end node
        pd = wpp_end.item(2)

        # specify start and end nodes from wpp_start and wpp_end
        # format: N, E, D, cost, parentIndex, connectsToGoalFlag,
        start_node = np.array([wpp_start.item(0), wpp_start.item(1), pd, 0, 0, 0])
        end_node = np.array([wpp_end.item(0), wpp_end.item(1), pd, 0, 0, 0])

        # establish tree starting with the start node
        tree = start_node

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

    # def generateRandomNode(self, map, pd, chi):

    def feasiblePath(self, start_node, end_node, map):
        #get parameters
        ps = np.array([start_node[0:3]]).T
        pe = np.array([end_node[0:3]]).T

        collision = self.collision(ps, pe, map)
        # # fliable = self.fliable(): #TODO add this function
        #
        # if fliable and not collision:
        #     return true
        # return false
        return collision

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


    # def downAtNE(self, map, n, e):

    # def extendTree(self, tree, end_node, segmentLength, map, pd):

    # def findMinimumPath(self, tree, end_node):

    # def smoothPath(self, path, map):

