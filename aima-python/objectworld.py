# from pip._vendor.requests.utils import should_bypass_proxies

from mdp import *
import numpy as np


class ObjWorld(GridMDP):
    def __init__(self, grid_size, terminals, init=(0, 0), gamma=.9, colors=['b', 'r', 'g', 'c', 'o'], stochastic = False):
        self.grid_size = grid_size
        self.colors = colors
        self.init_objects()
        self.obj1_threshold = 3
        self.obj2_threshold = 2
        self.grid = [[0.0] * grid_size for i in xrange(grid_size)]

        for i in xrange(grid_size): # effiecient if there are large number of objects
            for j in xrange(grid_size):
                obj1_dist = obj2_dist = float('inf')
                world_pose = (j, grid_size-1-i)
                for o in self.objects1:
                    obj1_dist = min(obj1_dist, euclidean_distance(o['pose'], world_pose))
                    if obj1_dist <= self.obj1_threshold:
                        for o in self.objects2:
                            obj2_dist = min(obj2_dist, euclidean_distance(o['pose'], world_pose))
                            if obj2_dist <= self.obj2_threshold:
                                self.grid[i][j] = 1
                                break
                        if self.grid[i][j] == 0: self.grid[i][j] = -1
                        break

        GridMDP.__init__(self, self.grid, terminals, init=(0, 0), gamma=gamma, stochastic = stochastic)

    def init_objects(self):
        grid_size = self.grid_size
        self.color_size = len(self.colors)  # C in paper
        self.object_size = grid_size ** 2 / 10
        self.color_index = {self.colors[x]: x for x in xrange(len(self.colors))}
        self.objects = []
        object_poses = set([])

        self.objects1 = []
        self.objects2 = []

        if grid_size == 5:
            obj = {'inner': 'c', 'outer': 'b', 'pose': (0, 2)}
            self.objects.append(obj)
            self.objects1.append(obj)
            obj = {'inner': 'g', 'outer': 'r', 'pose': (2, 1)}
            self.objects.append(obj)
            self.objects2.append(obj)
            return

        for i in xrange(self.object_size):
            obj = {}
            obj['inner'] = random.choice(self.colors)
            obj['outer'] = random.choice(self.colors)
            while True:
                sampled_pose = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
                if sampled_pose not in object_poses: break
            object_poses |= set((sampled_pose,))
            obj['pose'] = sampled_pose
            if obj['outer'] == self.colors[0]: self.objects1.append(obj)
            if obj['outer'] == self.colors[1]: self.objects2.append(obj)

            self.objects.append(obj)

    def get_feature_vector(self, state, binary= False):
        feature = np.array([0.0] * 2 * self.color_size)
        for o in self.objects:# Need to make this more effiecient by holding nearest object indices for 32 example and larger
            dist = euclidean_distance(state, o['pose'])
            if feature[self.color_index[o['inner']]] == 0 or feature[self.color_index[o['inner']]] > dist:
                feature[self.color_index[o['inner']]] =  dist
            if feature[self.color_size + self.color_index[o['outer']]] == 0 or feature[self.color_size + self.color_index[o['outer']]] > dist:
                feature[self.color_size + self.color_index[o['outer']]] = dist
        if not binary: return feature

        bin_feature = np.array([0] * 2 * self.color_size * self.grid_size)

        index = 0

        for d in feature:
            for i in xrange(1,self.grid_size+1):
                bin_feature[index] = 1*(d >= i)
                index += 1
        return bin_feature

    def grid_from_w(self, w):
        l = [[0.0] * self.grid_size for i in xrange(self.grid_size)]
        length = self.grid_size
        for i in xrange(self.grid_size):
            for j in xrange(self.grid_size):
                l[i][j] = np.dot(w, self.get_feature_vector((j, length - 1 - i)))
        return l
