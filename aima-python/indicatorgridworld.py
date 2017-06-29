from mdp import *
import numpy as np


class IndicatorWorld(GridMDP):
    def __init__(self, terminals, grid_size,cell_size , gamma=.9, stochastic = False):
        self.grid_size = grid_size
        self.cell_size = cell_size

        grid, self.true_weight = self.build_random(self.grid_size, self.cell_size )
        GridMDP.__init__(self, grid, terminals, init=(0, 0), gamma=gamma, stochastic = stochastic)

    def get_feature_vector(self, s):
        x = s[0]
        y = s[1]
        f = np.array([0.0] * (self.grid_size / self.cell_size) ** 2)
        f[self.macro_cell(self.grid_size - 1 - y, x)] = 1
        return f

    def grid_from_w(self, w):
        '''
        get grid world with rewards given the weigts to indicator features
        '''
        l = [[0.0] * self.grid_size for i in xrange(self.grid_size)]

        for i in xrange(self.grid_size):
            for j in xrange(self.grid_size):
                l[i][j] = w[self.macro_cell(i, j)]
        return l

    def macro_cell(self, x, y):
        size = self.grid_size
        cells = self.cell_size
        cells_per_row = (size / cells)
        return x / cells + y / cells * cells_per_row

    def w_from_grid(self, l):
        w = np.array([0.0]*self.grid_size**2)
        for i in xrange(self.grid_size):
            for j in xrange(self.grid_size):
                w[self.macro_cell(i, j)] = l[i][j]
        return w

    def build_random(self, size, cells):
        '''
        size is the size of the nxn grid (size = n)
        cells is the number of grid points in one macro cell region. - should divide size exactly
        '''
        w = []
        w_sum = 0
        for i in xrange((size / cells) ** 2):
            # if i < 32:
            #     w[i] = -1

            if random.uniform(0, 1) < 0.3:  # with probability 0.1
                if random.uniform(0, 1) < 0.6:  # with probability 0.8
                    w.append(-1)
                else:
                    w.append(random.uniform(0, 1))
            else:
                w.append(0)
            w_sum += abs(w[-1])

        if len(w) > 50:
            for i in xrange(len(w)):
                w[i] = w[i] / float(w_sum)

        elif len(w) > 20:
            w = [0] * (size / cells) ** 2
            w[-1] = 1
        else:
            w[0] = 10
            w[1] = 10
            w[2] = -10
            w[3] = 0

        # if len(w) > 4:
        #     print kk

        return self.grid_from_w(w), np.array(w)