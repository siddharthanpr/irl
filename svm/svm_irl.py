import time
import numpy as np
import sys, os

sys.path.append('/home/siddharth/thesis/code/irl/aima-python')
from mdp import *
import random
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt
from gui import *


class GridIRL:
    def __init__(self, window_size=800, grid_size=128, cell_size=16, gamma=0.99):
        self.window_size = window_size
        self.grid_size = grid_size
        self.cell_size = cell_size
        pix_per_grid = (window_size / grid_size)
        self.grid, self.actual_w = self.build_random()

        self.mdp_gw = GridMDP(self.grid, terminals=[], gamma=gamma)  # GridMDP flips grid like a noob
        self.grid_gui = GuiBoard("SVM IRL", [window_size / self.grid_size * self.grid_size,
                                             window_size / self.grid_size * self.grid_size], pix_per_grid)

    def get_feature_expectation(self, start_states, policy, tol=1.0e-7, graphics=False):
        '''
        This uses self.mdp_gw. Be careful when changing grid world or transfering
        '''
        expected_f = np.array([0.0] * (self.grid_size / self.cell_size) ** 2)

        for start_state in start_states:

            discount = 1
            current_state = start_state

            while discount > tol:
                expected_f[self.macro_cell(self.grid_size - 1 - current_state[1], current_state[
                    0])] += discount * 1  # 1 because of indicator feature, We subtract from x-dim because the grid is flipped in y and x,y are swapped. SO we must subtract y then swap which results in this x-dim
                discount *= self.mdp_gw.gamma
                current_state = self.mdp_gw.go(current_state, policy[current_state])
                self.grid_gui.render_move(current_state, graphics)

        return (1 - self.mdp_gw.gamma) * expected_f / float(
            len(start_states))  # assuming start_states asre equally probably

    def sparse_to_list(self, d, size, default=0.0):
        """ Convert dictionaty d from states to rewards to 2d list"""
        l = []
        for i in size[0]:
            l.append([default] * size[1])
        for (x, y) in d:
            l[x][y] = d[(x, y)]
        return l

    def macro_cell(self, x, y):
        size = self.grid_size
        cells = self.cell_size
        cells_per_row = (size / cells)
        return x / cells + y / cells * cells_per_row

    def grid_from_w(self, w):
        '''
        get grid world with rewards given the weigts to indicator features
        '''
        l = []
        for i in xrange(self.grid_size):
            l.append([0.0] * self.grid_size)
        for i in xrange(self.grid_size):
            for j in xrange(self.grid_size):
                l[i][j] = w[self.macro_cell(i, j)]
        return l

    def build_random(self, size=128, cells=16):
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

        for i in xrange(len(w)):
            w[i] = w[i] / float(w_sum)

        return self.grid_from_w(w), np.array(w)

    def sample_start_state(self, no=1):
        sampled_states = []
        for i in xrange(no):
            while True:
                sampled_state = (random.randint(0, 127), random.randint(0, 127))
                if self.mdp_gw.R(sampled_state) <= 0:
                    sampled_states.append(sampled_state)
                    break
        return sampled_states

    def get_lambda(self, mu_i, mu_E):

        P = np.append(np.eye(len(mu_E)), np.zeros((len(mu_i), len(mu_E))), axis=0)
        P = matrix(np.append(P, np.zeros((len(mu_E) + len(mu_i), len(mu_i))), axis=1).astype(np.double), tc='d')
        q = matrix(-np.append(1 * mu_E, [0.0] * len(mu_i)).astype(np.double), tc='d')
        G = matrix(np.append(np.zeros((len(mu_i), len(mu_E))), -np.eye(len(mu_i)), axis=1).astype(np.double), tc='d')
        h = matrix(np.array([0.0] * len(mu_i)).astype(np.double), tc='d')

        A = -np.eye(len(mu_E), dtype=float)
        MU_I = []
        ordered_keys = []
        for i in mu_i:
            ordered_keys.append(i)
            if len(MU_I) == 0:
                MU_I = np.reshape(mu_i[i], (len(mu_i[i]), 1))
            else:
                MU_I = np.append(MU_I, np.reshape(mu_i[i], (len(mu_i[i]), 1)), axis=1)
            A = np.append(A, np.reshape(mu_i[i], (len(mu_i[i]), 1)), axis=1)

        A = matrix(
            np.append(A, np.reshape([0.0] * len(mu_E) + [1] * len(mu_i), (1, len(mu_E) + len(mu_i))), axis=0).astype(
                np.double), tc='d')
        b = matrix(np.array([0.0] * len(mu_E) + [1]).astype(np.double), tc='d')
        sol = solvers.qp(P, q, G, h, A, b)
        # sol=solvers.qp(P, q, A=A,b=b)
        # print 'solution is'
        # print(sol['x']) 
        l = np.reshape(sol['x'][-len(mu_i):], (len(mu_i), 1))
        mu = np.dot(MU_I, l)
        mu_cvx = np.reshape(sol['x'][:-len(mu_i)], (len(mu_E)))

        print 'distance to cvx solution', np.linalg.norm(np.reshape(mu_cvx, len(mu_E)) - mu_E)

        return l, ordered_keys

    def show_learned(self, l, ordered_keys, start_states, pi_star, pi, length=100):

        p = np.reshape(l, len(l))

        for index, start_state in enumerate(start_states):
            pygame.display.set_caption("Demonstration %d" % index)

            current_state = start_state
            for i in xrange(length):
                current_state = self.mdp_gw.go(current_state, pi_star[current_state])
                self.grid_gui.render_move(current_state)

            current_state = start_state
            pygame.display.set_caption("Learned behavior %d" % index)

            for i in xrange(length):
                sampled_index = np.random.choice(len(ordered_keys), 1, p=p)[0]
                sampled_policy = pi[ordered_keys[sampled_index]]
                current_state = self.mdp_gw.go(current_state, sampled_policy[current_state])
                self.grid_gui.render_move(current_state)

    def irl(self, auto=True):
        # get_lambda({1:np.array([-1,2]),2:np.array([1,1]),3:np.array([0,0])}, np.array([0,2]))
        start_states = self.sample_start_state(10)
        self.grid_gui.render_mdp(self.mdp_gw)

        if auto:
            '''
            We assume we know the reward function and solve optimally the MDP and get pi_star
            '''
            V = value_iteration(self.mdp_gw)
            pi_star = best_policy(self.mdp_gw, V)
            self.grid_gui.V = V
            print "Optimal policy computed for the %d sampled states" % len(start_states)

        else:
            start_state, pi_star = self.grid_gui.record_demonstration()
            start_states = [start_state]

        self.grid_gui.pi = pi_star

        mu_E = self.get_feature_expectation(start_states, pi_star)

        pi = {0: self.mdp_gw.get_random_policy()}
        mu = {0: self.get_feature_expectation(start_states, pi[0])}
        mu_bar = mu[0]
        lambdas = {0: 1}
        w = {'E': self.actual_w}
        t = float('inf')
        eps = 0.005
        i = 1
        ts = []
        pV = None
        while t > eps:

            if i > 1:
                unit_mu_mu_bar = (mu[i - 1] - mu_bar) / np.linalg.norm(mu[i - 1] - mu_bar)
                theta = np.dot(mu[i - 1] - mu_E, unit_mu_mu_bar) / np.linalg.norm(mu[i - 1] - mu_bar)
                pm = mu_bar  # remove

                mu_bar = theta * mu_bar + (1 - theta) * mu[i - 1]
                lambdas[i - 1] = 1 - theta

                for j in xrange(i - 1):
                    lambdas[j] *= theta
            w[i] = mu_E - mu_bar
            t = np.linalg.norm(w[i])
            print 't', t, 'eps', eps

            if t < eps: break

            w[i] = w[i] / np.sum(abs(w[i]))
            mdp_estimate = GridMDP(self.grid_from_w(w[i]), terminals=[], gamma=self.mdp_gw.gamma)
            pV = value_iteration(mdp_estimate, seedU=pV)
            pi[i] = best_policy(mdp_estimate, pV)
            mu[i] = self.get_feature_expectation(start_states, pi[i])
            i += 1
            ts.append(t)

        fig = plt.figure()
        fig.suptitle('Deviation in feature expectation, ||mu_E - mu_bar||', fontsize=20)
        plt.plot(ts)
        plt.xlabel('Iterations')
        plt.ylabel('t')
        plt.show()

        print 'distance to mu_bar', np.linalg.norm(mu_E - mu_bar)
        l, ordered_keys = self.get_lambda(mu, mu_E)
        return [l, ordered_keys, start_states, pi_star, pi]


g = GridIRL(gamma=0.9)
[l, ordered_keys, start_states, pi_star, pi] = g.irl(1)
g.show_learned(l, ordered_keys, start_states, pi_star, pi)
g.grid_gui.hold_gui()
