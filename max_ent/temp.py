import numpy as np
from collections import defaultdict
import random
import sys

sys.path.append('/home/siddharth/thesis/code/irl/aima-python')
sys.path.append('/home/siddharth/thesis/code/irl/gui')
from mdp import *
from gui import *
from matplotlib import pyplot as plt


class MaxEntIRL(object):
    def __init__(self, window_size=800, grid_size=5, cell_size=1, gamma=0.9):

        self.window_size = window_size
        self.grid_size = grid_size
        self.cell_size = cell_size
        pix_per_grid = (window_size / grid_size)
        self.grid, self.actual_w = self.build_random(grid_size, cell_size)

        self.mdp_gw = GridMDP(self.grid, terminals=[], gamma=gamma)  # GridMDP flips grid like a noob
        self.grid_gui = GuiBoard("MaxEnt IRL", [window_size / self.grid_size * self.grid_size,
                                                window_size / self.grid_size * self.grid_size], pix_per_grid)
        self.grid_gui.render_mdp(self.mdp_gw)

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

    def get_maxent_policy(self, theta, N=200):

        '''
        Implementation of step 1,2 and 3 of Ziebart et. al. Maximum Entropy Inverse Reinforcement Learning
        '''

        Zs = defaultdict(lambda: 1, {})  # Initializing Z_s to 1
        Za = {}
        # [(1, self.go(state, action))]

        Zs_max_p = 1
        for i in xrange(N):
            self.grid_gui.render_move((0, 0), delay=0.0)

            Zs_copy = Zs.copy()
            Zs_max = -float('inf')

            for s in self.mdp_gw.states:

                Zs_copy[s] = 0

                exp_reward = np.exp(theta[self.macro_cell(self.grid_size - 1 - s[1], s[
                    0])])  # theta[.] is nothing but (theta^T).f, where f is the feature vector
                for a in self.mdp_gw.actions(s):
                    Za[(s, a)] = 0
                    for p, sk in self.mdp_gw.T(s, a):
                        Za[(s, a)] += 1. / Zs_max_p * Zs[sk] * p  # same as Za_{i,j} in the paper, Step 2 in algorithm
                    Za[(s, a)] *= exp_reward
                    Zs_copy[s] += Za[(s, a)]

                if Zs_max < Zs_copy[s]:
                    Zs_max = Zs_copy[s]
                    # print Zs[s], Zs_temp
            Zs = Zs_copy  # Step 2 second line in algorithm
            Zs_max_p = Zs_max

            # input()
            # print Za[((15,0),(0,1))], Za[((15,0),(0,-1))], Za[((15,0),(1,0))], Za[((15,0),(-1,0))]

        pi = defaultdict(lambda: [], {})

        for s in self.mdp_gw.states:
            pi[s] = [(Za[(s, a)] / float(Zs[s]), a) for a in self.mdp_gw.actions(s)]  # Step 3 in the algorithm
        self.grid_gui.pi = pi
        return pi

    def execute_policy(self, s, pi, length):
        current_state = s
        for i in xrange(length):
            self.grid_gui.render_move(current_state, delay=0.1)

            p = [prob[0] for prob in pi[current_state]]
            sampled_index = np.random.choice(len(p), 1, p=p)[0]
            sampled_action = pi[current_state][sampled_index][1]
            current_state = self.mdp_gw.go(current_state, sampled_action)

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

    def irl(self, n_start_states=10, n_repeat=3, w_range=[0, 1], eps=0.3, learning_rate=0.1):

        #### Get Demonstrations and feature expectations of demonstrations
        svf_epsilon = .001  # In order to use same convergence for demonstrations and svf calculations
        demo_length = 75  # 3 * self.grid_size
        print 'Started...'
        pi_star = get_stochastic_representation(policy_iteration(self.mdp_gw))
        print 'Computed optimal policy...'
        # pi_star = {}
        # actions = ((1, 0), (0, -1), (-1, 0), (0, -1))
        # for i in xrange(self.grid_size):
        #     for j in xrange(self.grid_size):
        #         pi_star[(i,j)] = [(0.5, (1, 0)),(0.5, (0, -1))]

        start_states = self.sample_start_states(n_start_states)
        print "Sampled start states"
        demonstrations = roll_out_policy(self.mdp_gw, pi_star, start_states, n_repeat,
                                         demo_length)  # change roll_out_policy to get stochastic policy
        print "Gathered demonstrations"
        D = self.get_trajectory_svf(demonstrations, eps=svf_epsilon)
        self.grid_gui.D = D

        f_ex = self.get_feature_expectation(D)
        print "Computed f_ex"

        #### Initialize random weight and gradient
        initial_distribution = {}
        for s in start_states:
            if s not in initial_distribution: initial_distribution[s] = 0
            initial_distribution[s] += 1. / len(start_states)
        print initial_distribution

        ########### Test if get_expected_svf and get_trajectory_svf have same behavior in case of deterministic MDP
        D = get_expected_svf(self.mdp_gw, pi_star, length = demo_length, initial_distribution = initial_distribution, eps = svf_epsilon)
        f_ex2 = self.get_feature_expectation(D)
        print np.linalg.norm(f_ex-f_ex2)
        print f_ex
        print f_ex2

        sys.exit()

        feature_size = (self.grid_size / self.cell_size) ** 2
        grad = np.array([float('inf')] * feature_size)
        w = np.random.uniform(w_range[0], w_range[1], size=(feature_size,))
        print "Starting Gradient Descent"
        epochs = 200
        while (epochs > 0 and max(abs(grad) > eps)):
            epochs -= 1
            mdp = GridMDP(self.grid_from_w(w), terminals=[], gamma=self.mdp_gw.gamma)
            pi, _, _ = soft_value_iteration(mdp)
            Ds = get_expected_svf(mdp, pi, length=demo_length, initial_distribution=initial_distribution,
                                  eps=svf_epsilon)
            f = self.get_feature_expectation(Ds)

            grad = f_ex - f

            w += learning_rate * grad
            print max(abs(grad)), np.mean(grad)
            # print w
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.pcolor(np.array(self.grid))
        plt.colorbar()
        plt.title("Groundtruth reward")
        plt.subplot(1, 2, 2)
        plt.pcolor(w.reshape((m.grid_size, m.grid_size)))
        plt.colorbar()
        plt.title("Recovered reward")
        plt.show()
        m.show_learned(start_states, pi_star, pi, length=demo_length)
        return w, pi

    def get_feature_expectation(self, D):

        feature_size = (self.grid_size / self.cell_size) ** 2
        f = np.array([0.0] * feature_size)

        for s in D:
            f[self.macro_cell(self.grid_size - 1 - s[1], s[0])] = D[s]
        return f

    def feature(x, y):
        f = np.array([0.0] * (self.grid_size / self.cell_size) ** 2)
        f[self.macro_cell(self.grid_size - 1 - y, x)] = 1
        return f

    def sample_start_states(self, no=1):
        sampled_states = []
        for i in xrange(no):
            while True:
                sampled_state = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
                if self.mdp_gw.R(sampled_state) <= 0:
                    sampled_states.append(sampled_state)
                    break
        return sampled_states

    def get_trajectory_svf(self, trajectories, eps=0.001):
        '''
        Get state visitation frequencies for a set of trajectories given indicator features
        '''
        D = {}
        for t in xrange(len(trajectories[0])):
            discount = self.mdp_gw.gamma ** t
            D_copy = D.copy()
            for traj in trajectories:
                s, _ = traj[t]
                if s not in D: D[s] = 0
                D[s] += discount * 1. / len(trajectories)
            delta = -1
            for s in D:
                if s not in D_copy: D_copy[s] = 0
                delta = max(delta, abs(D_copy[s] - D[s]))
            if delta >= 0 and delta < eps: return D
        return D

    def get_trajectory_FE(self, trajectories):
        '''
        Get feature expectation for a set of trajectories given indicator features
        '''
        expected_f = np.array([0.0] * (self.grid_size / self.cell_size) ** 2)

        for traj in trajectories:
            discount = 1
            for (s, _) in traj:
                expected_f[
                    self.macro_cell(self.grid_size - 1 - s[1], s[0])] += discount * 1  # Hardcoding 1 for vistation
                discount *= self.mdp_gw.gamma
        return expected_f / float(len(trajectories))

    def show_learned(self, start_states, pi_star, pi, length=100):

        for index, start_state in enumerate(start_states):
            pygame.display.set_caption("Demonstration %d / %d" % (index, len(start_states)))
            start_state
            self.execute_policy(start_state, pi_star, length)
            pygame.display.set_caption("Learned behavior %d / %d" % (index, len(start_states)))
            self.execute_policy(start_state, pi, length)


def roll_out_policy(mdp, pi, start_states, n=1, length=20):
    trajectories = []

    for s in start_states:
        for _ in xrange(n):
            traj = []
            current_state = s
            for _ in xrange(length):
                p = [x[0] for x in pi[current_state]]
                sampled_index = np.random.choice(len(p), 1, p=p)[0]
                traj.append((current_state, pi[current_state][sampled_index][1]))
                current_state = mdp.go(current_state, pi[current_state][sampled_index][1])
            trajectories.append(traj)

    return trajectories


def get_expected_svf(mdp, pi, initial_distribution, length=float('inf'), eps=0.001):
    '''
    get state visitation frequencies Algorithm 9.3 in Ziebart's thesis

    Note that this does not have same behavior as roll_out_policy in case of high values of eps (>1). THis is only because we do not compute delta for first iteration (going from 0 to initial svf)
    '''

    D = initial_distribution.copy()
    T, gamma = mdp.T, mdp.gamma

    Ds = initial_distribution.copy()

    mm_converged = False
    discount = 1
    while length > 1:
        length -= 1

        discount *= mdp.gamma

        if not mm_converged:  # If motion model not converged yet
            D_prime = {}
            delta = 0
            for sx in D:
                for pi_s_a, a in pi[sx]:
                    for P, sz in T(sx, a):
                        if sz not in D_prime: D_prime[sz] = 0
                        D_prime[sz] += D[sx] * pi_s_a * P

            for s in D:
                if s not in D_prime: D_prime[s] = 0
                delta = max(delta, abs(D[s] - D_prime[s]))
            D = D_prime

        if (delta < 0.001 * eps): mm_converged = True

        delta = 0
        for s in D:
            if s not in Ds: Ds[s] = 0
            Ds_t = discount * D[s]
            Ds[s] += Ds_t
            delta = max(delta, Ds_t)

        if (delta < eps):
            return Ds

    return Ds


def soft_value_iteration(mdp, epsilon=0.001, seedV=None):
    if seedV == None:
        V_soft = defaultdict(lambda: -float('inf'), {})
    else:
        V_soft = seedV.copy()

    R, T, gamma = mdp.R, mdp.T, mdp.gamma

    i = 0
    Q_soft = {}
    minimum_pi_prob = 1.0e-5
    while True:
        delta = 0
        V_soft_prime = {}
        for s in mdp.states:
            p_V_soft = V_soft[s]
            V_soft_prime[s] = R(s)
            for a in mdp.actions(s):
                V_soft_prime[s] = soft_max_2(V_soft_prime[s],
                                             R(s) + gamma * sum([p * V_soft[s1] for (p, s1) in T(s, a)]))
            delta = max(delta, abs(p_V_soft - V_soft_prime[s]))
        V_soft = V_soft_prime
        if delta < epsilon * (1. - gamma) / gamma:

            Q_soft = {}
            pi = {}
            for s in mdp.states:
                pi[s] = []
                nomalizing_factor = 0
                for a in mdp.actions(s):
                    Q_soft[(s, a)] = R(s) + gamma * sum([p * V_soft[s1] for (p, s1) in T(s, a)])

                    pi_a_s = np.exp(Q_soft[(s, a)] - V_soft[s])
                    if pi_a_s > minimum_pi_prob:
                        pi[s].append((pi_a_s, a))
                        nomalizing_factor += pi_a_s
                pi[s] = [(p / float(nomalizing_factor), a) for (p, a) in
                         pi[s]]  # due to log-exp instability, probability may not be normalized. So normalizing here

            return pi, Q_soft, V_soft


def soft_max(l):
    '''
    l - list of elements to compute softmax
    '''


def soft_max_2(x1, x2):
    '''
    x1, x2 - 2 elements to compute millowmax as in Ziebart's thesis
    '''
    max_x = max(x1, x2)
    min_x = min(x1, x2)

    return max_x + np.log(1 + np.exp(min_x - max_x))


def get_stochastic_representation(deterministic_policy):
    pi = {}
    for s in deterministic_policy:
        pi[s] = [(1, deterministic_policy[s])]
    return pi


if __name__ == "__main__":
    m = MaxEntIRL(window_size=800, grid_size=2, cell_size=1, gamma=0.99)
    w, pi = m.irl(n_start_states=20, n_repeat=10, w_range=[0, 1], eps=0, learning_rate=0.01)
    # m.execute_policy((0,0), pi)
    m.grid_gui.hold_gui()
