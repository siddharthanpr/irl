import numpy as np
from collections import defaultdict
import random
import sys

sys.path.append('/home/siddharth/thesis/code/irl/aima-python')
sys.path.append('/home/siddharth/thesis/code/irl/max_ent')
sys.path.append('/home/siddharth/thesis/code/irl/gui')
from mdp import *
from objectworld import ObjWorld
from gui import *
from max_ent_irl import *

from matplotlib import pyplot as plt


from max_ent_irl import MaxEntIRL

from matplotlib import pyplot as plt


class Gpirl(MaxEntIRL):
    def __init__(self, window_size=800, grid_size=5, cell_size=1, gamma=0.9, world = 'obj', stochastic = False):
        MaxEntIRL.__init__(self, window_size, grid_size, cell_size, gamma=gamma, world=world, stochastic=stochastic)
        self.object_world = self.mdp_gw
        self.sigma_sq = 0.5e-2


    def k(self, xi, xj, name='RBF', **kwargs):

        assert len(xi) == len(xj)
        if name == 'RBF':
            s = 0
            for k in xrange(len(xi)):
                s += (xi[k] - xj[k]) * (xi[k] - xj[k])
            return kwargs['sigma_f'] ** 2 * np.exp(-1. / (2 * kwargs['l'] ** 2) * s) + kwargs['sigma_n'] ** 2 * (not kwargs[
                'i_neq_j'])

        if name == 'ARD':
            assert len(xi) == len(kwargs['lambd'])
            trace = sum(kwargs['lambd'])
            s = 0
            for k in xrange(len(xi)):
                s += kwargs['lambd'][k] * (xi[k] - xj[k]) * (xi[k] - xj[k]) + 2 * kwargs['i_neq_j'] * kwargs['sigma_sq'] * trace
                # print s
            return kwargs['beta'] * np.exp(-s / 2.)

    def compute_demonstration_savf(self, demonstrations):
        self.mu_cap_sa = {} # state action pair visitation frequency. Actually not required for computations
        self.v_s = {}

        for traj in demonstrations:
            for (s, a) in traj:
                if (s, a) not in self.mu_cap_sa: self.mu_cap_sa[(s, a)] = 0
                if s not in self.v_s: self.v_s[s] = 0
                self.mu_cap_sa[(s, a)] += 1
                self.v_s[s] += 1
                for p, sz in self.mdp_gw.T(s,a):
                    if sz not in self.v_s: self.v_s[sz] = 0
                    self.v_s[sz] -= self.mdp_gw.gamma * p

        self.v_s = defaultdict(lambda:0, self.v_s)

    def K(self, x1, x2, **kwargs):
        '''

        :param x1: Matrix of dimension n1 x m where n1 = number of query points in x1
        :param x2: Matrix of dimension n2 x m where n2 = number of feature points in x2, m2 = dimension of each feature
        :return: A matrix of dimensoin: n1 x n2
        '''

        is_not_diagonal = len(x1) != len(x2) # replacing not all(x1 == x2) for faster checking. Does not matter in GPIRL


        K = []
        for i in xrange(len(x1)):
            row = []
            xi = x1[i]
            for j in xrange(len(x2)):
                xj = x2[j]
                row.append(self.k(xi, xj, 'ARD',lambd=kwargs['lambd'], beta=kwargs['beta'], i_neq_j= ((i!=j) or is_not_diagonal), sigma_sq = self.sigma_sq))#0.5e-2
            K.append(row)


        return np.array(K)


    def grad_lambda_K_coeffs(self, x1, x2):
        is_not_diagonal = len(x1) != len(x2)  # replacing (not all(x1 == x2)) for faster check. Does not matter in GPIRL
        m = np.shape(x1)[1]
        coeff_matrice = []
        for k in xrange(m):
            K = []
            for i in xrange(len(x1)):
                row = []
                xi = x1[i]
                for j in xrange(len(x2)):
                    xj = x2[j]
                    row.append(-0.5* (xi[k] - xj[k])**2 - (is_not_diagonal or (i!=j))*self.sigma_sq )
                K.append(row)
            coeff_matrice.append(np.array(K))
        return coeff_matrice

    def irl(self, demonstrations, w_range=[0, 1], learning_eps=0.3, svf_epsilon=.001, learning_rate=0.1, epochs = 200, verbose = True):
        Xu = []
        added = set([])
        l = []
        print 'len(demonstrations)',len(demonstrations)
        for traj in demonstrations:
            for s,_ in traj:
                if s in added: continue
                if len(Xu) == 0: Xu = np.array([self.mdp_gw.get_feature_vector(s)])
                else: Xu = np.append(Xu, [self.mdp_gw.get_feature_vector(s)], axis = 0)
                added |= set((s,))
                l.append(s)
        n = np.shape(Xu)[0]
        m = np.shape(Xu)[1] # No. of features

        # u = np.random.rand(n, 1)
        u = np.reshape([1.0]*n, (n,1))
        beta = random.uniform(0,1)
        beta = 0.5
        lambd = np.random.rand(m)
        lambd = np.array([1.0]*m)
        learning_rate = 0.07
        print '-----------------------------------------------'
        epochs = 15
        while epochs>0:
            epochs -=1
            K_u_u = self.K(Xu, Xu, lambd = lambd, beta = beta)
            K_u_u_inv = np.linalg.inv(K_u_u)

            reward_handle = lambda s: self.K([self.mdp_gw.get_feature_vector(s)], Xu,lambd = lambd, beta = beta).dot(K_u_u_inv).dot(u)[0][0] #[0][0] for unpacking 1x1 matrix to scalar

            # for i in xrange(len(l)):
            #     print reward_handle(l[i]), u[i]

            mdp = GridMDP(self.mdp_gw.rawgrid_from_reward(reward_handle), terminals=[], gamma=self.mdp_gw.gamma)

            pi, _, _ = soft_value_iteration(mdp)
            mu_cap = defaultdict(lambda:0, self.mu_cap_sa)
            mu_tilde = defaultdict(lambda:0, self.savf_from_policy(pi))
            grad_u_LG = -K_u_u_inv.dot(u).T
            grad_r_LD = np.array([[]])
            grad_u_r = np.empty(shape = (0,n)) # size of u
            grad_theta_r = []
            coeffs_uu = self.grad_lambda_K_coeffs(Xu, Xu)
            coeffs_uu.append(1./beta)
            grad_theta_Kuu = []
            alpha = K_u_u_inv*u
            grad_theta_LG = np.array([])
            grad_theta_LH = np.array([])

            for i in xrange(len(coeffs_uu)):
                grad_theta_Kuu.append(np.multiply(coeffs_uu[i], K_u_u))
                grad_theta_LG = np.append(grad_theta_LG, 0.5 * np.trace((alpha.dot(alpha.T) - K_u_u_inv).dot(grad_theta_Kuu[-1])))
                grad_H = np.trace(K_u_u_inv.dot(K_u_u_inv).dot(K_u_u_inv).dot(grad_theta_Kuu[-1]))
                if i != len(coeffs_uu) - 1:grad_H -= 1. / (1 + sum(lambd))  # lambda and not beta
                grad_theta_LH = np.append(grad_theta_LH, grad_H)
                # print grad_theta_LH, grad_theta_LH[0] - 1. / (1 + sum(lambd))
                # sys.exit()
            # Add dow_Kuu/dow_beta
            # grad_theta_Kuu.append(1. / beta * K_u_u)

            for s in mdp.states:
                feature = self.mdp_gw.get_feature_vector(s)
                K_r_u = self.K([feature], Xu, lambd=lambd, beta=beta)
                coeffs_ru = self.grad_lambda_K_coeffs([feature], Xu)
                coeffs_ru.append(1./beta)
                drdu_vec = []
                for i in xrange(len(coeffs_ru)):
                    drdu_vec.append((np.multiply(coeffs_ru[i], K_r_u) - K_r_u.dot(K_u_u_inv).dot(grad_theta_Kuu[i])).dot(K_u_u_inv).dot(u)[0][0])  # corresponding to lambda

                    # drdu_vec.append(1. / beta * (K_r_u - K_r_u.dot(K_u_u_inv).dot(K_u_u)).dot(K_u_u_inv).dot(u)[0][0])  # corresponding to beta
                for a in mdp.actions(s):
                    grad_r_LD = np.append(grad_r_LD, [[mu_cap[(s,a)] - mu_tilde[(s,a)]]], axis = 1)
                    grad_u_r = np.append(grad_u_r, K_r_u.dot(K_u_u_inv), axis=0)
                    grad_theta_r.append(drdu_vec)

            # print np.shape(grad_r_LD), np.shape(grad_u_r), np.shape(grad_u_LG)
            # sys.exit()
            grad_theta_r = np.array(grad_theta_r)
            grad_u_objective = grad_r_LD.dot(grad_u_r) + grad_u_LG
            grad_theta_objective = grad_r_LD.dot(grad_theta_r) + grad_theta_LG + grad_theta_LH

            print max(abs(grad_u_objective[0])), max (abs(grad_theta_objective[0]))

            # print np.shape(u), np.shape(grad_u_objective.T)
            u += learning_rate* grad_u_objective.T
            lambd +=  learning_rate * grad_theta_objective[0,0:-1].T
            beta += learning_rate * grad_theta_objective[0,-1]

            print beta
            print lambd

            sys.exit()
            # print 'u'
            # print u
            # print 'lambda'
            # print lambd
            # print 'beta'
            # print beta

            # if max(abs(grad_theta_objective[0])) < 0.05:
        print 'Going to show real reward after 1 secs. BE ALERT'
        self.visualize_results(start_states, self.mdp_gw.grid, mdp.grid, pi_star, pi, reverse = True)
        self.grid_gui.render_mdp(mdp)
        self.grid_gui.hold_gui()


    def savf_from_policy(self, pi, eps = 0.0001):
        '''
        Compute state action vistation frequency under given policy
        '''

        mu_tilde = defaultdict(lambda : 0 , {})
        while True:
            delta = 0
            mu_tilde_s = defaultdict(lambda : 0 , {})
            for s in self.mdp_gw.states:

                mu_tilde_s[s] += self.v_s[s]
                for a in self.mdp_gw.actions(s):
                    for p,sz in self.mdp_gw.T(s,a):
                        mu_tilde_s[sz] += self.mdp_gw.gamma * p * mu_tilde[(s,a)]

            # mu_tilde = {}
            for s in pi:
                for pi_p,a in pi[s]:
                    if (s, a) not in mu_tilde: p_mu_tilde = float('inf')
                    else: p_mu_tilde = mu_tilde[(s, a)];
                    mu_tilde[(s, a)] = pi_p * mu_tilde_s[s]
                    delta = max(delta , abs(mu_tilde[(s, a)] - p_mu_tilde))

            if delta < eps:
                return mu_tilde




# m = MaxEntIRL(window_size = 800, grid_size = 5, cell_size = 1, gamma = 0.99)
# w,pi = m.irl(n_start_states= 20, n_repeat = 10, w_range = [0,1], eps = 0, learning_rate = 0.01)

if __name__ == "__main__":
    g = Gpirl(window_size=800, grid_size=5, cell_size=1, gamma=0.99, world='obj', stochastic = True)
    # g.grid_gui.record_demonstration()
    pi_star = get_stochastic_representation(policy_iteration(g.mdp_gw))
    n_start_states = 100
    n_repeat = 1
    start_states = g.sample_start_states(n_start_states)
    demonstrations = roll_out_policy(g.mdp_gw, pi_star, start_states, n_repeat, 5)
    # for traj in demonstrations:
    #     g.execute_trajectory(traj)
    g.compute_demonstration_savf(demonstrations)
    w, _ = g.irl(demonstrations, w_range=[-1, 1], learning_eps=0, svf_epsilon=0.001, learning_rate=0.01, epochs=10,
          verbose=True)

    # for s in g.v_s:
    #     print s, g.v_s[s]