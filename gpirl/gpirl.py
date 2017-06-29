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



class Gpirl(MaxEntIRL):
    def __init__(self, window_size=800, grid_size=5, cell_size=1, gamma=0.9, world='object_world', stochastic=False):
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
                s += -0.5* kwargs['lambd'][k] * (xi[k] - xj[k]) * (xi[k] - xj[k]) - kwargs['i_neq_j'] * kwargs['sigma_sq'] * trace
                # print s
            return kwargs['beta'] * np.exp(s)

        if name == 'ARD_alt':
            assert len(xi) == len(kwargs['lambd'])
            trace = sum(kwargs['lambd'])
            s = 0
            e = np.exp
            for k in xrange(len(xi)):
                zk = lambda x: -(x-kwargs['c'][k])/kwargs['l'][k]
                w = lambda x: 1./(1+e(zk(x)))
                w_sig = lambda x : 1./kwargs['l'][k] * (1./(e(zk(x))+2+e(-zk(x)))) + kwargs['s'][k]
                s += -0.5 * (kwargs['lambd'][k] * (w(xi[k]) - w(xj[k]))**2 + kwargs['i_neq_j'] * kwargs['sigma_sq'] * (w_sig(xi[k]) + w_sig(xj[k])))
            # print kwargs['beta'] * np.exp(s)
            return kwargs['beta'] * np.exp(s)


    def savf_from_demonstration(self, demonstrations):
        self.mu_cap_sa = {}  # state action pair visitation frequency.
        self.v_s = {}

        for traj in demonstrations:
            for (s, a) in traj:
                if (s, a) not in self.mu_cap_sa: self.mu_cap_sa[(s, a)] = 0
                if s not in self.v_s: self.v_s[s] = 0
                self.mu_cap_sa[(s, a)] += 1
                self.v_s[s] += 1
                for p, sz in self.mdp_gw.T(s, a):
                    if sz not in self.v_s: self.v_s[sz] = 0
                    self.v_s[sz] -= self.mdp_gw.gamma * p

        self.v_s = defaultdict(lambda: 0, self.v_s)

    def K(self, x1, x2, **kwargs):
        '''

        :param x1: Matrix of dimension n1 x m where n1 = number of query points in x1
        :param x2: Matrix of dimension n2 x m where n2 = number of feature points in x2, m2 = dimension of each feature
        :return: A matrix of dimensoin: n1 x n2
        '''

        is_not_diagonal = len(x1) != len(x2)  # replacing not all(x1 == x2) for faster checking. Does not matter in GPIRL

        K = []
        for i in xrange(len(x1)):
            row = []
            xi = x1[i]
            for j in xrange(len(x2)):
                xj = x2[j]
                if kwargs['name'] == 'ARD': row.append(self.k(xi, xj, kwargs['name'], lambd=kwargs['lambd'], beta=kwargs['beta'], i_neq_j=((i != j) or is_not_diagonal), sigma_sq=self.sigma_sq))  # 0.5e-2
                elif kwargs['name'] == 'ARD_alt': row.append(self.k(xi, xj, kwargs['name'], lambd=kwargs['lambd'], beta=kwargs['beta'], i_neq_j=((i != j) or is_not_diagonal), sigma_sq=self.sigma_sq, l = kwargs['l'], s = kwargs['s'], c = kwargs['c']))  # 0.5e-2
            K.append(row)

        return np.array(K)

    def grad_lambda_K_coeffs(self, x1, x2):
        is_not_diagonal = len(x1) != len(x2)  # replacing (not all(x1 == x2)) for faster check. Does not matter in GPIRL
        m = np.shape(x1)[1]
        coeff_matrice = []
        for k in xrange(m):
            coeff = []
            for i in xrange(len(x1)):
                row = []
                xi = x1[i]
                for j in xrange(len(x2)):
                    xj = x2[j]
                    row.append(-0.5 * (xi[k] - xj[k]) ** 2 - (is_not_diagonal or (i != j)) * self.sigma_sq)
                coeff.append(row)
            coeff_matrice.append(np.array(coeff))
        return coeff_matrice

    def grad_theta_K(self, x1, x2, **kwargs):
        is_not_diagonal = len(x1) != len(x2)  # replacing (not all(x1 == x2)) for faster check. Does not matter in GPIRL
        e = np.exp
        m = np.shape(x1)[1]

        matrice_lambda = []
        matrice_c = []
        matrice_l = []
        matrice_s = []

        for k in xrange(m):
            coeff_lambda = []
            coeff_c = []
            coeff_l = []
            coeff_s = []

            lk = kwargs['l'][k]
            lambda_k = kwargs['lambd'][k]
            for i in xrange(len(x1)):

                def zk(s):
                    return -(s - kwargs['c'][k]) / lk

                def h(s):
                    z = zk(s)
                    return 1. / (e(z) + 2 + e(-z))

                def g(s):
                    z = zk(s)
                    return 2. / (e(2 * z) + 3 * e(z) + 3 + e(-z))

                def wk(s):
                    return 1. / (1 + e(zk(s)))

                row_lambda = []
                row_c = []
                row_l = []
                row_s = []
                xi = x1[i]

                for j in xrange(len(x2)):
                    xj = x2[j]
                    if kwargs['name'] == 'ARD_alt':
                        row_lambda.append(-0.5 * (wk(xi[k]) - wk(xj[k])) ** 2 - (is_not_diagonal or (i != j)) * self.sigma_sq)
                    elif kwargs['name'] == 'ARD':
                        row_lambda.append(-0.5 * ((xi[k]) - (xj[k])) ** 2 - (is_not_diagonal or (i != j)) * self.sigma_sq)
                    row_c.append(-0.5 * lambda_k * ((wk(xi[k]) - wk(xj[k])) * ((-h(xi[k]) + h(xj[k])) / lk) + (is_not_diagonal or (i != j)) * ((h(xi[k]) - g(xi[k]) + h(xj[k]) - g(xj[k])) / lk ** 2)))
                    row_l.append(-0.5 * lambda_k * ((wk(xi[k]) - wk(xj[k])) * ((h(xi[k]) * zk(xi[k]) - h(xj[k]) * zk(xj[k])) / lk) + (is_not_diagonal or (i != j)) * ((g(xi[k]) * zk(xi[k]) + h(xi[k]) * (zk(xi[k]) - 1) + g(xj[k]) * zk(xj[k]) + h(xj[k]) * (zk(xj[k]) - 1)) / lk ** 2)))
                    row_s.append(-0.5 * lambda_k * ((is_not_diagonal or (i != j)) * 2))

                coeff_lambda.append(row_lambda)
                coeff_c.append(row_c)
                coeff_l.append(row_l)
                coeff_s.append(row_s)

            matrice_lambda.append(np.multiply(coeff_lambda, kwargs['K']))
            matrice_c.append(np.multiply(coeff_c, kwargs['K']))
            matrice_l.append(np.multiply(coeff_l, kwargs['K']))
            matrice_s.append(np.multiply(coeff_s, kwargs['K']))

        return {'lambda': matrice_lambda, 'c': matrice_c, 'l': matrice_l, 's': matrice_s, 'beta': [1. / kwargs['beta'] * kwargs['K']]}

    def irl(self, demonstrations, w_range=[0, 1], learning_eps=0.3, svf_epsilon=.001, learning_rate=0.1, epochs=200, verbose=True):
        Xu = []
        added = set([])
        print 'len(demonstrations)', len(demonstrations)
        for traj in demonstrations:
            for s, _ in traj:
                if s in added: continue
                Xu.append(self.mdp_gw.get_feature_vector(s))
                added |= set((s,))
        Xu = np.array(Xu)
        n = np.shape(Xu)[0]  # no. of inducing points
        m = np.shape(Xu)[1]  # No. of features, k

        # u = np.random.rand(n, 1)
        u = np.reshape([1.0] * n, (n, 1))

        # beta = random.uniform(0, 1)
        beta = 0.5
        # lambd = np.random.rand(m)
        lambd = np.array([1.0] * m)

        def param_pdf(theta): # likelihood of parameters
            Kuu  = self.K(Xu, Xu, name = 'ARD', lambd = theta[:-1], beta = theta[-1])
            Kuu_inv = np.linalg.inv(Kuu)
            s = 0
            for i in xrange(m):
                s += np.log(theta[i] + 1)

            return np.exp(-0.5 * np.trace(Kuu_inv.dot(Kuu_inv)) - s)


        theta = get_samples_from_dist(param_pdf, 0, 10, m+1, 1, 1000)[0]
        lambd = theta [:-1]
        beta = theta[-1]


        l_p = np.random.normal(0, 1, m)
        s_p = np.random.normal(0, 1, m)
        c_p = np.random.gamma(2,2, m)


        learning_rate = 0.07 # remove this
        epochs = 10
        print '-----------------------------------------------'
        kernel_name = 'ARD'
        mu_cap = defaultdict(lambda: 0, self.mu_cap_sa)

        while epochs > 0:
            epochs -= 1
            if epochs < 6:
                kernel_name = 'ARD_alt'
                # print 'switched kernels'

            K_u_u = self.K(Xu, Xu, name = kernel_name, lambd = lambd, l = l_p, beta = beta, s = s_p, c = c_p)
            K_u_u_inv = np.linalg.inv(K_u_u)
            K_u_u_inv_3 =  K_u_u_inv.dot(K_u_u_inv).dot(K_u_u_inv)
            reward_handle = lambda s: self.K([self.mdp_gw.get_feature_vector(s)], Xu, name = kernel_name, lambd=lambd, l = l_p, beta = beta, s = s_p, c = c_p).dot(K_u_u_inv).dot(u)[0][0]  # [0][0] for unpacking 1x1 matrix to scalar
            mdp = GridMDP(self.mdp_gw.rawgrid_from_reward(reward_handle), terminals=[], gamma=self.mdp_gw.gamma)
            pi, _, _ = soft_value_iteration(mdp)
            mu_tilde = defaultdict(lambda: 0, self.savf_from_policy(pi))

            grad_u_LG = -K_u_u_inv.dot(u).T

            grad_r_LD = []
            grad_u_r = []
            alpha = K_u_u_inv * u
            grad_theta_Kuu = self.grad_theta_K(Xu, Xu, name = kernel_name,c = c_p, lambd = lambd, l = l_p, beta = beta, K = K_u_u)
            grad_theta_LG = {}
            grad_theta_LH = {}
            for param in grad_theta_Kuu:
                grad_theta_LG[param] = []
                grad_theta_LH[param] = []
                for i in xrange(len(grad_theta_Kuu[param])):
                    grad = grad_theta_Kuu[param][i]
                    grad_theta_LG[param].append(0.5 * np.trace((alpha.dot(alpha.T) - K_u_u_inv).dot(grad)))
                    if param == 'lambda':grad_theta_LH[param].append(np.trace(K_u_u_inv_3.dot(grad)) - 1. / (1 + sum(lambd)))
                    elif param == 'l': grad_theta_LH[param].append(np.trace(K_u_u_inv_3.dot(grad)) - l_p[i])
                    elif param == 'c': grad_theta_LH[param].append(np.trace(K_u_u_inv_3.dot(grad)) + 1./c_p[i] -1./2)
                    elif param == 's': grad_theta_LH[param].append(np.trace(K_u_u_inv_3.dot(grad)) - s_p[i])
                    elif param == 'beta': grad_theta_LH[param].append(np.trace(K_u_u_inv_3.dot(grad)))

            grad_theta_r = {}

            for s in mdp.states: #TODO make this ordered
                feature = self.mdp_gw.get_feature_vector(s)
                K_r_u = self.K([feature], Xu, name = kernel_name, lambd = lambd, l = l_p, beta = beta, s = s_p, c = c_p)
                grad_theta_Kru = self.grad_theta_K([feature], Xu, name = kernel_name,c = c_p, lambd = lambd, l = l_p, beta = beta, K = K_r_u)

                for param in grad_theta_Kru:
                    if param not in grad_theta_r: grad_theta_r[param] = []
                    dr_s_dtheta = []
                    for i in xrange(len(grad_theta_Kru[param])):
                        dr_s_dtheta.append((grad_theta_Kru[param][i] - K_r_u.dot(K_u_u_inv).dot(grad_theta_Kuu[param][i])).dot(K_u_u_inv).dot(u)[0][0])

                    for a in mdp.actions(s): # actions are ordered
                        grad_theta_r[param].append(dr_s_dtheta)

                for a in mdp.actions(s): # actions are ordered
                    grad_r_LD.append(mu_cap[(s, a)] - mu_tilde[(s, a)])
                    grad_u_r.append( K_r_u.dot(K_u_u_inv)[0])

            grad_r_LD = np.array(grad_r_LD)
            grad_u_r = np.array(grad_u_r)

            #compute gradients
            grad_u_objective = grad_r_LD.dot(grad_u_r) + grad_u_LG
            grad_lambda_objective = grad_r_LD.dot(grad_theta_r['lambda']) + grad_theta_LG['lambda'] + grad_theta_LH['lambda']
            grad_l_objective = grad_r_LD.dot(grad_theta_r['l']) + grad_theta_LG['l'] + grad_theta_LH['l']
            grad_m_objective = grad_r_LD.dot(grad_theta_r['c']) + grad_theta_LG['c'] + grad_theta_LH['c']
            grad_s_objective = grad_r_LD.dot(grad_theta_r['s']) + grad_theta_LG['s'] + grad_theta_LH['s']
            grad_beta_objective = grad_r_LD.dot(grad_theta_r['beta']) + grad_theta_LG['beta'] + grad_theta_LH['beta']

            #update params.
            u += learning_rate * grad_u_objective.T
            lambd += learning_rate * np.reshape(grad_lambda_objective, np.shape(lambd))
            l_p += learning_rate * np.reshape(grad_l_objective, np.shape(l_p))
            c_p += learning_rate * np.reshape(grad_m_objective, np.shape(c_p))
            s_p += learning_rate * np.reshape(grad_s_objective, np.shape(s_p))
            beta += learning_rate * grad_beta_objective[0]


            print beta
            # print c_p
            print lambd

            # sys.exit()
        print 'Going to show real reward after 1 secs. BE ALERT'
        self.visualize_results(start_states, self.mdp_gw.grid, mdp.grid, pi_star, pi, reverse=True)
        self.grid_gui.render_mdp(mdp)

        self.grid_gui.hold_gui()

    def savf_from_policy(self, pi, eps=0.0001):
        '''
        Compute state action vistation frequency under given policy
        '''

        mu_tilde = defaultdict(lambda: 0, {})
        while True:
            delta = 0
            mu_tilde_s = defaultdict(lambda: 0, {})
            for s in self.mdp_gw.states:

                mu_tilde_s[s] += self.v_s[s]
                for a in self.mdp_gw.actions(s):
                    for p, sz in self.mdp_gw.T(s, a):
                        mu_tilde_s[sz] += self.mdp_gw.gamma * p * mu_tilde[(s, a)]

            # mu_tilde = {}
            for s in pi:
                for pi_p, a in pi[s]:
                    if (s, a) not in mu_tilde:
                        p_mu_tilde = float('inf')
                    else:
                        p_mu_tilde = mu_tilde[(s, a)];
                    mu_tilde[(s, a)] = pi_p * mu_tilde_s[s]
                    delta = max(delta, abs(mu_tilde[(s, a)] - p_mu_tilde))

            if delta < eps:
                return mu_tilde


# m = MaxEntIRL(window_size = 800, grid_size = 5, cell_size = 1, gamma = 0.99)
# w,pi = m.irl(n_start_states= 20, n_repeat = 10, w_range = [0,1], eps = 0, learning_rate = 0.01)

def get_samples_from_dist(pdf, low, high, dim, n_samples, n_pop = 10000):
    '''

    :param p: The function handle to the pdf
    :param dim: dimension of the imput space of pdf
    :param n_samples:
    :param n_pop:
    :return:
    '''
    prob = []
    samples = []
    for i in xrange(n_pop):
        s = (high - low) * np.random.rand(dim) - low * np.ones(dim)
        samples.append(s)
        prob.append(pdf(s))

    prob = np.array(prob)
    prob /= sum(prob)
    if n_samples == 1:
        return [samples[list(prob).index(max(prob))]]
    indice = np.random.choice(len(samples), n_samples, p=prob)
    return [samples[i] for i in indice]


if __name__ == "__main__":
    g = Gpirl(window_size=800, grid_size=5, cell_size=1, gamma=0.99, world='object_world', stochastic=True)
    # g.grid_gui.record_demonstration()
    pi_star = get_stochastic_representation(policy_iteration(g.mdp_gw))
    n_start_states = 100
    n_repeat = 1
    start_states = g.sample_start_states(n_start_states)
    demonstrations = roll_out_policy(g.mdp_gw, pi_star, start_states, n_repeat, 5)
    # for traj in demonstrations:
    #     g.execute_trajectory(traj)
    g.savf_from_demonstration(demonstrations)
    w, _ = g.irl(demonstrations, w_range=[-1, 1], learning_eps=0, svf_epsilon=0.001, learning_rate=0.01, epochs=15,
                 verbose=True)

    # for s in g.v_s:
    #     print s, g.v_s[s]
