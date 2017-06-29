import numpy as np


class GaussianEvolutionOptimizer:
    def __init__(self, utility_handle, sigma, alpha, dim, n_pop=50, max_generations=500):
        self.U = utility_handle
        self.init_sigma = sigma
        self.init_alpha = alpha
        self.dim = dim
        self.pop = n_pop
        self.generations = max_generations

    def set_utility(self, utility_handle):
        self.U = utility_handle

    def solve(self):
        alpha = self.init_alpha
        sigma = self.init_sigma
        pop = self.pop
        U = self.U
        eps = 0.001
        max_u = -float('inf')

        w = np.random.randn(self.dim)  # initial guess
        R = np.zeros(self.pop)
        for i in xrange(self.generations):
            E = np.random.randn(self.pop, self.dim)  # matrix of epsilon disturbances
            for j in xrange(self.pop):

                w_try = w + sigma * E[j]
                R[j] = self.U(w_try)
                if max_u < R[j]:
                    max_u = R[j]
                    best_w = w_try

            mean = np.mean(R)
            # if mean > 0 and sigma > .01: sigma /= 10.0
            R_n = (R - mean)  # / np.std(R)
            pu = U(w)
            w += alpha / (pop * sigma) * np.dot(E.T, R_n)  # * 1./np.abs(np.linalg.norm(np.dot(N.T, A)))
            u = U(w)

            if u - pu < 0 and alpha > .0001:
                alpha /= 10.0

            print u
        print 'best', max_u
        return best_w


