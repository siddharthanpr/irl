from max_ent_irl import *
from tf_utils import *
from es import *


class NNMaxEnt(MaxEntIRL):

    def __init__(self, window_size=800, grid_size=5, cell_size=1, gamma=0.9, world='object_world', stochastic=False, stochastic_policy = False):
        MaxEntIRL.__init__(self, window_size, grid_size, cell_size, gamma=gamma, world=world, stochastic=stochastic)
        self.ordered_actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        actions_dim = len (self.ordered_actions)

        if world == 'object_world':
            state_dim = len(self.mdp_gw.get_feature_vector((0,0)))
        elif world == 'grid_world':
            state_dim = 2

        state_dim = 2
        self.inp = tf.placeholder(tf.float32, [None, state_dim])
        hidden_size = 10
        self.hidden = False
        
        if self.hidden:
            self.l1, self.wh, self.bh = add_layer(self.inp, state_dim, hidden_size, activation_function=None)  # hidden layer
            self.prediction, self.wo, self.bo = add_layer(self.l1, hidden_size, actions_dim, activation_function = tf.nn.sigmoid)
            dim = (1 + state_dim) * hidden_size + (hidden_size + 1) * actions_dim
            self.dists = [(state_dim, hidden_size), (hidden_size, actions_dim)]
            self.weights = [self.wh,self.wo]
            self.biases = [self.bh, self.bo]
        else:
            self.prediction, self.wo, self.bo = add_layer(self.inp, state_dim, actions_dim, activation_function = tf.nn.sigmoid)
            dim = (1 + state_dim) * actions_dim
            self.dists = [(state_dim, actions_dim)]
            self.weights = [self.wo]
            self.biases = [self.bo]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        sigma = 1
        alpha = 0.8
        self.solver = GaussianEvolutionOptimizer(None, sigma, alpha, dim, n_pop = 15, max_generations = 5)
        self.stochastic_policy = stochastic_policy

    def get_stochastic_action(self, state):

        out = self.sess.run(self.prediction, feed_dict={self.inp: [state]})[0]
        out = np.exp(out)
        out = out/out.sum()
        return [(out[i], self.ordered_actions[i]) for i in xrange(len(self.ordered_actions))]

    def get_deterministic_action(self, state):
        # return (0,-1)
        out = self.sess.run(self.prediction, feed_dict={self.inp: [state]})[0]
        return self.ordered_actions[list(out).index(max(out))]

    def set_vec_weigths(self, w_v):

        '''
        Set a given vector of weights to our neural network
        :param w:
        :return:
        '''

        w, b = self.vec_to_weights(w_v, self.dists)

        for i in xrange(len(self.weights)):
            self.sess.run(self.weights[i].assign(w[i]))
            self.sess.run(self.biases[i].assign(b[i]))


        # self.sess.run(self.wh.assign(w[0]))
        # self.sess.run(self.wo.assign(w[1]))
        # self.sess.run(self.bh.assign(b[0]))
        # self.sess.run(self.bo.assign(b[1]))

    def U(self, w_v, mdp):

        self.set_vec_weigths(w_v)

        r = 0
        eps = 0.01
        for s in self.start_states:
            current_state = s
            discount = 1.0

            iter = 0
            while discount > eps and iter < self.grid_size*2:
                rs = - discount * mdp.shaping_reward[current_state] #rs = - gamma^t * phi(s)
                r += discount * mdp.R(current_state)
                # print current_state
                # input()
                if self.stochastic_policy:
                    action = sample_action(self.get_stochastic_action(current_state))
                else:
                    action = self.get_deterministic_action(current_state)
                if current_state == (4,0):
                    pass
                    # action = (1,0)
                current_state = mdp.go(current_state, action)
                rs += mdp.gamma * discount * mdp.shaping_reward[current_state]  #rs = gamma^(t+1)*phi(s')- gamma^t *phi(s)
                # print current_state,r, rs
                # input()
                r += rs # Add shaping term to reward
                discount *= mdp.gamma
                iter += 1

        # input()

        return r

    def weights_to_vec(self, w, b):
        '''
        :param w: List of weights
        :param b: List of biases
        :return: flattened vector
        '''
        np_w = np.array([])
        for i in xrange(len(w)):

            np_w = np.append(np_w, self.sess.run(w[i]).flatten())
            np_w = np.append(np_w, self.sess.run(b[i]).flatten())
        return np_w

    def vec_to_weights(self, vec, dists):

        '''

        :param vec: Vector to be unflattened
        :param dists: list of dimension of each components
        :return: list of weights and a list of biases
        '''
        ptr = 0
        w = []
        b = []
        for n,m in dists:
            w.append(np.resize(vec[ptr:ptr + n * m], (n, m)))
            ptr += n * m
            b.append(np.resize(vec[ptr:ptr + n * m], (1, m)))
            ptr += 1 * m

        return w,b

    def test_vectorization(self):

        if self.hidden:
            v = self.weights_to_vec([self.wh, self.wo], [self.bh, self.bo])
            w, b = self.vec_to_weights(v, self.dists)
            assert (w[0] == self.sess.run(self.wh)).all() and (w[1] == self.sess.run(self.wo)).all() and (b[0] == self.sess.run(self.bh)).all() and (b[1] == self.sess.run(self.bo)).all()
        else:
            v = self.weights_to_vec([self.wo], [self.bo])
            w, b = self.vec_to_weights(v, self.dists)
            assert (w[0] == self.sess.run(self.wo)).all() and (b[0] == self.sess.run(self.bo)).all()

    def approx_soft_value_iteration(self):

        grid = deepcopy(self.mdp_gw.grid)
        grid.reverse()
        mdp = GridMDP(grid, self.mdp_gw.terminals, init=(0, 0), gamma = self.mdp_gw.gamma, stochastic = self.mdp_gw.stochastic, shaping = True)
        self.grid_gui.render_mdp(mdp)
        self.solver.set_utility(lambda w: self.U(w, mdp))
        self.set_vec_weigths(self.solver.solve())

        pi = {}
        for s in mdp.states:
            if self.stochastic_policy:
                pi[s] = self.get_stochastic_action(s)
            else:
                pi[s] = [(1,self.get_deterministic_action(s))]

        for s in self.start_states:
            self.execute_policy(s, pi, self.grid_size*2)

if __name__ == "__main__":
    nn = NNMaxEnt(window_size=800, grid_size=128, cell_size=1, gamma=0.99, world='indicator_grid', stochastic=True, stochastic_policy = False)

    nn.test_vectorization()

    n_start_states = 5
    n_repeat = 1
    nn.start_states = nn.sample_start_states(n_start_states)
    # nn.start_states = [(4, 0), (4, 0), (4, 0), (4, 0)]
    # nn.start_states = [(2, 2), (2, 2), (2, 2), (2, 2)]
    nn.approx_soft_value_iteration()
    nn.grid_gui.hold_gui()
