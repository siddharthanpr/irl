
from max_ent_irl import *


class Agent:
    w = None
    psi = None
    def __init__(self, w, psi):
        self.w = w
        self.psi = psi


class MultiAgentIRL(MaxEntIRL):
    def __init__(self, num_agents = 2, window_size=800, grid_size=128, cell_size=1, gamma=0.99, world='indicator_grid', stochastic=True, w_range = [-1,1]):
        self.w_range = w_range
        self.num_agents=num_agents
        MaxEntIRL.__init__(self, window_size, grid_size, cell_size, gamma=gamma, world=world, stochastic=stochastic)

        if num_agents > 0:
            self.agents = [self.get_random_w() for i in xrange(num_agents)]
            self.psi = np.array([1. / num_agents] * num_agents)

        else:
            self.agents = []
            self.psi = []

    def get_random_w(self):
        if self.paired:
            feature_size = len(self.mdp_gw.get_feature_vector(((0, 0), (0,0)) , paired = True))
        else:
            feature_size = len(self.mdp_gw.get_feature_vector((0, 0), paired=False))

        return np.random.uniform(self.w_range[0], self.w_range[1], size=(feature_size,))

    def add_agent(self):
        self.agents.append(self.get_random_w())

    def get_trajectory_savf(self, trajectories, beta, j,  eps=0.001):
        '''
        Get state visitation frequencies for a set of trajectories given indicator features
        Intentionally modified so that get_expected_svf and this converge at same point for equal lenght in deterministic setting
        '''
        D = defaultdict(lambda: 0, {})
        for t in xrange(len(trajectories[0])): # t is time index
            discount = self.mdp_gw.gamma ** t
            D_copy = deepcopy(D)
            for i in xrange(len(trajectories)):
                traj = trajectories[i]
                s, a = traj[t]
                D[s, a] += discount * float(beta[i][j]) / len(trajectories)
            delta = 0
            for e in D:
                delta = max(delta, abs(D_copy[e] - D[e]))
            if delta < eps: return D
        return D

    def get_trajectory_svf(self, trajectories, beta, j,  eps=0.001):
        '''
        Get state visitation frequencies for a set of trajectories given indicator features
        Intentionally modified so that get_expected_svf and this converge at same point for equal lenght in deterministic setting
        '''
        D = defaultdict(lambda: 0, {})
        for t in xrange(len(trajectories[0])): # t is time index
            discount = self.mdp_gw.gamma ** t
            D_copy = deepcopy(D)
            for i in xrange(len(trajectories)):
                traj = trajectories[i]
                s, _ = traj[t]
                if s not in D: D[s] = 0
                D[s] += discount * float(beta[i][j]) / len(trajectories)
            delta = 0
            for s in D:
                delta = max(delta, abs(D_copy[s] - D[s]))
            if delta < eps: return D
        return D

    def em_irl_crp(self, demonstrations, svf_epsilon=.001, learning_rate=0.1, epochs = 200, alpha = 0.5, verbose = True):

        resample_prob = 0.0
        demo_length = len(demonstrations[0])
        n_samples = 10
        n = len(demonstrations)
        nc = [] # No of demonstrations in each class
        c = [-1.0] * n # Class assignment of each demonstration
        m = 10
        while epochs > 0:

            empty_classes = set(range(len(nc)))
            beta = [defaultdict(lambda: 0, {}) for i in xrange(n)]
            epochs -= 1
            pi = {}
            mdp = {}
            dr_dw = self.get_linear_reward_gradient()
            print epochs, nc

            for i in xrange(n):

                if verbose:
                    print 'bfore, i,c,nc', i,c,nc

                if -1.0 not in c:
                    assert len(c)==sum(nc)

                if c[i] >= 0:
                    nc[c[i]] -= 1
                p = np.array(nc + [alpha]).astype(float)
                p /= p.sum()
                self.add_agent()

                assert len(self.agents) == len(p)
                for j in xrange(len(p)):

                    if j not in mdp:
                        mdp[j] = GridMDP(self.get_zero_grid(), terminals=[], gamma=self.mdp_gw.gamma,
                                         actlist=self.mdp_gw.actlist,
                                         reward=self.get_reward_from_w(self.agents[j], dr_dw), obs=self.mdp_gw.cars)
                        # mdp[j] = GridMDP(self.mdp_gw.grid_from_w(self.agents[j]), terminals=[], gamma=self.mdp_gw.gamma)
                        pi[j], _, _ = soft_value_iteration(mdp[j])
                    p[j] *= get_trajectory_likelihood(demonstrations[i], self.mdp_gw, pi[j])

                if p.sum() == 0:
                    p += 1
                p /= p.sum()

                # new_class = False
                # sampled_indice = np.random.choice(len(p), len(p), p=p)
                # counts = np.array([0.0] * len(p))
                # for s in sampled_indice:
                #     counts[s] += 1
                sampled_index = np.random.choice(len(p), 1, p=p)[0]
                if verbose:
                    print 'sampled_index', sampled_index
                if sampled_index == len(p) - 1: # New class sampled
                    nc.append(1)

                else:
                    del self.agents[-1]
                    if np.random.uniform() < resample_prob:
                        del  mdp[len(p) - 1]
                        del  pi[len(p) - 1]
                    nc[sampled_index] += 1


                c[i] = sampled_index

                if sampled_index in empty_classes:
                    empty_classes.remove(sampled_index)

                beta[i][sampled_index] = 1

            for j in xrange(len(self.agents)):

                if self.paired:
                    D = self.get_trajectory_savf(demonstrations, beta, j, eps=svf_epsilon)
                    Dg = defaultdict(lambda: [], {})  # reformat to show in gui
                    for s, a in D:
                        Dg[s].append([a, D[s, a]])
                    self.grid_gui.D = Dg
                else:
                    D = self.get_trajectory_svf(demonstrations, beta, j, eps=svf_epsilon)
                    self.grid_gui.D = D

                # D = self.get_trajectory_svf(demonstrations, beta, j, eps=svf_epsilon)
                f_ex = self.get_feature_expectation(D)

                initial_distribution = {}
                for i in xrange(n):
                    d = demonstrations[i]
                    s = d[0][0]
                    if s not in initial_distribution: initial_distribution[s] = 0
                    initial_distribution[s] += beta[i][j] / len(demonstrations)

                Dpi = svf_from_policy(mdp[j], pi[j], length=demo_length, initial_distribution=initial_distribution, eps=svf_epsilon)

                if self.paired:
                    Dpi = savf_from_svf(Dpi, pi[j])
                f = self.get_feature_expectation(Dpi)

                self.agents[j] += learning_rate * (f_ex - f)

            #Remove classes if they were never sampled
            nc = [nc[x] for x in xrange(len(nc)) if x not in empty_classes]
            self.agents = [self.agents[x] for x in xrange(len(self.agents)) if x not in empty_classes]
            empty_classes = np.array(list(empty_classes))
            for i in xrange(n):
                c[i] -= sum (empty_classes < c[i])
            if verbose:
                print 'empty_classes', empty_classes
                print 'after, i,c,nc', i, c, nc

        # Remove unsampled classes from mdp to return
        mdp = {i:mdp[i] for i in xrange(len(mdp)) if i < len(nc)}

        # for i in mdp:
        #     print i , mdp[i].grid
        self.psi = np.array(nc).astype(float)/ sum(nc)
        return self.agents[0], pi, mdp, beta, []

    def em_irl_fixed(self, demonstrations, svf_epsilon=.001, learning_rate=0.1, epochs = 200, verbose = True):

        n = len(demonstrations)
        beta = [np.array([0.0] * len(self.agents)) for i in xrange(n)]
        demo_length = len(demonstrations[0])
        demo_length = float('inf')
        confidences = []
        dr_dw = self.get_linear_reward_gradient()
        while epochs>0:
            print epochs
            epochs -= 1
            pi = {}
            mdp = {}
            for j in xrange(len(self.agents)):

                mdp[j] = GridMDP(self.get_zero_grid(), terminals=[], gamma=self.mdp_gw.gamma, actlist=self.mdp_gw.actlist,
                              reward=self.get_reward_from_w(self.agents[j], dr_dw), obs=self.mdp_gw.cars)

                # mdp[j] = GridMDP(self.mdp_gw.grid_from_w(self.agents[j]), terminals=[], gamma=self.mdp_gw.gamma)
                pi[j], _, _ = soft_value_iteration(mdp[j])

                for i in xrange(n):
                    # print i, j, get_trajectory_likelihood(demonstrations[i], self.mdp_gw, pi[j])
                    beta[i][j] = get_trajectory_likelihood(demonstrations[i], self.mdp_gw, pi[j]) * self.psi[j]
            # input()
            confidence = 0
            for i in xrange(n):
                confidence += beta[i].sum()
                if beta[i].sum() == 0:
                    beta[i] += 1
                beta[i] /= beta[i].sum()

            confidences.append(confidence)

            for j in xrange(len(self.agents)):

                if self.paired:
                    D = self.get_trajectory_savf(demonstrations, beta, j, eps=svf_epsilon)
                    Dg = defaultdict(lambda: [], {})  # reformat to show in gui
                    for s, a in D:
                        Dg[s].append([a, D[s, a]])
                    self.grid_gui.D = Dg
                else:
                    D = self.get_trajectory_svf(demonstrations, beta, j, eps=svf_epsilon)
                    self.grid_gui.D = D

                f_ex = self.get_feature_expectation(D)


                initial_distribution = {}
                for i in xrange(n):
                    d = demonstrations[i]
                    s = d[0][0]
                    if s not in initial_distribution: initial_distribution[s] = 0
                    initial_distribution[s] +=  beta[i][j] / len(demonstrations)

                # Ds = get_expected_svf(mdp[j], pi[j], length= len(demonstrations[0]), initial_distribution=initial_distribution, eps=svf_epsilon)
                Dpi = svf_from_policy(mdp[j], pi[j], length=demo_length, initial_distribution=initial_distribution, eps=svf_epsilon)

                if self.paired:
                    Dpi = savf_from_svf(Dpi, pi[j])

                f = self.get_feature_expectation(Dpi)

                self.agents[j] += learning_rate * (f_ex - f)
                s_beta = 0
                for i in xrange(n):
                    s_beta += beta[i][j]
                self.psi[j] = s_beta


            self.psi /= self.psi.sum()

            print 'PSI', self.psi

        for i in xrange(len(mdp)):
            mdp[i].set_grid(reverse=False)

        return self.agents[0], pi[0], mdp, beta, confidences

    def visualize_results(self, index, grids, beta, c, save = False):

        ncols = len(grids)
        plt.figure(1+3*index)
        for i in xrange(ncols):
            grid = grids[i]
            plt.subplot(1, ncols, i+1)
            plt.pcolor(np.array(grid))
            plt.colorbar()
            plt.title("Agent %d" %(i+1) +'\n Psi = ' + str(self.psi[i]))
        if save:
            plt.savefig('rewards' + str(index) + '.png')

        plt.figure(2+3*index)
        plt.title('Psi' + str(self.psi))
        for j in xrange(ncols):
            b = []
            for i in xrange(len(beta)):
                b.append(beta[i][j])

            plt.subplot(1, ncols, j+1)
            plt.plot(b)
            plt.title("Agent %d" %(j+1)+'\n Psi = ' + str(self.psi[j]))
            x1, x2, y1, y2 = plt.axis()
            plt.axis([x1,x2, -0.1,1.1])

        if save: plt.savefig('P(c|D)' + str(index) + '.png')

        plt.figure(3+3*index)

        plt.title("Confidence metric: Sum of unnormalized beta_ij \n" +'Psi = ' + str(self.psi))
        plt.plot(c)

        if save:  plt.savefig('confidence' + str(index) + '.png')
        plt.show()




def run(num_agents, world = 'indicator_grid', grid_dim = 10, show_demo = True):

    ns = np.arange(1, 2) * 100
    # ns = np.arange(1, 4) * 50
    n_repeat = 1



    n_agents = num_agents
    if n_agents == 0:
        n_agents = np.random.randint(2, 3)
        # n_agents = 1
        # n_agents = np.random.randint(2, 5)

    demonstrations = []

    print 'Creating demo for', n_agents, 'agents'

    if world == 'race_track':
        m = MultiAgentIRL(num_agents=num_agents, window_size=800, grid_size=grid_dim, cell_size=1, gamma=0.9,
                          world=world, stochastic=False)
        demo_length = grid_dim[0]
        b = min(m.grid_dim)
        l = max(m.grid_dim)
        policies = [m.mdp_gw.get_aggressive_policy, m.mdp_gw.get_evasive_policy]
        # policies = [get_aggressive_policy]
        # policies.reverse()
        f_ex =[]
        if m.paired:
            feature_names = ['action == (direcx,1) and dely_a > b                                                       ',
                             'action == (direcx,1) and dely_a <= b and dely_a > 0                                       ',
                             'action == (direcx, 1) and dely_a == 0                                                     ',
                             'state == aggressive_sub_target and action == (np.sign(aggressive_target[0] - state[0]), 1)',
                             'action == (direcx_e,1) and dely_e > b                                                     ',
                             'action == (direcx_e,1) and dely_e <= b and dely_e > 0                                     ',
                             'action == (direcx_e,1) and dely_e == 0                                                    ',
                             'y                                                                                         ']
        else:
            feature_names = ['aggressive indicator distance',
                             'aggressive indicator         ',
                             'is colliding                 ',
                             'evasive distance             ',
                             'evasive indicator            ',
                             'y                            ']
        for i in xrange(n_agents):

            n_agent_start_states = np.random.choice(ns, 1)[0]
            print '%d th agent: Policy %s, no. of demonstrations %d' % (i, policies[i].__name__, n_agent_start_states)
            agent_pi = policies[i]()
            xs = np.random.choice(np.arange(b/2)*2 + 1, n_agent_start_states) # start from non car columns
            agent_start_states = [(xs[i],0) for i in xrange(len(xs))]
            print 'agent_start_states', agent_start_states
            p = roll_out_policy(m.mdp_gw, agent_pi, agent_start_states, n_repeat, demo_length)
            demonstrations += p
            if m.paired:
                D = MaxEntIRL.get_trajectory_savf(m,p)
            else:
                D = MaxEntIRL.get_trajectory_svf(m, p)
            f_ex.append(m.get_feature_expectation(D))
        print ''
        print ''
        for i in xrange(len(feature_names)):
            print feature_names[i],  f_ex[0][i], f_ex[1][i]

    else:
        m = MultiAgentIRL(num_agents=num_agents, window_size=800, grid_size=grid_dim, cell_size=1, gamma=0.9,
                          world=world, stochastic=False)
        demo_length = 2 * grid_dim[0]
        ac = {0:[(0.5, (0, 1)), (0.5, (1, 0))], 1: [(0.5, (0, -1)), (0.5, (-1, 0))], 2:  [(0.5, (0,-1)), (0.5, (1,0))], 3: [(0.5, (0, 1)), (0.5, (-1, 0))]}
        for i in xrange(n_agents):
            n_agent_start_states = np.random.choice(ns, 1)[0]
            agent_pi = {}

            print '%d th agent: Policy %s, no. of demonstrations %d' % (i, str(ac[i]), n_agent_start_states)
            for s in m.mdp_gw.states:
                a = deepcopy(ac[i])
                for p_a in a: # remove bumping actions at edges
                    if s == m.mdp_gw.go(s, p_a[1]):
                        a.remove(p_a)
                        break
                p = float(sum([p[0] for p in a]))
                a = [(x[0]/p, x[1]) for x in a]

                agent_pi[s] = a
            agent_start_states = m.sample_start_states(n_agent_start_states)
            demonstrations += roll_out_policy(m.mdp_gw, agent_pi, agent_start_states, n_repeat, demo_length)

    if show_demo:
        for traj in demonstrations:
            m.execute_trajectory(traj)

    for index in xrange(20):
        if num_agents != 0:
            m.agents = [m.get_random_w() for i in xrange(num_agents)]
            m.psi = np.array([1./num_agents] * num_agents)
            mdps = {}
            w, pi, mdps, beta, c = m.em_irl_fixed(demonstrations, svf_epsilon=0.001, learning_rate=0.01, epochs=40, verbose=False)
        else:
            m.agents = []
            m.psi = []
            w, pi, mdps, beta, c = m.em_irl_crp(demonstrations, svf_epsilon=0.001, learning_rate=0.03, epochs=40, verbose=False)

        grids = []

        for i in xrange(len(mdps)):
            grids.append(mdps[i].grid)
        print 'done'

        m.visualize_results(index, grids, beta, c)

        for _ in xrange(10):
            for j in pi:
                # continue
                m.show_learned(agent_start_states, None, pi[j], caption = 'Cluster %d'%j, length=demo_length)


if __name__ == '__main__':
    run(0, 'race_track', grid_dim = (40,3), show_demo=0)

    m.grid_gui.hold_gui()





