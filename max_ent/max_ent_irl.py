import numpy as np
from collections import defaultdict
import random
import sys

sys.path.append('/home/siddharth/thesis/code/irl/aima-python')
from mdp import *
from gui import *
from matplotlib import pyplot as plt
from objectworld import ObjWorld
from indicatorgridworld import IndicatorWorld
from racetrack import RaceTrack

class MaxEntIRL:
    def __init__(self, window_size=800, grid_dim=(5,5), cell_size=1, gamma=0.9, world = 'indicator_grid', stochastic = False):

        self.window_size = window_size
        self.grid_dim = grid_dim
        self.grid_size = max(grid_dim)
        self.cell_size = cell_size
        self.paired = True # state action pair (True) or just state (False)
        pix_per_grid = (window_size / self.grid_size)

        if world == 'indicator_grid':

            self.mdp_gw = IndicatorWorld([], self.grid_size, cell_size, gamma, stochastic = stochastic)
            # self.grid = self.mdp_gw.grid
            self.actual_w = self.mdp_gw.true_weight

        elif world == 'race_track':
            self.mdp_gw = RaceTrack([], grid_dim, cell_size, gamma, stochastic=stochastic)
            # self.grid = self.mdp_gw.grid
            # self.actual_w = self.mdp_gw.true_weight

        elif world == 'object_world':
            self.mdp_gw = ObjWorld([], self.grid_size,  gamma=gamma, stochastic = stochastic)

        self.grid_gui = GuiBoard("MaxEnt IRL", [window_size / self.grid_size * grid_dim[1],
                                             window_size / self.grid_size * grid_dim[0]], pix_per_grid)

        self.grid_gui.render_mdp(self.mdp_gw, world =world)

        # print self.mdp_gw.grid
        #
        # self.grid_gui.hold_gui()

    def get_maxent_policy(self, theta, N=200):

        '''
        Obsolete - need changes
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
            current_state = self.mdp_gw.go(current_state, sample_action(pi[current_state]))
        self.grid_gui.render_move(current_state, delay=0.1)

    def get_reward_from_w(self, w, phi):
        r = {}
        i = 0
        for s in self.mdp_gw.states:
            if self.paired:
                for a in self.mdp_gw.actions(s):
                    r[s,a] = w.dot(phi[i])
                    i += 1
            else:
                r[s] = w.dot(phi[i])
                i += 1

        return r

    def get_zero_grid(self):
        return [[0.0] * self.grid_dim[1] for i in xrange(self.grid_dim[0])]

    def irl(self, demonstrations, w_range=[0, 1], learning_eps=0.3, svf_epsilon=.001, learning_rate=.1, epochs = 200, verbose = True):

        #### Get Demonstrations and feature expectations of demonstrations
        # In order to use same convergence for demonstrations and svf calculations

        demo_length = len(demonstrations[0])
        # print len(demonstrations[0]), demo_length
        # demo_length =3 * self.grid_size
        pi_star = get_stochastic_representation(policy_iteration(self.mdp_gw))
        if self.paired:
            D = self.get_trajectory_savf(demonstrations, eps=svf_epsilon)
            Dg = defaultdict(lambda: [], {})  # reformat to show in gui
            for s, a in D:
                Dg[s].append([a, D[s, a]])
            self.grid_gui.D = Dg

        else:
            D = self.get_trajectory_svf(demonstrations, eps=svf_epsilon)
            self.grid_gui.D = D



        f_ex = self.get_feature_expectation(D)
        print "Computed f_ex", f_ex

        #### Initialize random weight and gradient
        initial_distribution = {}
        for d in demonstrations:
            s = d[0][0]
            if s not in initial_distribution: initial_distribution[s] = 0
            initial_distribution[s] += 1. / len(demonstrations)

        ########## Test if get_expected_svf and get_trajectory_svf have same behavior in case of deterministic MDP (they should have same behavior)
        #
        # D = get_expected_svf(self.mdp_gw, pi_star, length=demo_length, initial_distribution=initial_distribution,
        #                      eps=svf_epsilon)
        # f_ex2 = self.get_feature_expectation(D)
        # print np.linalg.norm(f_ex - f_ex2)
        # print f_ex
        # print f_ex2
        #
        # sys.exit()

        best = float('inf')
        if self.paired:
            feature_size = len(self.mdp_gw.get_feature_vector(((0,0),(0,0)), paired = True))
        else:
            feature_size = len(self.mdp_gw.get_feature_vector((0,0), paired = False))

        grad = np.array([float('inf')] * feature_size)
        w = np.random.uniform(w_range[0], w_range[1], size=(feature_size,))
        # mu_cap = self.mu_cap_sa_vec
        dr_dw = self.get_linear_reward_gradient()
        print "Starting Gradient Descent"
        while (epochs > 0 and max(abs(grad) > learning_eps)):
            epochs -= 1
            mdp = GridMDP(self.get_zero_grid(), terminals=[], gamma=self.mdp_gw.gamma, actlist=self.mdp_gw.actlist, reward= self.get_reward_from_w(w,dr_dw), obs = self.mdp_gw.cars)
            pi, _, _ = soft_value_iteration(mdp)

            Dpi = svf_from_policy(mdp, pi, length=demo_length, initial_distribution=initial_distribution, eps=svf_epsilon)

            if self.paired:
                Dpi = savf_from_svf(Dpi, pi)

            f = self.get_feature_expectation(Dpi)
            grad = f_ex - f

            w += learning_rate * grad
            if verbose:
                print ' ||gradient||_infinity =', max(abs(grad)), ' mean(gradient) =', np.mean(grad)

            L = 0

            for d in demonstrations:
                L += get_trajectory_likelihood(d, self.mdp_gw, pi)
            print 'traj_likelihood', L


        print 'f_ex', f_ex
        print 'learned f', f
        self.grid_gui.pi = pi
        mdp.set_grid()
        return w, pi, mdp

    def get_feature_expectation(self, D):
        if self.paired:
            m = len(self.mdp_gw.get_feature_vector((self.mdp_gw.states[0], self.mdp_gw.actions(self.mdp_gw.states[0])[0]),
                paired = True))  # TODO assumes all states have equal number of actions
        else:
            m = len(self.mdp_gw.get_feature_vector(self.mdp_gw.states[0],
                                               paired=False))  # TODO assumes all states have equal number of actions

        f = np.zeros(m)
        for e in D:
            f += self.mdp_gw.get_feature_vector(e, paired = self.paired) * D[e]
        return f

    def sample_start_states(self, no=1):
        sampled_states = []
        for i in xrange(no):
            while True:
                sampled_state = (random.randint(0, self.grid_dim[1] - 1), random.randint(0, self.grid_dim[0] - 1))
                if self.mdp_gw.R(sampled_state) <= 0:
                    sampled_states.append(sampled_state)
                    break
        return sampled_states

    def get_trajectory_svf(self, trajectories, eps=0.001):
        '''
        Get state visitation frequencies for a set of trajectories
        Intentionally modified so that get_expected_svf and this converge at same point for equal lenght in deterministic setting
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


    def get_trajectory_savf(self, trajectories, eps=0.001):

        '''
        Get state action visitation frequencies for a set of trajectories
        Intentionally modified so that get_expected_svf and this converge at same point for equal lenght in deterministic setting
        '''
        D = defaultdict(lambda :0, {})
        for t in xrange(len(trajectories[0])):
            discount = self.mdp_gw.gamma ** t
            D_copy = deepcopy(D)
            for traj in trajectories:
                s, a = traj[t]
                D[s,a] += discount * 1. / len(trajectories) # averaging over trajectories
            delta = 0
            for e in D:
                delta = max(delta, abs(D_copy[e] - D[e]))
            if delta < eps: return D
        return D

    def show_learned(self, start_states, pi_star, pi, length=100, caption = None):

        if caption is not None:
            pygame.display.set_caption(caption)
            self.execute_policy(start_states[0], pi, length)
            return

        for index, start_state in enumerate(start_states):
            pygame.display.set_caption("Demonstration %d / %d" % (index, len(start_states)))
            self.execute_policy(start_state, pi_star, length)
            pygame.display.set_caption("Learned behavior %d / %d" % (index, len(start_states)))
            self.execute_policy(start_state, pi, length)

    def execute_trajectory(self, traj, delay = 0.1):
        for s,a in traj:
            self.grid_gui.render_move(s, delay=delay)

    def get_linear_reward_gradient(self):
        '''
        Finds the gradient dr(s,a)/dw for linear rewards, i.e r = w^Tf. Assumes all states have equal no. of |a| actions
        :return: dr(s,a)/dw an sxa by m matrix if self.paired, s by m matrix otherwise. 
        '''
        ns = len(self.mdp_gw.states)
        na =  len(self.mdp_gw.actions(self.mdp_gw.states[0]))
        if self.paired:
            m = len(self.mdp_gw.get_feature_vector((self.mdp_gw.states[0], self.mdp_gw.actions(self.mdp_gw.states[0])[0]), paired = True)) #TODO assumes all states have equal number of actions
        else:
            m = len(self.mdp_gw.get_feature_vector(self.mdp_gw.states[0], paired = self.paired))

        if self.paired: grad = np.zeros((ns*na, m))
        else: grad = np.zeros((ns, m))

        i = 0
        for s in self.mdp_gw.states:
            if self.paired:
                for a in self.mdp_gw.actions(s):
                    grad[i,:] = self.mdp_gw.get_feature_vector((s,a), self.paired)
                    i += 1
            else:
                grad[i, :] = self.mdp_gw.get_feature_vector(s, self.paired)
                i += 1


        return grad

    def visualize_results(self, start_states, truth, learned, pi_star, pi, reverse=True):
        if reverse:
            truth.reverse()
            learned.reverse()
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.pcolor(np.array(truth))
        plt.colorbar()
        plt.title("Groundtruth reward")
        plt.subplot(1, 2, 2)
        plt.pcolor(np.array(learned))
        plt.colorbar()
        plt.title("Recovered reward")
        plt.show()

        self.show_learned(start_states, pi_star, pi, length=1* len(truth))

    def savf_from_policy(self, pi, eps=0.0001, vec = True):
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
                if not vec :
                    return mu_tilde
                # convert to vector form and return
                S = len(self.mdp_gw.states)
                A = len(self.mdp_gw.actions(self.mdp_gw.states[0]))
                f = np.zeros(S*A)
                i = 0
                for s in self.mdp_gw.states:
                    for a in self.mdp_gw.actions(s):
                        f[i] = mu_tilde[(s, a)]
                        i+=1
                return f

    def savf_from_demonstration(self, demonstrations):
        self.mu_cap_sa = defaultdict(lambda: 0,{})  # state action pair visitation frequency. Actually not required for computations
        self.v_s = defaultdict(lambda: 0,{})

        for traj in demonstrations:
            for (s, a) in traj:
                # if (s, a) not in self.mu_cap_sa: self.mu_cap_sa[(s, a)] = 0
                # if s not in self.v_s: self.v_s[s] = 0
                self.mu_cap_sa[(s, a)] += 1
                self.v_s[s] += 1
                for p, sz in self.mdp_gw.T(s, a):
                    # if sz not in self.v_s: self.v_s[sz] = 0
                    self.v_s[sz] -= self.mdp_gw.gamma * p

        # self.v_s = defaultdict(lambda: 0, self.v_s)
        # self.mu_cap_sa = defaultdict(lambda: 0, self.mu_cap_sa)

        # convert to vector form and return
        S = len(self.mdp_gw.states)
        A = len(self.mdp_gw.actions(self.mdp_gw.states[0]))
        self.mu_cap_sa_vec = np.zeros(S * A)
        self.v_s_vec = np.zeros(S * A)
        i = 0
        j = 0
        for s in self.mdp_gw.states:
            self.v_s_vec[i] = self.v_s[s]
            i += 1
            for a in self.mdp_gw.actions(s):
                self.mu_cap_sa_vec[j] = self.mu_cap_sa[(s, a)]
                j+=1


def savf_from_svf(svf, pi):

    savf = defaultdict(lambda: 0, {})
    for s in svf:
        for p,a in pi[s]:
            savf[s,a] = svf[s]*p
    return savf


def sample_action(prob_action):
        '''
        Samples an action from pi[s] (= prob_action)
        :param prob_action: pi[s], the stochastic representation of policy at state s
        :return: sampled action
        '''
        p = np.array([prob[0] for prob in prob_action])
        sampled_index = np.random.choice(len(p), 1, p=p)[0]
        return prob_action[sampled_index][1]

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

def get_trajectory_likelihood(traj, mdp, pi):

    '''
    TODO: make it more effiecient for non sparse T and high dimensional actions
    :param traj:
    :param mdp:
    :param pi:
    :return:
    '''
    p = 1

    for i in xrange(len(traj)):
        s,a = traj[i]

        found_action = False
        for p_a,ac in pi[s]:
            if ac == a:
                p *= p_a
                found_action = True
                break
        if not found_action: return 0

        if i+1 < len (traj):
            sn, _ = traj[i + 1]
            for p_t, s_t in mdp.T(s,a):
                if s_t == sn:
                    p *= p_t
    return p





def svf_from_policy(mdp, pi, initial_distribution, length=float('inf'), eps=0.001):
    '''
    get state visitation frequencies Algorithm 9.3 in Ziebart's thesis

    Note that this does not have same behavior as roll_out_policy in case of high values of eps (>1). This is only because we do not compute delta for first iteration (going from 0 to initial svf)
    '''

    D = initial_distribution.copy()
    T, gamma = mdp.T, mdp.gamma

    Ds = defaultdict(lambda:0, initial_distribution)

    mm_converged = False
    discount = 1
    while length > 1:
        length -= 1

        discount *= mdp.gamma

        if not mm_converged:  # If motion model not converged yet
            D_prime = defaultdict(lambda : 0, {})
            delta = 0
            for sx in D:
                for pi_s_a, a in pi[sx]:
                    for P, sz in T(sx, a):
                        D_prime[sz] += D[sx] * pi_s_a * P

            for s in D:
                delta = max(delta, abs(D[s] - D_prime[s]))
            D = D_prime

        if (delta < 0.001 * eps): mm_converged = True

        delta = 0
        for s in D: #TODO make this more effieceient after mm_converged with 1/1-gamma
            Ds_t = discount * D[s]
            Ds[s] += Ds_t
            delta = max(delta, Ds_t)

        if (delta < eps) and length == float('inf'):
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
            # assert mdp.paired == False
            V_soft_prime[s] = 0 #R(s) # Does not matter how we initialize it
            for a in mdp.actions(s):
                V_soft_prime[s] = soft_max_2(V_soft_prime[s],
                                             R(s,a) + gamma * sum([p * V_soft[s1] for (p, s1) in T(s, a)]))
            delta = max(delta, abs(p_V_soft - V_soft_prime[s]))

        V_soft = V_soft_prime
        if delta < epsilon * (1. - gamma) / gamma:

            Q_soft = {}
            pi = {}
            for s in mdp.states:
                pi[s] = []
                nomalizing_factor = 0
                for a in mdp.actions(s):
                    Q_soft[(s, a)] = R(s,a) + gamma * sum([p * V_soft[s1] for (p, s1) in T(s, a)])

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


def run(policy_handle, world = 'indicator_grid', grid_dim = 10, show_demo = True):
    """
    
    :param policy_handle: Function handle that returns a policy
    :param world: options: indicator_grid', 'race_track', 'object_world'
    :param grid_dim: l x b
    :param show_demo: if True runs the demo in graphics
    :return: 
    """

    ns = np.arange(1,4)*50
    n_repeat = 1
    demo_length = 1 * grid_dim[0]
    m = MaxEntIRL(window_size = 800, grid_dim=grid_dim, cell_size=1, gamma=0.95, world=world, stochastic=False)
    print len(m.mdp_gw.get_support_to_goal([(0,0)]))
    print 'here', m.mdp_gw.get_support_to_goal([(0, 0)])[(1, 4)]
    print 'here', m.mdp_gw.get_support_to_goal([(0, 0)])[(1, 6)]
    print len(m.mdp_gw.cars)
    print 'car', (m.mdp_gw.cars[0])
    print m.mdp_gw.get_support_to_obstacles(m.mdp_gw.cars)[m.mdp_gw.cars[0]]
    print m.mdp_gw.get_support_to_obstacles(m.mdp_gw.cars)[(m.mdp_gw.cars[0][0]-1, m.mdp_gw.cars[0][1]-1)]
    m.grid_gui.hold_gui()

    demonstrations = []

    if world == 'race_track':
        if policy_handle == 'aggressive': #TODO: Change type to handle or to string
            policy_handle = m.mdp_gw.get_aggressive_policy
        elif policy_handle == 'tailgate':
            policy_handle = m.mdp_gw.get_tailgate_policy
        elif policy_handle == 'evasive':
            policy_handle = m.mdp_gw.get_evasive_policy
        else:
            raise Exception('Policy type can only be "aggressive" or "tailgate". Got %s'%policy_handle)
        b = min(m.grid_dim)
        l = max(m.grid_dim)

        n_agent_start_states = np.random.choice(ns, 1)[0]
        print 'Policy %s, no. of demonstrations %d' % (policy_handle.__name__, n_agent_start_states)
        pi_star = policy_handle()
        xs = np.random.choice(np.arange(b/2)*2 + 1, n_agent_start_states) # start from non car columns
        agent_start_states = [(xs[i],0) for i in xrange(len(xs))]
        p = roll_out_policy(m.mdp_gw, pi_star, agent_start_states, n_repeat, demo_length)
        demonstrations += p
        m.grid_gui.pi = pi_star
    else:
        ac = {0:[(0.5, (0, 1)), (0.5, (1, 0))], 1: [(0.5, (0, -1)), (0.5, (-1, 0))], 2:  [(0.5, (0,-1)), (0.5, (1,0))], 3: [(0.5, (0, 1)), (0.5, (-1, 0))]}
        for i in xrange(n_agents):
            n_agent_start_states = np.random.choice(ns, 1)[0]
            agent_pi = {}
            a = ac[i]
            print '%d th agent: Policy %s, no. of demonstrations %d' % (i, str(a), n_agent_start_states)
            for s in m.mdp_gw.states:
                agent_pi[s] = a
            agent_start_states = m.sample_start_states(n_agent_start_states)
            demonstrations += roll_out_policy(m.mdp_gw, agent_pi, agent_start_states, n_repeat, demo_length)


    if show_demo:
        for traj in demonstrations:
            m.execute_trajectory(traj)

    w, pi,mdp = m.irl(demonstrations, w_range=[-1, 1], learning_eps = 0.3, svf_epsilon=.001, learning_rate=0.1, epochs = 40, verbose = True)
    print 'learned weights', w
    m.visualize_results(agent_start_states, deepcopy(m.mdp_gw.grid), deepcopy(mdp.grid), pi_star, pi, reverse=True)



if __name__ == '__main__':
    run('aggressive', world = 'race_track', grid_dim = (40,3), show_demo=0)
    #TODO IMPORTANT reward = None for the obstacles
    #Change go to move function



    # m = MaxEntIRL(window_size=800, grid_dim=(128,128), cell_size=1, gamma=0.8, world='indicator_grid', stochastic = False)
    # n_start_states = 100
    # n_repeat = 1
    # demo_length = 10 * m.grid_size
    # pi_star = get_stochastic_representation(policy_iteration(m.mdp_gw))
    # start_states = m.sample_start_states(n_start_states)
    # demonstrations = roll_out_policy(m.mdp_gw, pi_star, start_states, n_repeat,
    #                                  demo_length)
    #
    # w, pi, mdp = m.irl(demonstrations, w_range=[-1, 1], learning_eps=0, svf_epsilon=0.001, learning_rate=0.001, epochs = 200, verbose = True)
    #
    # m.visualize_results(start_states,deepcopy(m.mdp_gw.grid), deepcopy(mdp.grid), pi_star, pi, reverse = True)
    #
    # m.grid_gui.hold_gui()
