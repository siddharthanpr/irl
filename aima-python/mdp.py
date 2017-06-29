"""Markov Decision Processes (Chapter 17)

First we define an MDP, and the special case of a GridMDP, in which
states are laid out in a 2-dimensional grid.  We also represent a policy
as a dictionary of {state:action} pairs, and a Utility function as a
dictionary of {state:number} pairs.  We then define the value_iteration
and policy_iteration algorithms."""

from utils import *
from copy import deepcopy
from collections import defaultdict
from collections import deque

class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of P(s' | s, a) being a probability number for each
    state/state/action triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 646]"""

    def __init__(self, init, actlist, terminals, gamma=.9, stochastic = False):
        update(self, init=init, actlist=actlist, terminals=terminals,
               gamma=gamma, states=set(), reward={})
        self.stochastic = stochastic

    def R(self, state, action = None):
        "Return a numeric reward for this state, action pair."
        if state in self.reward:
            return self.reward[state]
        if action is None:
            r = 0
            for a in self.actions(state):
                r += self.reward[state, a]
            return float(r)/len(self.actions(state))
        if (state,action) not in self.reward:
            return self.reward[(state)]
        return self.reward[(state,action)]

    def T(self, state, action):
        """Transition model.  From a state and an action, return a list
        of (probability, result-state) pairs."""
        abstract

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9, stochastic = False, shaping=False, actlist = orientations, reward = None, obs = []):

        grid = deepcopy(grid)
        grid.reverse() ## because we want row 0 on bottom, not on top
        orientations = actlist
        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, gamma=gamma, stochastic = stochastic)
        update(self, grid=grid, rows=len(grid), cols=len(grid[0]))
        if shaping:
            self.shaping_reward = defaultdict(lambda:0 , {})
        for x in xrange(self.cols):
            for y in xrange(self.rows):
                r = grid[y][x]
                if r != 0 and shaping:
                    for xs in xrange(self.cols):
                        for ys in xrange(self.rows):
                            self.shaping_reward[(xs,ys)] += - .001*r * ((x-xs) ** 2 + (y-ys)**2) # reward/1000 * log of gaussian with mean at x,y

                for a in self.actions((x, y)):
                    self.reward[(x, y),a] = r
                if grid[y][x] is not None:
                    self.states.add((x, y))

        if reward is not None: self.reward = reward
        for o in obs:
            self.states.remove(o)
        self.states_set = self.states
        self.states = list(self.states)
        self.states.sort()

        self.a2ha = {}
        self.action_dict = defaultdict(lambda : self.actlist,{})
        for s in self.states:
            self.action_dict[s] = set(self.actlist)
        for a in self.actlist:
            self.a2ha[a] = self.action2hidden_action(a)


    def set_grid(self, reverse = True):
        '''
        Sets grid reward from self.reward
        '''
        for s in self.states:
            if reverse:
                self.grid[self.rows - s[1] - 1][s[0]] = self.R(s)
            else:
                self.grid[s[1]][s[0]] = self.R(s)


    def get_random_policy(self):
        pi = {}
        for i in xrange(len(self.grid)):
            for j in xrange(len(self.grid[0])):
                pi[i,j] = random.choice(self.actions((i,j)))
        return pi

    def action2hidden_action(self, action):


        if action is None:
            return [(0.0, None)]

        if not self.stochastic:
            return [(1.0,action)]
        else:
            return [(0.8, action),
                    (0.1, turn_right(action)),
                    (0.1, turn_left(action))]

    def T(self, state, action):
        if action not in self.actions(state):
            raise Exception('Trying to take invalid action from state')
        if action in self.a2ha:
            p_a = self.a2ha[action]
        else:
            p_a = self.action2hidden_action(action)

        ret = []
        for p,a in p_a:
            if a is None:
                ret.append((0.0, state))
            ret.append((p,self.go(state,a)))
        return ret

    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = vector_add(state, direction)
        return if_(state1 in self.states_set, state1, state)

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x,y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0):'>', (0, 1):'^', (-1, 0):'<', (0, -1):'v', None: '.'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))

    def get_s_to_come(self, state):
        '''
        Returns the state of reaching a target state
        :param state: target state
        :return: format list of states
        '''

        ret = []
        for a in self.actlist:
            direction = (-a[0], -a[1])
            neighbour = vector_add(state, direction)
            if neighbour in self.states_set:
                ret.append(neighbour)
            else:
                ret.append(state)
        return ret


    def get_sa_to_come(self, state):
        '''
        Returns the state action pairs and probability of reaching a target state
        :param state: target state
        :return: format [((state,action), prob)]
        '''

        ret = []
        for a in self.actlist:
            direction = (-a[0], -a[1])
            neighbour = vector_add(state, direction)
            # if neighbour == (1,5): # test case
            #     continue
            isbump = vector_add(state, a) not in self.states_set

            if isbump:
                for p, ha in self.a2ha[a]:
                    if ha == a:
                        ret.append(((state,a), p))
            if neighbour in self.states_set:
                for p,ha in self.a2ha[a]:
                    ret.append(((neighbour, ha), p))

        return ret

    def get_support_to_goal(self, goal_states):

        open_queue = deque(goal_states)
        closed_set = set(goal_states)
        support = defaultdict(lambda: set([]), {o:self.action_dict[o] for o in goal_states if o in self.states_set}) # state action pairs to be returned

        while open_queue: # while not empty
            current_state = open_queue.popleft()


            for (s,a),p in self.get_sa_to_come(current_state):
                support[s].add(a)
                if s not in closed_set:
                    closed_set.add(s)
                    open_queue.append(s)
        return support

    def get_support_to_obstacles(self, obstacles):
        open_queue = deque(obstacles)
        support = defaultdict(lambda: set([]), {o:self.action_dict[o] for o in obstacles if o in self.states_set}) # forbidden state action pairs to be returned
        closed_set = set(obstacles) # expanded nodes

        while open_queue:# while not empty
            current_state = open_queue.popleft()
            support[current_state] = set(self.actions(current_state))
            for (s,a),p in self.get_sa_to_come(current_state):
                support[s].add(a)
                if support[s] == set(self.action_dict[s]) and s not in closed_set: # if all actions are forbidden, add the state to openlist
                    closed_set.add(s)
                    open_queue.append(s)
        return support

    def set_permissive_strategy(self, goals, obstacles):
        # ps = {} #permissive strategy
        goal_support = self.get_support_to_goal(goals)
        obs_support = self.get_support_to_obstacles(obstacles)
        for s in goal_support:
            ps = goal_support[s] - obs_support[s]
            if not ps:
                self.states_set.remove(s)
                



    def rawgrid_from_reward(self, reward):
        '''
        Return the grid (unreversed) with which constructing an mdp will fetch the actual reward using R(s)
        :param reward: reward function handle
        :return: rawgrid
        '''
        grid_size = len(self.grid)
        grid = [[0.0] * grid_size  for i in xrange(grid_size)]
        for i in xrange(grid_size):
            for j in xrange(grid_size):
                world_pose = (j, grid_size - 1 - i)
                grid[i][j] = reward(world_pose)
        return grid
#______________________________________________________________________________

Fig[17,1] = GridMDP([[-0.04, -0.04, -0.04, +1],
                     [-0.04, None,  -0.04, -1],
                     [-0.04, -0.04, -0.04, -0.04]],
                    terminals=[(3, 2), (3, 1)])

#______________________________________________________________________________

def value_iteration(mdp, epsilon=0.001, seedU = None):
    "Solving an MDP by value iteration. [Fig. 17.4]"
    if seedU == None: U1 = defaultdict(lambda:0, {})
    else: U1 = seedU.copy()
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:

        # U = U1
        # U = U1.copy()
        delta = 0
        for s in mdp.states_set:
            p_U1s = U1[s]
            # print mdp.actions(s)
            U1[s] = R(s) + gamma * max([sum([p * U1[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - p_U1s))
        if delta < epsilon * (1 - gamma) / gamma:
             return U1

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp.states_set:
        pi[s] = argmax(mdp.actions(s), lambda a:expected_utility(a, s, U, mdp))
    return pi

def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])

#______________________________________________________________________________

def policy_iteration(mdp): # Not converging for (0,0):1, (0,1):1, (1,1):-1, (1,0):0
    "Solve an MDP by policy iteration [Fig. 17.7]"
    U = dict([(s, 0) for s in mdp.states])
    pi = dict([(s, random.choice(mdp.actions(s))) for s in mdp.states])
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), lambda a: expected_utility(a,s,U,mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi

def policy_evaluation(pi, U, mdp, k=20): 
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum([p * U[s1] for (p, s1) in T(s, pi[s])])
    return U

__doc__ += """
>>> pi = best_policy(Fig[17,1], value_iteration(Fig[17,1], .01))

>>> Fig[17,1].to_arrows(pi)
[['>', '>', '>', '.'], ['^', None, '^', '.'], ['^', '>', '^', '<']]

>>> print_table(Fig[17,1].to_arrows(pi))
>   >      >   .
^   None   ^   .
^   >      ^   <

>>> print_table(Fig[17,1].to_arrows(policy_iteration(Fig[17,1])))
>   >      >   .
^   None   ^   .
^   >      ^   <
"""

__doc__ += random_tests("""
>>> pi
{(3, 2): None, (3, 1): None, (3, 0): (-1, 0), (2, 1): (0, 1), (0, 2): (1, 0), (1, 0): (1, 0), (0, 0): (0, 1), (1, 2): (1, 0), (2, 0): (0, 1), (0, 1): (0, 1), (2, 2): (1, 0)}

>>> value_iteration(Fig[17,1], .01)
{(3, 2): 1.0, (3, 1): -1.0, (3, 0): 0.12958868267972745, (0, 1): 0.39810203830605462, (0, 2): 0.50928545646220924, (1, 0): 0.25348746162470537, (0, 0): 0.29543540628363629, (1, 2): 0.64958064617168676, (2, 0): 0.34461306281476806, (2, 1): 0.48643676237737926, (2, 2): 0.79536093684710951}

>>> policy_iteration(Fig[17,1])
{(3, 2): None, (3, 1): None, (3, 0): (0, -1), (2, 1): (-1, 0), (0, 2): (1, 0), (1, 0): (1, 0), (0, 0): (1, 0), (1, 2): (1, 0), (2, 0): (1, 0), (0, 1): (1, 0), (2, 2): (1, 0)}

""")

