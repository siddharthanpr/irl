# from pip._vendor.requests.utils import should_bypass_proxies

from mdp import *
import numpy as np


class RaceTrack(GridMDP):

    def __init__(self, terminals, grid_dim, init=(0, 0), gamma=.9, stochastic = False):
        self.n_next_cars = 1
        self.grid_dim = grid_dim
        self.cars = []
        self.grid = [[0.0] * grid_dim[1] for i in xrange(grid_dim[0])]
        self.init_cars()

        GridMDP.__init__(self, self.grid, terminals, init=(0, 0), gamma=gamma, stochastic = stochastic, actlist= [(1, 0), (1,1), (0, 1),(-1,1),  (-1, 0), (0, -1)], obs = self.cars)

    def init_cars(self):
        evasive_distance = 2

        l = max(self.grid_dim)
        b = min(self.grid_dim)
        # total_cars = np.floor(l/(2*(evasive_distance-1)))
        total_cars = l / (2 * evasive_distance)

        while True: # sample car positions such that they are not too close by
            y_cars = np.random.choice(l, total_cars, replace = False)
            y_cars.sort()
            close_by = False
            for i in xrange(len(y_cars) - 1):
                if abs(y_cars[i] - y_cars[i + 1]) < 3:
                    close_by = True
                    break
            if not close_by:
                break

        car_lanes = np.arange(int(np.ceil(b/2.0))) * 2
        for y_car in y_cars:
            car_pose = (np.random.choice(car_lanes, 1)[0], y_car)
            self.cars.append(car_pose)
            self.grid[l-car_pose[1]-1][car_pose[0]] = -1

        self.next_cars = {}
        self.prev_cars = {}
        car_index = 0

        for y in xrange(l):
            while car_index < len(y_cars) and y_cars[car_index] < y :
                car_index += 1
            self.next_cars[y] = [self.cars[i] for i in xrange(car_index, car_index+ self.n_next_cars) if i < len(y_cars)]

        car_index = len(y_cars) -1

        for y in xrange(l):
            y = l-y-1
            while car_index > 0 and y_cars[car_index] > y:
                car_index -= 1
            self.prev_cars[y] = [self.cars[i] for i in reversed(xrange(car_index - self.n_next_cars, car_index+1)) if i > 0]

    # def get_feature_vector(self, s):
    #     x = s[0]
    #     y = s[1]
    #     f = np.array([0.0] * (self.grid_dim[0] * self.grid_dim[1]))
    #     f[self.macro_cell(self.grid_dim[1] - 1 - y, x)] = 1
    #     return f
    #
    # def grid_from_w(self, w):
    #     '''
    #     get grid world with rewards given the weigts to indicator features
    #     '''
    #     l = [[0.0] * self.grid_dim[1] for i in xrange(self.grid_dim[0])]
    #
    #     for i in xrange(self.grid_dim[0]):
    #         for j in xrange(self.grid_dim[1]):
    #             l[i][j] = w[self.macro_cell(i, j)]
    #     return l
    #
    # def macro_cell(self, x, y):
    #     size = self.grid_dim[0]
    #     cells = 1
    #     cells_per_row = (size / cells)
    #     return x / cells + y / cells * cells_per_row


    def get_feature_vector(self, p, paired = True, y_feature = True):
        '''
        
        :param p: state action pair or just state
        :param paired: True if sending state action pair
        :param y_feature: 
        :return: 
        '''

        if paired:
            state = p[0]
            action = p[1]
        else:
            state = p
            action = None
        y = state[1]
        b = min(self.grid_dim)
        l = max(self.grid_dim)
        next_cars = self.next_cars[y]
        prev_cars = self.prev_cars[y]


        if not paired:
            features = np.zeros(self.n_next_cars * 5 + y_feature)
            if y_feature:
                features[-1] = y
            if len(self.next_cars[state[1]]) == 0:
                return features
            # for i in xrange(len(next_cars)):
            #     features[0*self.n_next_cars+i] = abs(next_cars[i][0]-state[0]) + abs(next_cars[i][1]-state[1])

            # for i in xrange(len(next_cars)):
            #     features[0*self.n_next_cars + i] = next_cars[i][0]-state[0] == 0

            # for i in xrange(len(next_cars)):
            #      features[0*self.n_next_cars + i] = (next_cars[i][1] - state[1])* next_cars[i][0]-state[0] == 0 # tailgator indicator distance

            for i in xrange(len(prev_cars)):
                features[0*self.n_next_cars + i] = (prev_cars[i][1] - state[1]) * prev_cars[i][0]==state[0] # aggressive indicator distance


            for i in xrange(len(prev_cars)):
                features[1*self.n_next_cars + i] = (prev_cars[i][1] - state[1] == -1 and prev_cars[i][0]-state[0] == 0) # aggressive indicator feature

            # for i in xrange(len(next_cars)):
            #      features[3*self.n_next_cars + i] = (next_cars[i][1] - state[1] == 1 and next_cars[i][0]-state[0] == 0) # tailgate indicator feature

            for i in xrange(len(next_cars)):
                features[2*self.n_next_cars + i] = (next_cars[i][1] - state[1] == 0 and next_cars[i][0]-state[0] == 0) # colliding with a car

            for i in xrange(len(next_cars)):
                features[3*self.n_next_cars + i] = abs(next_cars[i][0]-state[0])  # evasive distance to next car

            for i in xrange(len(next_cars)):
                features[4*self.n_next_cars + i] = (next_cars[i][1] == state[1]) and abs(next_cars[i][0]-state[0]) > 1# evasive indicator

            # for i in xrange(len(next_cars)):
            #     features[5*self.n_next_cars + i] = abs(next_cars[i][1]-state[1]) + abs(next_cars[i][0]-state[0])  # evasive distance > b indicator

            # for i in xrange(len(next_cars)):
            #     features[5*self.n_next_cars + i] = abs(next_cars[i][0]-state[0]) > 1# evasive indicator

        else: # getting a state action pair


            n_actions = len(self.actions(state))
            # features = np.zeros(16 + y_feature)
            #
            #
            #
            # if y_feature:
            #     features[-1] = y
            zero_factor = 1
            if len(self.next_cars[state[1]]) == 0:
                # print np.zeros(11)
                # return np.zeros(11)
                next_car_pose = (0,0)
                zero_factor = 0
            else:
                next_car_pose = self.next_cars[state[1]][0]  # consider the first next car

            if len(self.prev_cars[state[1]]) == 0:
                prev_aggressive_target = (-1,-1)
            else:
                prev_car_pose = self.prev_cars[state[1]][0]
                dy = (next_car_pose[1] + 1) < l
                prev_aggressive_target = (prev_car_pose[0], prev_car_pose[1] +dy)  # consider the first next car

            tail_gate_target = (next_car_pose[0], next_car_pose[1] - 1)

            dx = np.sign(state[0] - next_car_pose[0])
            if dx == 0: dx = random.choice([x for x in [1, - 1] if state[0] + x > 0 and state[0] + x < b])
            aggressive_sub_target = (next_car_pose[0] + dx, next_car_pose[1])
            dely_a = aggressive_sub_target[1] - state[1]
            dely_e = next_car_pose[1] - state[1]

            direcx_e = np.sign(state[0] - next_car_pose[0])
            if direcx_e == 0: direcx_e = random.choice([x for x in [1 ,- 1] if state[0] + x > 0 and state[0] + x < b])
            direcx_e *= (state[0]+ direcx_e) >= 0 and (state[0]+ direcx_e) < b

            direcx_a = np.sign(aggressive_sub_target[0] - state[0])
            dy = (next_car_pose[1] + 1) < l
            aggressive_target = (next_car_pose[0], next_car_pose[1] + dy)

            features = [
                state != aggressive_sub_target and action == (direcx_a, 1) and dely_a > b,
                state != aggressive_sub_target and action == (direcx_a,1) and dely_a <= b,
                direcx_a == 0 or state == prev_aggressive_target,
                state == prev_aggressive_target,
                state == aggressive_sub_target and action[0] == np.sign(aggressive_target[0] - state[0]),


                action == (direcx_e,1) and dely_e > b,
                action == (direcx_e,1) and dely_e <= b,
                # action == (direcx_e, 1) and dely_e == 0,

                direcx_e != 0 and action == (0, 1) and dely_e > b,
                direcx_e != 0 and action == (0, 1) and dely_e <= b,
                # direcx_e != 0 and action == (0, 1) and dely_e == 0,

                # direcx_e != 0 and action == (-direcx_e, 1) and dely_e == 0,

                # direcx_a != 0 and action == (0, 1) and dely_a > b,
                # direcx_a != 0 and action == (0, 1) and 0 < dely_a <= b,
                # direcx_a != 0 and action == (0, 1) and dely_a == 0,
                # direcx_a != 0 and action[0] == -direcx_a and dely_a > b,
                # direcx_a != 0 and action[0] == -direcx_a and 0 < dely_a <= b
            ]

            if zero_factor == 0:
                features = [0] * len(features)

            features.append(zero_factor == 0 and action==(0,1))

            if y_feature:
                features.append(y)

            features = np.array(features, dtype=float)


            # for j in xrange(n_actions):
            #     features[0*self.n_next_cars * n_actions + j] = (action == self.actions(state)[j] and dely_a>b) * direcx
            #
            # for j in xrange(n_actions):
            #     features[1*self.n_next_cars * n_actions + j] = (action == self.actions(state)[j] and dely_a <= b and dely_a >0) * direcx
            #
            # for j in xrange(n_actions):
            #     features[2*self.n_next_cars * n_actions + j] = (action == self.actions(state)[j] and dely_a == 0) * direcx
            #
            # for j in xrange(n_actions):
            #     features[3*self.n_next_cars * n_actions + j] = (action == self.actions(state)[j] and dely_e > b) * direcx
            #
            # for j in xrange(n_actions):
            #     features[4*self.n_next_cars * n_actions + j] = (action == self.actions(state)[j] and aggressive_sub_target == state) * direcx
            #
            # for j in xrange(n_actions):
            #     features[5*self.n_next_cars * n_actions + j] = (action == self.actions(state)[j] and dely_e <= b and dely_e >0) * direcx
            #
            # for j in xrange(n_actions):
            #     features[6*self.n_next_cars * n_actions + j] = (action == self.actions(state)[j] and dely_e == 0) * direcx

        return features

    def grid_from_w(self, w, paired = False):
        l = [[0.0] * self.grid_dim[1] for i in xrange(self.grid_dim[0])]
        length = max(self.grid_dim)
        for i in xrange(self.grid_dim[0]):
            for j in xrange(self.grid_dim[1]):
                l[i][j] = np.dot(w,self.get_feature_vector((j,length-1-i)))
        return l

    def get_tailgate_policy(self):
        b = min(self.grid_dim)
        l = max(self.grid_dim)
        agent_pi = {}
        for s in self.states:
            if len(self.next_cars[s[1]]) == 0:
                agent_pi[s] = [(1.0, (0, 1))]
                continue
            next_car_pose = self.next_cars[s[1]][0]
            tail_gate_target = (next_car_pose[0], next_car_pose[1] - 1)
            direcx = np.sign(tail_gate_target[0] - s[0])
            dely = tail_gate_target[1] - s[1]

            if dely > b:  # next car is far away, can take any action
                agent_pi[s] = [(0.6, (direcx, 1)), (0.1, (-direcx, 1)), (0.3, (0, 1))]
            elif dely > 0:
                agent_pi[s] = [(0.9, (direcx, 1)), (0.01, (-direcx, 1)), (0.09, (0, 1))] # next car is clse by a high tendancy to ben in the same lane
            elif dely == 0:
                agent_pi[s] = [(0.9, (direcx, 0)), (0.1, (0, 1))]
            elif dely < 0: # if we are beside a car, just go front never be aggressive
                agent_pi[s] = [(1.0, (0, 1))]

            if tail_gate_target == s:
                agent_pi[s] = [(1, (0, 0)), (0, (1, 1)), (0, (-1, 1))]

        return agent_pi


    def get_aggressive_policy(self):
        b = min(self.grid_dim)
        l = max(self.grid_dim)
        agent_pi = {}
        for s in self.states:
            if len(self.next_cars[s[1]]) == 0:
                agent_pi[s] = [(1.0, (0, 1))]
                continue
            next_car_pose = self.next_cars[s[1]][0]
            tail_gate_target = (next_car_pose[0], next_car_pose[1] - 1)
            dy = (next_car_pose[1] + 1) < l
            aggressive_target = (next_car_pose[0], next_car_pose[1] + dy)
            dx = np.sign(s[0] - next_car_pose[0])
            if dx == 0: dx = random.choice([x for x in [1 ,- 1] if s[0] + x > 0 and s[0] + x < b])
            aggressive_sub_target = (next_car_pose[0] + dx, next_car_pose[1])
            direcx = np.sign(aggressive_sub_target[0] - s[0])
            dely = aggressive_sub_target[1] - s[1]

            if dely > b:  # next car is far away, can take any action
                agent_pi[s] = [(0.6, (direcx, 1)), (0.0, (-direcx, 1)), (0.4, (0, 1))]
            elif dely >= 0:
                agent_pi[s] = [(0.99, (direcx, 1)), (0.00, (-direcx, 1)), (0.01, (0, 1))]
            # elif dely == 0:
            #     agent_pi[s] = [(0.6, (direcx, 0)), (0.4, (direcx, 1))] # move right or left to be agressive without moving forward
            if aggressive_sub_target == s:
                agent_pi[s] = [(1, (np.sign(aggressive_target[0] - s[0]), 1)), (0.0, (0, 1)), (0.0, (np.sign(s[0]- aggressive_target[0]), 1))]
            if tail_gate_target == s:
                if s != self.go(s, (1, 1)):
                    agent_pi[s] = [(1.0, (1, 1))]
                else:
                    agent_pi[s] = [(1.0, (-1, 1))]
        return agent_pi

    def get_evasive_policy(self):
        b = min(self.grid_dim)
        l = max(self.grid_dim)
        agent_pi = {}
        for s in self.states:
            if len(self.next_cars[s[1]]) == 0:
                agent_pi[s] = [(1.0, (0, 1))]
                continue
            next_car_pose = self.next_cars[s[1]][0] # consider the first next car
            tail_gate_target = (next_car_pose[0], next_car_pose[1] - 1)
            direcx = np.sign(s[0] - next_car_pose[0])
            if direcx == 0: direcx = random.choice([x for x in [1 ,- 1] if s[0] + x > 0 and s[0] + x < b])
            direcx *= (s[0]+ direcx) >= 0 and (s[0]+ direcx) < b
            dely = next_car_pose[1] - s[1]

            if dely > b:  # next car is far away, can take any action
                agent_pi[s] = [(0.6, (direcx, 1)), (0.0, (-direcx, 1)), (0.4, (0, 1))]
            elif dely > 0:
                agent_pi[s] = [(0.9, (direcx, 1)), (0.1, (0, 1))]
            elif dely == 0: # if we are overtaking right now
                agent_pi[s] = [(1, (direcx, 1))]

            if tail_gate_target == s:
                if s != self.go(s, (1, 1)):
                    agent_pi[s] = [(1.0, (1, 1))]
                else:
                    agent_pi[s] = [(1.0, (-1, 1))]
        return agent_pi