import pygame
import numpy as np
import sys
import time
from copy import deepcopy


class GuiBoard:
    def __init__(self, title, dim, pix_per_grid):

        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode([dim[0], dim[1]])
        self.dimension = dim
        self.on_event = {pygame.QUIT: self.on_quit, pygame.MOUSEBUTTONDOWN: self.on_click,
                         pygame.KEYDOWN: self.on_keyboard}
        self.max_reward_color = np.array([255, 255, 255])
        self.min_reward_color = np.array([0, 0, 0])
        self.agent_color = np.array([240, 128, 128])
        self.current_state = []
        self.pix_per_grid = pix_per_grid
        self.V = None
        self.pi = None
        self.D = None
        self.driving_status = 'Waiting'
        self.color_mode = 'linear'
        self.rgb = {'r': (255, 0, 0), 'b': (0, 0, 255), 'g': (0, 255, 0), 'c': (0, 255, 255),
                    'o': (255, 165, 0)}  # c is cyan
        self.objects = {}
        self.zero_out = True

    def refresh(self, graphics=True, delay=0):
        pygame.display.update()
        self.check_event()
        if graphics: time.sleep(delay)

    def check_event(self):

        '''
        Checks if any event occured on the gui and calls corresponding function
        '''
        for event in pygame.event.get():
            if event.type in self.on_event:
                return self.on_event[event.type](event)

    def on_keyboard(self, event):
        if event.key == pygame.K_LEFT and self.driving_status != 'Waiting':
            left = (-1, 0)
            if self.driving_status == 'Active':
                self.pi[self.current_state] = left

            s = self.local_mdp.go(self.current_state, left)
            self.render_move(s)
            print 'left'

        if event.key == pygame.K_RIGHT and self.driving_status != 'Waiting':
            right = (1, 0)
            if self.driving_status == 'Active':
                self.pi[self.current_state] = right
            s = self.local_mdp.go(self.current_state, right)
            self.render_move(s)
            print 'right'

        if event.key == pygame.K_UP and self.driving_status != 'Waiting':
            up = (0, 1)
            if self.driving_status == 'Active':
                self.pi[self.current_state] = up

            s = self.local_mdp.go(self.current_state, up)
            self.render_move(s)
            print 'up'

        if event.key == pygame.K_DOWN and self.driving_status != 'Waiting':
            down = (0, -1)
            if self.driving_status == 'Active':
                self.pi[self.current_state] = down

            s = self.local_mdp.go(self.current_state, down)
            self.render_move(s)

            print 'down'

        if event.key == pygame.K_RETURN and self.driving_status == 'Active':
            self.driving_status = 'Done'
            self.recording = [self.start_state, self.pi]
            print 'pociy recorded = ', self.pi
            print 'Finished recording'
            return True

        if event.key == pygame.K_RCTRL and self.driving_status == 'Waiting':
            self.start_state = self.current_state
            self.pi = {}
            self.driving_status = 'Active'
            print 'Start state set. Starting to record moves'

    def add_trigger(self, event_name, slot):
        '''
        Add event slots with name (event_name) and function handle (slot)
        '''
        self.on_event[event_name] = slot

    def on_quit(self, event):
        pygame.quit()
        sys.exit()

    def on_click(self, event):
        '''
        Requires render_reward_grid to be called first.
        '''
        if len(self.current_state) == 0:
            print 'Warning: Please call render_reward_grid before on_click'
            self.refresh()
            return
        x, y = event.pos[0] / self.pix_per_grid, len(self.grid) - 1 - event.pos[
            1] / self.pix_per_grid  # here we flip y-sim because the display has positive y in down direction
        info = "position: (%d,%d) reward: %f" % (x, y, self.grid[y][x])
        if self.V != None: info += ' Value: %f' % (self.V[(x, y)])
        if self.pi != None: info += ' Optimal action: ' + str(self.pi[(x, y)])
        if self.D != None:
            if (x, y) in self.D:
                D = str(self.D[(x, y)])
            else:
                D = str(0)
            info += 'Discounted Visitation: ' + D
        print info

    def get_color(self, reward):
        if reward is None: return [255,0,0]
        if reward == 0 and self.zero_out:
            return (0, 0, 0)
        if self.color_mode == 'linear':
            theta = float(reward - self.r_min) / (self.r_max - self.r_min)
            return tuple((self.max_reward_color * theta + (1 - theta) * self.min_reward_color).astype(int))
        elif self.color_mode == 'split':
            if reward >= 0:
                return self.max_reward_color * float(reward) / self.r_max
            else:
                return self.min_reward_color * float(reward) / self.r_min


    def render_mdp(self, mdp_gw, current_state=(0, 0), world='grid'):
        if world == 'race_track':
            self.max_reward_color = np.array([255, 0, 0])
            self.min_reward_color = np.array([255, 255, 255])
            self.zero_out = True

        if world == 'object_world': self.zero_out = False
        else: self.color_mode = 'split'

        self.current_state = current_state
        self.grid = mdp_gw.grid
        self.local_mdp = mdp_gw
        self.r_max, self.r_min = max([max(x for x in i if x is not None) for i in self.grid]), min([min(x for x in i if x is not None) for i in self.grid])

        self.pix_per_grid = self.dimension[0] / len(self.grid[0])
        for rows in xrange(len(self.grid[0])):  # pygame inherently flips rows and cols to allow humans to have a sense of x and y
            for cols in xrange(len(self.grid)):
                pygame.draw.rect(self.screen, self.get_color(self.grid[len(self.grid) - 1 - cols][rows]),
                                 [(rows) * self.pix_per_grid, cols * self.pix_per_grid, self.pix_per_grid,
                                  self.pix_per_grid])
        pygame.draw.rect(self.screen, self.agent_color, [current_state[0] * self.pix_per_grid,
                                                         (len(self.grid) - 1 - current_state[1]) * self.pix_per_grid,
                                                         self.pix_per_grid, self.pix_per_grid])

        if world == 'object_world':

            self.color_mode = 'linear'
            for o in mdp_gw.objects:
                current_state = o['pose']
                self.objects[current_state] = []
                color = self.rgb[o['outer']]
                settings = [current_state[0] * self.pix_per_grid,(len(self.grid) - 1 - current_state[1]) * self.pix_per_grid,self.pix_per_grid, self.pix_per_grid]
                pygame.draw.rect(self.screen, color, settings)
                self.objects[current_state].append([color,settings])

                color = self.rgb[o['inner']]
                settings = [current_state[0] * self.pix_per_grid + self.pix_per_grid / 4,(len(self.grid) - 1 - current_state[1]) * self.pix_per_grid + self.pix_per_grid / 4,self.pix_per_grid / 2, self.pix_per_grid / 2]
                pygame.draw.rect(self.screen, color, settings)
                self.objects[current_state].append([color, settings])

        self.refresh()

    def render_move(self, s, graphics=True, delay=0):
        '''
        Requires render_mdp to be called first.
        '''
        if len(self.current_state) == 0:
            print 'Warning: Please call render_mdp before render_move'
            self.refresh(graphics, delay)
            return

        # print [self.g.macro_cell(len(self.grid)-1-self.current_state[1], self.current_state[0])]
        if graphics:

            ps = self.current_state
            if ps in self.objects:
                for color,settings in self.objects[ps]:
                    pygame.draw.rect(self.screen, color, settings)
            else:
                pygame.draw.rect(self.screen, self.get_color(self.grid[ps[1]][ps[0]]),
                                 [ps[0] * self.pix_per_grid, (len(self.grid) - 1 - ps[1]) * self.pix_per_grid,
                                  self.pix_per_grid,
                                  self.pix_per_grid])  # here we flip y-sim because the display has positive y in down direction

            pygame.draw.rect(self.screen, self.agent_color,
                             [s[0] * self.pix_per_grid, (len(self.grid) - 1 - s[1]) * self.pix_per_grid,
                              self.pix_per_grid,
                              self.pix_per_grid])  # here we flip y-sim because the display has positive y in down direction
            self.current_state = s
        self.refresh(graphics, delay)

    def record_demonstration(self):
        self.driving_status = 'Ready'
        clock = pygame.time.Clock()
        while 1:
            if self.check_event():
                return self.recording
            clock.tick(10)

    def hold_gui(self):
        clock = pygame.time.Clock()
        while True:
            self.check_event()
            clock.tick(10)
