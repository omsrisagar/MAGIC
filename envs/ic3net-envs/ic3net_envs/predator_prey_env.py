#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a predator prey environment.
Each agent can just observe itself (it's own identity) i.e. s_j = j and vision sqaure around it.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
         -1 out of bound,
         indexing for predator agent (from 2?)
         ??? for prey agent (1 for fixed case, for now)
    - Action Space & Observation Space are according to an agent
    - Rewards -0.05 at each time step till the time
    - Episode never ends
    - Obs. State: Vocab of 1-hot < predator, preys & units >
"""

# core modules
import random
import math
import curses

# 3rd party modules
import gym
import numpy as np
from gym import spaces


class PredatorPreyEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.OUTSIDE_CLASS = 0
        self.PREY_CLASS = 1
        self.PREDATOR_SENSE_CLASS = 2
        self.PREDATOR_CAPTURE_CLASS = 3
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0
        self.POS_PREY_REWARD = 0.05
        self.episode_over = False

    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        self.stdscr.clear()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)
        curses.init_pair(5, curses.COLOR_BLUE, -1)
        curses.init_pair(6, curses.COLOR_BLACK, -1)


    def init_args(self, parser):
        env = parser.add_argument_group('Prey Predator task')
        env.add_argument('--nenemies', type=int, default=1,
                         help="Total number of preys in play")
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of box")
        env.add_argument('--vision', type=int, default=2,
                         help="Vision of sense predator")
        env.add_argument('--moving_prey', action="store_true", default=False,
                         help="Whether prey is fixed or moving")
        env.add_argument('--no_stay', action="store_true", default=False,
                         help="Whether predators have an action to stay in place")
        parser.add_argument('--mode', default='mixed', type=str,
                        help='cooperative|competitive|mixed (default: mixed)')
        env.add_argument('--enemy_comm', action="store_true", default=False,
                         help="Whether prey can communicate.")

    def multi_agent_init(self, args):

        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'moving_prey', 'mode', 'enemy_comm']
        for key in params:
            setattr(self, key, getattr(args, key))

        self.nprey = args.nenemies
        self.n_sense_predator = args.n_sense_agents
        self.n_capture_predator = args.n_capture_agents
        self.npredator = self.n_sense_predator + self.n_capture_predator
        self.dims = dims = (self.dim, self.dim)
        self.stay = not args.no_stay

        if args.moving_prey:
            raise NotImplementedError
            # TODO

        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        # Define what an agent can do -
        if self.stay:
            self.naction = 5
        else:
            self.naction = 4

        self.action_space = spaces.MultiDiscrete([self.naction])

        self.BASE = (dims[0] * dims[1])
        self.OUTSIDE_CLASS += self.BASE # 100
        self.PREY_CLASS += self.BASE # 101
        self.PREDATOR_SENSE_CLASS += self.BASE # 102
        self.PREDATOR_CAPTURE_CLASS += self.BASE # 103

        # Setting max vocab size for 1-hot encoding
        self.vocab_size = 1 + 1 + self.BASE + 1 + 1 # 2 predators + grid + prey + outside

        # Observation for each agent will be vision * vision ndarray
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1), dtype=int)
        # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size

        return

    def step(self, action):
        """
        The agents take a step in the environment.

        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :

            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")
        action = np.array(action).squeeze()
        action = np.atleast_1d(action)

        for i, a in enumerate(action):
            self._take_action(i, a) # update self.predator_loc based on the taken action

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."


        self.episode_over = False
        self.obs = self._get_obs()

        debug = {'predator_locs':self.predator_loc,'prey_locs':self.prey_loc}
        return self.obs, self._get_reward(), self.episode_over, debug

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.reached_prey = np.zeros(self.n_capture_predator)
        self.preys_sensed = np.zeros((self.n_sense_predator, self.nprey)) # whether sense agent sensed a prey
        self.preys_captured = np.zeros((self.n_capture_predator, self.nprey)) # whether capture agent captured a prey

        # Locations
        locs = self._get_cordinates() # randomly allocate positions for predators and preys
        self.predator_loc, self.prey_loc = locs[:self.npredator], locs[self.npredator:]

        self._set_grid() # generate an initialized padded grid with onehot encoding (104 dim) for each cell 12x12x104 # self.grid = 12x12; self.empty_bool_base_grid is above.

        # stat - like success ratio
        self.stat = dict()

        # Observation will be npredator * vision * vision ndarray
        self.obs = self._get_obs() # 5 x 3 x 3 x 104 5 predators, 3x3 visible area; each cell 104 onehot encoding

        return self.obs

    def seed(self):
        return

    def _get_cordinates(self):
        idx = np.random.choice(np.prod(self.dims),(self.npredator + self.nprey), replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Mark agents in grid
        # self.grid[self.predator_loc[:,0], self.predator_loc[:,1]] = self.predator_ids
        # self.grid[self.prey_loc[:,0], self.prey_loc[:,1]] = self.prey_ids

        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)
        self.grid_len = self.grid.shape[0]

        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

    def in_slice(self, n, s, length):
        return n in range(*s.indices(length))

    def find_preys_in_obs(self, pred_indx, slice_x, slice_y):
        for i, p in enumerate(self.prey_loc):
            if self.preys_sensed[pred_indx, i]:
                continue
            else:
                in_y = self.in_slice(p[0] + self.vision, slice_y, self.grid_len)
                in_x = self.in_slice(p[1] + self.vision, slice_x, self.grid_len)
                self.preys_sensed[pred_indx, i] = in_x and in_y

    def _get_obs(self):
        self.bool_base_grid = self.empty_bool_base_grid.copy() # always uses the same empty initial grid and then updates it with predator and prey's positions

        for i, p in enumerate(self.predator_loc[:self.n_sense_predator]):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_SENSE_CLASS] += 1

        for i, p in enumerate(self.predator_loc[self.n_sense_predator:]):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CAPTURE_CLASS] += 1

        for i, p in enumerate(self.prey_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] += 1

        obs = []
        for i, p in enumerate(self.predator_loc[:self.n_sense_predator]): # sense predators have visiblity as per vision parameter; this code is not correct for vision==0
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1) # note that p is in orig grid (10x10); while slice_y, slice_x are in the full grid (12x12) with vision=
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            self.find_preys_in_obs(i, slice_x, slice_y)
            obs.append(self.bool_base_grid[slice_y, slice_x])

        for p in self.predator_loc[self.n_sense_predator:]: # capture predators have no visibility
            slice_y = slice(p[0] + self.vision, p[0] + self.vision + 1) # exact p[0] location considering vision padding
            slice_x = slice(p[1] + self.vision, p[1] + self.vision + 1)
            ego_obs = self.bool_base_grid[slice_y, slice_x] # (1, 1, 104)
            mod_obs = np.zeros((2 * self.vision + 1, 2 * self.vision + 1, self.vocab_size), dtype=np.int64)
            center_indx = (2 * self.vision + 1) // 2
            mod_obs[center_indx, center_indx, :] = np.squeeze(ego_obs)
            obs.append(mod_obs)

        if self.enemy_comm:
            for p in self.prey_loc:
                slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
                slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
                obs.append(self.bool_base_grid[slice_y, slice_x])

        obs = np.stack(obs)
        return obs

    def _take_action(self, idx, act):
        # prey action
        if idx >= self.npredator:
            # fixed prey
            if not self.moving_prey:
                return
            else:
                raise NotImplementedError

        if idx >= self.n_sense_predator and idx < self.npredator: # capture agent
            idx_cap = idx - self.n_sense_predator
            if self.reached_prey[idx_cap] == 1: # capture agent has reached the prey
                return

        # STAY action
        if act==4: # isn't this supposed to 4?
            return

        # UP
        if act==0 and self.grid[max(0,
                                    self.predator_loc[idx][0] + self.vision - 1),
                                self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = max(0, self.predator_loc[idx][0]-1)

        # right
        elif act==1 and self.grid[self.predator_loc[idx][0] + self.vision,
                                  min(self.dims[1] -1,
                                      self.predator_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = min(self.dims[1]-1,
                                       self.predator_loc[idx][1]+1)

        # down
        elif act==2 and self.grid[min(self.dims[0]-1,
                                      self.predator_loc[idx][0] + self.vision + 1),
                                  self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = min(self.dims[0]-1,
                                       self.predator_loc[idx][0]+1)

        # left
        elif act==3 and self.grid[self.predator_loc[idx][0] + self.vision,
                                  max(0,
                                      self.predator_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = max(0, self.predator_loc[idx][1]-1)

    def _get_reward(self):
        n = self.npredator if not self.enemy_comm else self.npredator + self.nprey
        reward = np.full(n, self.TIMESTEP_PENALTY)

        # on_prey = np.full((self.n_capture_predator, ), False)
        for i, p in enumerate(self.prey_loc):
            self.preys_captured[:, i] = np.all(self.predator_loc[self.n_sense_predator:] == p, axis=1)

        nb_predator_on_prey = self.preys_captured.sum()
        self.reached_prey = np.any(self.preys_captured, axis=1)
        if self.mode == 'cooperative':
            for i in range(self.n_sense_predator):
                sensed = self.preys_sensed[i, :].astype(bool) # this agent must have sensed it
                # sensed = np.any(self.preys_sensed, axis=0) # any sense agent could have sensed
                captured = np.any(self.preys_captured, axis=0) # any capture agent could have captured it
                sensed_and_captured = (sensed & captured).sum()
                if sensed_and_captured:
                    reward[i] = sensed_and_captured * self.POS_PREY_REWARD
            for i in range(self.n_capture_predator):
                captured = self.preys_captured[i, :].astype(bool) # whether this agent captured a prey
                # num_captured = np.sum(self.preys_captured, axis=0) # number of capture agents who captured a prey
                sensed = np.any(self.preys_sensed, axis=0) # any sense agent could have sensed it
                captured_and_sensed = (captured & sensed).sum()
                captured_and_not_sensed = (captured & ~sensed).sum()
                # captured_and_sensed = (captured * num_captured * sensed).sum()
                # captured_and_not_sensed = (captured * num_captured * ~sensed).sum()
                if captured_and_sensed or captured_and_not_sensed:
                    reward[i+self.n_sense_predator] = captured_and_sensed * self.POS_PREY_REWARD + captured_and_not_sensed * 0.5 * self.POS_PREY_REWARD

        # on_prey = np.where(np.all(self.predator_loc[self.n_sense_predator:] == self.prey_loc, axis=1))[0]
        # nb_predator_on_prey = on_prey.size
        #
        # self.reached_prey[on_prey] = 1
        # on_prey += self.n_sense_predator
        #
        # if self.mode == 'cooperative':
        #     reward[on_prey] = self.POS_PREY_REWARD * nb_predator_on_prey # whoever was on prey will each get pos_prey_reward * on_prey
        #     if on_prey.any():
        #         reward[:self.n_sense_predator] = self.POS_PREY_REWARD * nb_predator_on_prey # sense nodes receive same reward
        # elif self.mode == 'competitive':
        #     if nb_predator_on_prey:
        #         reward[on_prey] = self.POS_PREY_REWARD / nb_predator_on_prey # whoever was on prey will each get a part of pos_prey_reward
        #         if on_prey.any():
        #             reward[:self.n_sense_predator] = self.POS_PREY_REWARD / nb_predator_on_prey
        # elif self.mode == 'mixed':
        #     reward[on_prey] = self.PREY_REWARD # whoever was on prey will each get prey_reward
        #     if on_prey.any():
        #         reward[:self.n_sense_predator] = self.PREY_REWARD
        # else:
        #     raise RuntimeError("incorrect mode, available modes: [cooperative|competitive|mixed]")

        # if np.all(self.reached_prey == 1) and self.mode == 'mixed': # why only in mixed mode this is true?
        if np.all(self.reached_prey == 1):
                self.episode_over = True

        # # Prey reward
        # if nb_predator_on_prey == 0:
        #     reward[self.npredator:] = -1 * self.TIMESTEP_PENALTY
        # else:
        #     # TODO: discuss & finalise
        #     reward[self.npredator:] = 0

        # Success ratio
        if self.mode != 'competitive':
            if nb_predator_on_prey == self.n_capture_predator:
                self.stat['success'] = 1
            else:
                self.stat['success'] = 0

        return reward

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())


    def _onehot_initialization(self, a):
        ncols = self.vocab_size
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def render(self, mode='human', close=False):
        grid = np.zeros(self.BASE, dtype=object).reshape(self.dims)
        self.stdscr.erase()

        for p in self.predator_loc[:self.n_sense_predator]:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'S'
            else:
                grid[p[0]][p[1]] = 'S'

        for p in self.predator_loc[self.n_sense_predator:]:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'C'
            else:
                grid[p[0]][p[1]] = 'C'

        for p in self.prey_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'

        try:
            for row_num, row in enumerate(grid):
                for idx, item in enumerate(row):
                    if item != 0:
                        if 'C' in item and 'P' in item:
                            self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(6))
                        elif 'C' in item:
                            self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                        elif 'P' in item:
                            self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(5))
                        else:
                            self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, '0'.center(3), curses.color_pair(4))

            self.stdscr.addstr(len(grid), 0, '\n')
            self.stdscr.refresh()
        except:
            curses.nocbreak()
            self.stdscr.keypad(0)
            curses.echo()
            curses.endwin()
            raise

    def exit_render(self):
        # curses.nocbreak()
        # self.stdscr.keypad(0)
        # curses.echo()
        # curses.endwin()
        pass
