import h5py
import numpy as np
import torch
import os
# import random
# import math
import h5py
from itertools import *

# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.


class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self, alpha, epsilon, episode_count):
        # Initialize training parameters
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # exploration rate
        self.episode = 0    # current episode
        self.episode_count = episode_count  # nr. of episodes

        self.Q_PATH = 'data/qtables/*'
        self.R_PATH = 'data/rewards/*'

    def fn_init(self, gameboard):
        self.gameboard = gameboard
        self.tiles = len(self.gameboard.tiles)
        self.reward_tots = np.zeros(self.episode_count)

        self.Na = 9*self.gameboard.N_col  # nr. of actions
        self.Ns = np.power(2, self.gameboard.N_col*self.gameboard.N_row)*self.tiles  # nr. of states

        # clear directories from old files
        os.system('rm -rf ' + self.Q_PATH)
        os.system('rm -rf ' + self.R_PATH)

        self.q_file = h5py.File(f'data/qtables/q.hdf5', 'x')  # create h5py-file for qtable
        self.q_table = self.q_file.create_dataset('q_table', (self.Ns, self.Na), dtype=np.dtype(float))  # dataset from qfile

        # maps of state id:s -> index and action index -> action
        self.state_id_to_idx = self.state_id_to_idx()
        self.idx_to_action = self.idx_to_action()

    # load strategy file
    def fn_load_strategy(self, strategy_file):
        self.q_file = h5py.File(strategy_file, 'r')
        self.q_table = self.q_file['q_table']

    # encode states, returns a unique id for a given state matrix + tile
    @staticmethod
    def get_state_id(state_matrix, tile_id):
        state_flat = np.append(state_matrix.flatten(), np.ones(tile_id))
        state_id = 0

        for i, elem in enumerate(state_flat):
            if elem > 0:
                state_id += 2**i
        return state_id

    # returns map with state id as key and a corresponding index in the q-table as value
    def state_id_to_idx(self):
        N_row = self.gameboard.N_row
        N_col = self.gameboard.N_col

        all_perms = np.array([[list(i[x:x + N_row]) for x in range(0, len(i), N_row)]
                             for i in product([1, -1], repeat=N_row*N_col)])
        states, row, col = np.shape(all_perms)
        state_lst = np.zeros(self.tiles*states)

        idx = 0
        for i in range(self.tiles):
            for j in range(states):
                state_lst[idx] = self.get_state_id(all_perms[j, :, :], i)
                idx += 1
        return dict(zip(state_lst, np.arange(0, states*self.tiles)))

    # returns map with action index as key and (tile, rot, col) as corresponding value
    def idx_to_action(self):
        tiles = self.gameboard.tiles

        action_map = {}
        idx = 0
        for i in range(len(tiles)):
            tmp = tiles[i]
            for j in range(len(tmp)):
                for k in range(self.gameboard.N_col):
                    action_map[idx] = (i, j, k)
                    idx += 1
        return action_map

    # compute state id from a matrix+tile set
    def fn_read_state(self):
        self.state_id = self.get_state_id(self.gameboard.board[:, :], self.gameboard.cur_tile_type)

    # select action using epsilon-greedy
    def fn_select_action(self):
        action_valid = False
        while not action_valid:
            state_idx = self.state_id_to_idx.get(self.state_id)

            if np.random.random() < self.epsilon:
                self.action_idx = np.random.randint(0, self.Na)
            else:
                max_q = np.nanmax(self.q_table[state_idx, :])
                self.action_idx = np.random.choice(np.nonzero(self.q_table[state_idx, :] == max_q)[0])

            self.action = self.idx_to_action.get(self.action_idx)
            action_valid = self.gameboard.fn_move(self.action[2], self.action[1]) == 0

            if not action_valid:
                self.q_table[state_idx, self.action_idx] = np.NaN

    # update qtable based on chosen action
    def fn_reinforce(self, old_state_idx, reward):
        state_idx = self.state_id_to_idx.get(self.state_id)
        self.q_table[old_state_idx, self.action_idx] += self.alpha * (reward + np.nanmax(self.q_table[state_idx, :]) -
                                                                      self.q_table[old_state_idx, self.action_idx])

    # run
    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1

            if self.episode % 1000 == 0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ', str(np.sum(
                    self.reward_tots[range(self.episode-100, self.episode)])), ')')

            if self.episode % 2000 == 0:
                saveEpisodes = np.arange(200000, step=1)
                # saveEpisodes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

                if self.episode in saveEpisodes:
                    with open('data/rewards/r.csv', 'a') as f:
                        f.write(f'{self.episode},{self.reward_tots[self.episode-1]}\n')

            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            old_state_idx = np.copy(self.state_id_to_idx.get(self.state_id))

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored
            # as attributes in self)
            self.fn_reinforce(old_state_idx, reward)


class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self, alpha, epsilon, epsilon_scale, replay_buffer_size, batch_size, sync_target_episode_count,
                 episode_count):
        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_scale = epsilon_scale
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.sync_target_episode_count = sync_target_episode_count
        self.episode = 0
        self.episode_count = episode_count
        self.gameboard = None

    def fn_init(self, gameboard):
        self.gameboard = gameboard

        # TODO
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks
        # (one for calculating actions and one target network), experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

    def fn_load_strategy(self, strategy_file):
        pass
        # TODO
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        pass
        # TODO
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is
        # occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board
        # (integer between 0 and len(self.gameboard.tiles))

    def fn_select_action(self):
        pass
        # TODO
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if
        # epsilon greedy
        # This function should not return a value, store the action as an attribute of self and execute the action by
        # moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to
        # epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile
        # (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self, batch):
        pass
        # TODO
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a)
        # , i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state
        # (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1
            if self.episode % 100 == 0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ', str(np.sum(
                    self.reward_tots[range(self.episode-100, self.episode)])), ')')
            if self.episode % 1000 == 0:
                saveEpisodes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
                if self.episode in saveEpisodes:
                    pass
                    # TODO
                    # Here you can save the rewards and the Q-network to data files
            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and \
                        ((self.episode % self.sync_target_episode_count) == 0):
                    pass
                    # TODO
                    # Here you should write line(s) to copy the current network to the target network
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TODO
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored
            # in the experience replay buffer

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()

            # TODO
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so
            # you can save it to disk later

            # Read the new state
            self.fn_read_state()

            # TODO
            # Here you should write line(s) to store the state in the experience replay buffer

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TODO
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                self.fn_reinforce(batch)


class THumanAgent:
    def __init__(self):
        self.episode = None
        self.reward_tots = None
        self.gameboard = None

    def fn_init(self, gameboard):
        self.episode = 0
        self.reward_tots = [0]
        self.gameboard = gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self, pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots = [0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x, (self.gameboard.tile_orientation+1) %
                                               len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1, self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1, self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode] += self.gameboard.fn_drop()
