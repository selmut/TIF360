import numpy as np
import random
import torch
from collections import deque
from itertools import product
import h5py
import os
from torch import nn
from torch import optim
from copy import deepcopy


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


class QNetwork(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(20, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x


class TDQNAgent:
    # Agent for learning to play tetris using deep Q-learning
    def __init__(self, alpha, epsilon, epsilon_scale, replay_buffer_size, batch_size, sync_target_episode_count,
                 episode_count):
        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilonE = 1
        self.epsilon_scale = epsilon_scale
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.sync_target_episode_count = sync_target_episode_count
        self.episode = 0
        self.episode_count = episode_count

        self.R_PATH = 'data/rewards/*'

    def fn_init(self, gameboard):
        # clear directory from old files
        os.system('rm -rf ' + self.R_PATH)

        self.gameboard = gameboard
        self.reward_tots = np.zeros(self.episode_count)

        self.tile_identifiers = self.fn_get_tile_identifiers()

        self.Na = 4*self.gameboard.N_col  # 4 rotations, N_col placement possibilities

        self.current_qvals = np.zeros(self.Na)
        self.current_tile = self.gameboard.cur_tile_type
        self.current_board = self.gameboard.board
        self.current_state = -1*np.ones((self.gameboard.N_row+1, self.gameboard.N_col))

        self.model = QNetwork(self.Na).float()
        self.target_model = deepcopy(self.model)

        self.criterion = nn.MSELoss()
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.alpha, amsgrad=True)

        self.exp_buffer = deque(maxlen=self.replay_buffer_size)

    def fn_load_strategy(self, strategy_file):
        self.model.load_state_dict(torch.load(strategy_file))

    def fn_get_tile_identifiers(self):
        tile_types = list(range(4))
        out = {}
        for n in range(4):
            tile_identifier = -1*np.ones(self.gameboard.N_col)
            tile_identifier[n] = 1
            out[tile_types[n]] = tile_identifier
        out[-1] = -1*np.ones(self.gameboard.N_col)
        return out

    def fn_read_state(self):
        self.current_board = self.gameboard.board
        self.current_tile = self.gameboard.cur_tile_type

        tile_id = self.tile_identifiers[self.current_tile]

        empty = -1 * np.ones((self.gameboard.N_row+1, self.gameboard.N_col))
        empty[1:, :] = self.current_board
        empty[0, :] = tile_id

        self.current_state = empty

    def fn_update_buffer(self, transition):
        self.exp_buffer.append(transition)

    def get_qvals(self, state):
        self.model.eval()
        return self.model(torch.from_numpy(state.reshape(-1, *state.shape))).detach().numpy()

    def fn_decay_epsilon(self):
        self.epsilonE = np.maximum(self.epsilon, 1-self.episode/self.epsilon_scale)

    def fn_select_action(self):
        rand = np.random.random()
        self.current_qvals = self.get_qvals(self.current_state)
        sorted_qvals_idx = np.argsort(self.current_qvals)

        if rand < self.epsilonE:
            self.action_idx = np.random.randint(0, self.Na)
        else:
            self.action_idx = sorted_qvals_idx[0, -1]

        col = self.action_idx // 4
        rot = self.action_idx % 4

        self.gameboard.fn_move(col, rot)

    def fn_reinforce(self, batch):
        self.model.eval()
        self.target_model.eval()

        first_states = np.array([transition[0] for transition in batch])
        first_qvals_batch = self.model(torch.from_numpy(first_states)).detach().numpy()

        second_states = np.array([transition[3] for transition in batch])
        second_qvals_batch = self.target_model(torch.from_numpy(second_states)).detach().numpy()

        X = []
        y = []

        for i, (first_state, action_idx, reward, second_state, done) in enumerate(batch):
            if not done:
                max_second_qval = np.max(second_qvals_batch[i, :])
                new_qval = reward + max_second_qval
            else:
                new_qval = reward

            first_qvals_transition = np.copy(first_qvals_batch[i])
            first_qvals_transition[action_idx] = new_qval

            X.append(first_state)
            y.append(first_qvals_transition)


        X = np.array(X)
        y = np.array(y)

        self.model.train()
        self.loss = self.criterion(self.model(torch.from_numpy(X)), torch.from_numpy(y))

        self.loss.backward()
        self.model_optimizer.step()
        self.model_optimizer.zero_grad()

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1
            if self.episode % 100 == 0:
                print(f'episode {self.episode}/{self.episode_count} '
                      f'(reward: {np.sum(self.reward_tots[range(self.episode-100, self.episode)])}), '
                      f'epsilon_E: {self.epsilonE:.3f}')
            if self.episode % 1000 == 0:
                saveEpisodes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
                if self.episode in saveEpisodes:
                    torch.save(self.model.state_dict(), f'qnets/model_{self.episode}.pth')
                    torch.save(self.target_model.state_dict(), f'qnets/target_{self.episode}.pth')
                    with open('data/rewards/r.csv', 'a') as f:
                        f.write(f'{self.episode},{self.reward_tots[self.episode-1]}\n')

            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and \
                        ((self.episode % self.sync_target_episode_count) == 0):
                    self.target_model = deepcopy(self.model)

                self.gameboard.fn_restart()
        else:
            self.fn_select_action()

            self.current_state = self.current_state.flatten()

            old_state = np.copy(self.current_state)

            reward = np.copy(self.gameboard.fn_drop())
            self.reward_tots[self.episode] += reward

            self.fn_read_state()

            transition = (old_state, np.copy(self.action_idx), reward, np.copy(self.current_state), np.copy(self.gameboard.gameover))
            self.fn_update_buffer(transition)

            if len(self.exp_buffer) >= self.replay_buffer_size:
                batch = random.sample(self.exp_buffer, self.batch_size)
                self.fn_reinforce(batch)
                self.fn_decay_epsilon()


class THumanAgent:
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
                        self.gameboard.fn_move(self.gameboard.tile_x, (self.gameboard.tile_orientation + 1) % len(
                            self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x - 1, self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x + 1, self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode] += self.gameboard.fn_drop()