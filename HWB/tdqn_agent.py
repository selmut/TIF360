# old code
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
        self.idx_to_action = self.idx_to_action()

        self.Na = 9*self.gameboard.N_col    # nr. of actions
        self.Ns = np.power(2, self.gameboard.N_col*self.gameboard.N_row)*len(self.gameboard.tiles)    # nr. of states

        self.current_qvals = np.zeros(self.Na)
        self.current_tile = self.gameboard.cur_tile_type
        self.current_board = self.gameboard.board
        self.current_state = -1*np.ones((self.gameboard.N_row+1, self.gameboard.N_col))

        '''using tensorflow
        self.input_shape = (self.gameboard.N_row+1, self.gameboard.N_col, 1)
        self.model = self.fn_create_model()
        self.target_model = self.fn_create_model()
        self.target_model.set_weights(self.model.get_weights())'''

        # Using pytorch
        self.model = Model(self.Na).float()
        self.target_model = deepcopy(self.model)

        self.loss_fct = nn.MSELoss()
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.alpha, amsgrad=True)

        self.exp_buffer = deque(maxlen=self.replay_buffer_size)

    def fn_load_strategy(self, strategy_file):
        pass
        # TODO
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_get_tile_identifiers(self):
        tile_types = list(range(4))
        out = {}
        for n in range(4):
            tile_identifier = -1*np.ones(self.gameboard.N_col)
            tile_identifier[n] = 1
            out[tile_types[n]] = tile_identifier
        out[-1] = -1*np.ones(self.gameboard.N_col)
        return out

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

    def fn_read_state(self):
        self.current_board = self.gameboard.board
        self.current_tile = self.gameboard.cur_tile_type

        tile_id = self.tile_identifiers[self.current_tile]

        empty = -1 * np.ones((self.gameboard.N_row+1, self.gameboard.N_col))
        empty[1:, :] = self.current_board
        empty[0, :] = tile_id

        self.current_state = empty

    # using tensorflow
    def fn_create_model(self):
        model = Sequential()
        '''model.add(Conv2D(64, (3, 3), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.2))'''

        '''model.add(Conv2D(64, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.2))'''

        model.add(Input(shape=self.input_shape))
        model.add(Flatten())

        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(self.Na))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha), metrics=['accuracy'])

        return model

    def fn_update_buffer(self, transition):
        self.exp_buffer.append(transition)

    def get_qvals(self, state):
        self.model.eval()
        return self.model(torch.from_numpy(state.reshape(-1, *state.shape))).detach().numpy()
    # return self.model.predict(state, verbose=0)[0]  # using tensorflow

    def fn_decay_epsilon(self):
        self.epsilonE = np.maximum(self.epsilon, 1-self.episode/self.epsilon_scale)

    def fn_select_action(self):
        action_valid = False
        while not action_valid:
            rand = np.random.random()
            self.current_qvals = self.get_qvals(self.current_state)

            if rand < self.epsilonE:
                self.action_idx = np.random.randint(0, self.Na)
            else:
                max_q = np.max(self.current_qvals)
                self.action_idx = np.random.choice(np.where(self.current_qvals == max_q)[0])

            self.action = self.idx_to_action.get(self.action_idx)
            action_valid = self.gameboard.fn_move(self.action[2], self.action[1]) == 0

            '''if not action_valid:
                if self.episode > 2500:
                    print(self.current_qvals)
                self.current_qvals[0, self.action_idx] = np.nan'''

    def fn_reinforce(self, batch):
        self.model.eval()
        self.target_model.eval()
        current_states = np.array([transition[0] for transition in batch])
        current_qvals_arr = self.model(torch.from_numpy(current_states))
        # current_qvals_arr = self.model.predict(current_states, verbose=0)  # using tf

        new_current_states = np.array([transition[3] for transition in batch])
        future_qvals_arr = self.target_model(torch.from_numpy(new_current_states))
        # future_qvals_arr = self.target_model.predict(new_current_states, verbose=0)  # using tf

        X = []
        y = []

        current_qvals_arr = current_qvals_arr.detach().numpy()
        future_qvals_arr = future_qvals_arr.detach().numpy()

        for i, (current_state, action_idx, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.nanmax(future_qvals_arr[i])
                new_q = reward + max_future_q
            else:
                new_q = reward

            current_qvals = current_qvals_arr[i]
            current_qvals[action_idx] = new_q

            X.append(current_state)
            y.append(current_qvals)

        # using torch
        X = np.array(X)
        y = np.array(y)

        self.model.train()
        self.model_optimizer.zero_grad()
        self.loss = self.loss_fct(self.model(torch.from_numpy(X)), torch.from_numpy(y))
        self.loss.backward()
        self.model_optimizer.step()

        # self.model.fit(np.array(X), np.array(y), batch_size=self.batch_size, verbose=0)  # using tf

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
                    self.reward_tots[range(self.episode-100, self.episode)])), ')', f', epsilon_E: {self.epsilonE}')
            if self.episode % 10 == 0:
                saveEpisodes = np.arange(self.episode_count+1, step=1)
                # saveEpisodes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
                if self.episode in saveEpisodes:
                    # Here you can save the rewards and the Q-network to data files
                    # self.model.save(f'data/qnetworks/qmodel__{self.episode}.model')
                    with open('data/rewards/r.csv', 'a') as f:
                        f.write(f'{self.episode},{self.reward_tots[self.episode-1]}\n')

            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and \
                        ((self.episode % self.sync_target_episode_count) == 0):
                    self.target_model = deepcopy(self.model)
                    # Here you should write line(s) to copy the current network to the target network
                    # self.target_model.set_weights(self.model.get_weights())  # using tf
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()

            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored
            # in the experience replay buffer
            old_state = np.copy(self.current_state)

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            # Here you should write line(s) to store the state in the experience replay buffer
            transition = (old_state, self.action_idx, reward, self.current_state, self.gameboard.gameover)
            self.fn_update_buffer(transition)

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets
                batch = random.sample(self.exp_buffer, self.batch_size)
                self.fn_reinforce(batch)
                self.fn_decay_epsilon()
                # K.clear_session()  # using tf
                gc.collect()