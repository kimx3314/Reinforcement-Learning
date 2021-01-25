# sean sungil kim

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K


class DQNAgent:
    def __init__(self, config, state_size, action_size):
        # initialize the parameters and the model
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=self.config.MEMORY_SIZE)

        self.model = self.build_model()

    def build_model(self):
        # building the NN model
        model = Sequential()

        # input shape should be the state size
        model.add(Dense(32, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(32, activation = 'relu'))

        # output shape should be the action size
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = self.custom_loss, optimizer = Adam(lr = self.config.LEARNING_RATE))

        return model

    def custom_loss(self, yTrue, yPred):
        # MSE loss
        return K.mean(K.square(yTrue - np.amax(yPred)))

    def act(self, state):
        # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a random action
        if np.random.rand() < self.config.EXPLORATION_RATE:
            return random.randrange(self.action_size)

        # return the action with the highest rewards, exploitation
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        # store in the replay experience queue
        self.memory.append((state, action, reward, next_state, done))

    def bellman(self, reward, next_state):
        # return the bellman total return
        # Q(s,a) = r + γ(max(Q(s’,a’))
        #q_values = reward + (self.config.GAMMA * np.amax(self.target_model.predict(next_state)[0]))
        q_values = reward + (self.config.GAMMA * np.amax(self.model.predict(next_state)[0]))

        return q_values

    def train(self):
        # if there are not enough data in the replay buffer, skip the training
        if len(self.memory) < self.config.BATCH_SIZE:
            return

        # randomly sample from the replay experience que
        batch = random.sample(self.memory, self.config.BATCH_SIZE)

        # bellman equation for the q values
        for state, action, reward, next_state, done in batch:
            q_update = reward
            
            # bellman equation
            if not done:
                q_update = self.bellman(reward, next_state)

            q_values = self.model.predict(state)
            q_values[0][action] = q_update

            # train the model
            self.model.fit(state, q_values, verbose=0)

        # if the exploration rate is greater than the set minimum, apply the decay rate
        if self.config.EXPLORATION_RATE > self.config.EXPLORATION_MIN:
            self.config.EXPLORATION_RATE *= self.config.EXPLORATION_DECAY
