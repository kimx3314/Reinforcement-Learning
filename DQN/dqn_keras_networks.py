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
        self.memory = deque(maxlen=self.config.MEMORY_CAPACITY)

        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        # building the NN model
        model = Sequential()

        # input shape should be the state size
        model.add(Dense(32, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(32, activation = 'relu'))

        # output shape should be the action size
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = 'MSE', optimizer = Adam(lr = self.config.LEARNING_RATE))

        return model

    def act(self, state):
        # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a random action
        if np.random.rand() < self.config.EPSILON:
            # if the exploration rate is greater than the set minimum, apply the decay rate
            if self.config.EPSILON > self.config.EPSILON_MIN:
                self.config.EPSILON *= self.config.EPSILON_DECAY

            return random.randrange(self.action_size)

        # return the action with the highest rewards, exploitation
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        # store in the replay experience queue
        self.memory.append((state, action, reward, next_state, done))

        # add 1 to the counter
        self.config.COUNTER += 1

    def bellman(self, reward, next_state):
        # return the bellman total return
        # Q(s,a) = r + γ(max(Q(s’,a’))
        q_values = reward + (self.config.GAMMA * np.amax(self.target_model.predict(next_state)[0]))
        #q_values = reward + (self.config.GAMMA * np.amax(self.model.predict(next_state)[0]))

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

    def soft_update(self):
        # update weights of the target_model (with weights of the primary model)
        #print(self.model.get_weights())
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        # load saved model
        self.model.load_weights(name)

    def save(self, name):
        # save model weights
        self.model.save_weights(name)
        