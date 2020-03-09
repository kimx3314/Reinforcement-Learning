# sean sungil kim

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random


class Networks:
    def __init__(self, state_size, action_size):
        # initialize the parameters and the model
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 2000)
        self.gamma = 0.95                     # discount rate for the bellman equation
        self.epsilon = 1.0                    # exploration rate, starting value set at 100% chance of exploring
        self.min_epsilon = 0.001              # minimum exploration rate
        self.epsilon_decay = 0.999            # the rate at which the exploration rate decays
        self.learning_rate = 0.001            # learning rate for the NN optimizer

        self.target_model = self.build_model()
        self.model = self.build_model()
        self.model_loss = []

    def build_model(self):
        # building the NN model
        model = Sequential()

        # input shape should be the state size
        model.add(Dense(32, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(32, activation = 'relu'))

        # output shape should be the action size
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = 'mae', optimizer = Adam(lr = self.learning_rate))

        return model

    def act(self, state):
        # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a random action 
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # return the action with the highest rewards, exploitation
        else:
            return np.argmax(self.model.predict(state))

    def remember(self, state, action, reward, next_state, done):
        # store in the replay experience queue
        self.memory.append((state, action, reward, next_state, done))

    def bellman(self, cur_reward, next_state):
        # return the bellman total return
        return cur_reward + (self.gamma * np.amax(self.target_model.predict(next_state)))

    def train(self, batch_size):
        # randomly sample from the replay experience que
        replay_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in replay_batch:
            # if the episode is finished, total return is given. otherwise, calculate the bellman total return using the target_model
            if done:
                total_return = reward
            else:
                total_return = self.bellman(reward, next_state)

            # calculate the q-values in the current state and update the total return
            q_values = self.model.predict(state)
            q_values[0][action] = total_return

            # fit the model using the replay experience with updated q_values
            history = self.model.fit(state, q_values, epochs = 1, verbose = 0)
            loss = history.history['loss'][0]
            self.model_loss.append(loss)

            # if the exploration rate is greater than the set minimum, apply the decay rate
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

            #return loss

    def update_target_model(self):
        # update weights of the target_model (with weights of the model)
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        # load saved model
        self.model.load_weights(name)

    def save(self, name):
        # save model weights
        self.model.save_weights(name)
