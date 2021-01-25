# sean sungil kim

import numpy as np
from tensorflow import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import keras.backend as K
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size, action_size, config):
        # initialize the parameters and the model
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = self.config.MEMORY_CAPACITY)

        self.target_model = self.build_model()
        self.model = self.build_model()

    def build_model(self):
        # building the NN model
        model = Sequential()

        # input shape should be the state size
        model.add(Dense(32, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(32, activation = 'relu'))

        # output shape should be the action size
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = self.custom_loss, optimizer = Adam(lr = self.config.LR))

        return model

    def custom_loss(self, yTrue, yPred):
        return K.mean(K.square(yTrue - np.amax(yPred)))

    def act(self, state):
        #reshape according to the input shape
        state = state.reshape(-1, self.state_size)

        # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a random action
        if np.random.rand() <= self.config.EPSILON:
            return random.randrange(self.action_size)
        
        # return the action with the highest rewards, exploitation
        else:
            return np.argmax(self.model.predict(state))

    def remember(self, state, action, reward, next_state, done):
        # store in the replay experience queue
        self.memory.append((state, action, reward, next_state, done))

    def bellman(self, batch_data, q_values):
        # return the bellman total return
        # Q(s,a) = r + γ(max(Q(s’,a’))
        q_values = np.array(batch_data['reward']) - np.array(1 - np.array(batch_data['done']) * self.config.GAMMA * np.amax(self.target_model.predict(np.array(batch_data['next_state'])), axis = 1))

        return q_values

    def train(self):
        field_names = ['state', 'action', 'reward', 'next_state', 'done']
        batch_data = {}

        # randomly sample from the replay experience que
        replay_batch = random.sample(self.memory, self.config.BATCH_SIZE)
        for i in range(len(field_names)):
            batch_data[field_names[i]] = [data for data in list(zip(*replay_batch))[i]]
        batch_data.update({'done' : [int(bl) for bl in batch_data['done']]})

        # calculate the q-values in the current state and update the total return
        q_values = self.model.predict(np.array(batch_data['state']))
        q_values = self.bellman(batch_data, q_values)

        # fit the model using the replay experience with updated q_values
        self.model.fit(np.array(batch_data['state']), q_values, epochs = 10, verbose = 0)
        #history = self.model.fit(state, q_values, epochs = 1, verbose = 0)
        #loss = history.history['loss'][0]
        #self.model_loss.append(loss)

        # if the exploration rate is greater than the set minimum, apply the decay rate
        if self.config.EPSILON > self.config.MIN_EPSILON:
            self.config.EPSILON *= self.config.EPSILON_DECAY

        #return loss

    def update_target_model(self):
        # update weights of the target_model (with weights of the primary model)
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        # load saved model
        self.model.load_weights(name)

    def save(self, name):
        # save model weights
        self.model.save_weights(name)
