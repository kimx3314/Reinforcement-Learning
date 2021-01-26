# sean sungil kim

import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K


class DQNAgent:
    def __init__(self, sess, config, state_size, action_size):
        # initialize the parameters and the model
        self.config = config
        self.sess = sess
        self.memory = deque(maxlen=self.config.MEMORY_CAPACITY)
        self.state_size = state_size
        self.action_size = action_size

        # placeholders for inputs
        self.states= tf.compat.v1.placeholder(tf.float32, (None, state_size), 'States')
        self.next_states = tf.compat.v1.placeholder(tf.float32, (None, state_size), 'Next_States')
        self.rewards = tf.compat.v1.placeholder(tf.float32, (None, 1), 'Rewards')
        self.actions = tf.compat.v1.placeholder(tf.int32, (None, 1), 'Actions')
        self.done = tf.compat.v1.placeholder(tf.int32, (None, 1), 'Done')

        with tf.compat.v1.variable_scope('DQN'):
            # q_values from the primary and the target model
            self.q = self.build_model(self.states, scope='Primary')
            self.q_target = self.build_model(self.next_states, scope='Target', trainable=False)

        # the Bellman equation
        # Q(s,a) = r + γ(max(Q(s’,a’))
        q_target_bellman = self.rewards + (1. - self.done) * self.config.GAMMA * tf.reduce_max(self.q_target, axis = 1)

        self.model_loss = tf.compat.v1.losses.mean_squared_error(labels=q_target_bellman, predictions=self.q, scope='Model_loss')
        self.model_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config.LEARNING_RATE, name = 'Model_Adam_opt').minimize(self.model_loss)

        # initialize variables
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def build_model(self, states, scope, trainable=True):
        with tf.compat.v1.variable_scope(scope):
            # fully connected layers
            fc_1 = tf.compat.v1.layers.dense(states, 32, activation=tf.nn.relu, name='fc_1', trainable=trainable)
            fc_2 = tf.compat.v1.layers.dense(fc_1, 32, activation=tf.nn.relu, name='fc_1', trainable=trainable)

            # output layer, softmax activation puts the out in the range of (0, 1)
            out = tf.compat.v1.layers.dense(fc_2, self.action_size, activation=tf.nn.softmax, name='out', trainable=trainable)
            
            return out

    def act(self, state):
        # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a random action
        if np.random.rand() < self.config.EPSILON:
            # if the exploration rate is greater than the set minimum, apply the decay rate
            if self.config.EPSILON > self.config.EPSILON_MIN:
                self.config.EPSILON *= self.config.EPSILON_DECAY

            return random.randrange(self.action_size)

        # return the action with the highest rewards, exploitation
        else:
            return np.argmax(self.sess.run(self.q, {self.states: state})[0])

    def remember(self, state, action, reward, next_state, done):
        # store in the replay experience queue
        self.memory.append((state, action, reward, next_state, done))

        # add 1 to the counter
        self.config.COUNTER += 1

    def train(self):
        # if there are not enough data in the replay buffer, skip the training
        if len(self.memory) < self.config.BATCH_SIZE:
            return

        # randomly sample from the replay experience que
        batch = random.sample(self.memory, self.config.BATCH_SIZE)
        
        # bellman equation for the q values
        for state, action, reward, next_state, done in batch:
            self.sess.run(self.model_optimizer, {self.states: state,
                                                 self.actions: action,
                                                 self.rewards: reward,
                                                 self.next_states: next_state,
                                                 self.done: done})

        # if the exploration rate is greater than the set minimum, apply the decay rate
        if self.config.EPSILON > self.config.MIN_EPSILON:
            self.config.EPSILON *= self.config.EPSILON_DECAY

    def soft_update(self):
    # obtain all the variables in the Q target network
        self.target_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Target')
        
        # obtain all the variables in the Q primary network
        self.primary_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Primary')

        # run the soft-update process
        self.sess.run([var_t.assign(self.config.TAU * var + (1.0 - self.config.TAU) * var_t) for var_t, var in zip(self.target_vars, self.primary_vars)])

    def load(self):
        # load saved model
        self.imported_graph = tf.train.import_meta_graph('model.meta')
        self.imported_graph.restore(self.sess, './model')

    def save(self):
        # save model weights
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, './model')
