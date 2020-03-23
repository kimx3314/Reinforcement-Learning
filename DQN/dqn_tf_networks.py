# sean sungil kim

import numpy as np
import tensorflow as tf
from collections import deque
import random


class DQNAgent:
    def __init__(self, sess, state_size, action_size, config):
        # initialize the parameters and the model
        self.config = config
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = self.config.MEMORY_CAPACITY)

        # placeholders for inputs x (state size)
        self.states = tf.compat.v1.placeholder(tf.float32, shape = (None, self.state_size), name = 'State')
        self.next_states = tf.compat.v1.placeholder(tf.float32, shape = (None, self.state_size), name = 'Next_state')

        # placeholders for actions, rewards, done_flags (used during training)
        self.actions = tf.compat.v1.placeholder(tf.int32, shape = (None, ), name = 'Actions')
        self.rewards = tf.compat.v1.placeholder(tf.float32, shape = (None, ), name = 'Rewards')
        self.done_flags = tf.compat.v1.placeholder(tf.float32, shape = (None, ), name = 'Done_flags')

        # q-values
        self.target_model = self.build_target_model()
        self.model = self.build_model()

        # one-hot encoding
        action_one_hot = tf.one_hot(self.actions, self.action_size, 1.0, 0.0, name = 'Action_one_hot')

        # q-values for the activated actions
        pred = tf.reduce_sum(self.model * action_one_hot, axis = 1, name = 'Pred')
        
        # the Bellman equation
        y_hat = self.rewards + (1. - self.done_flags) * self.config.GAMMA * tf.reduce_max(self.target_model, axis = 1)
        
        # MSE between the primary (activated actions) and target (highest) q-vlaues
        model_loss = tf.reduce_mean(tf.square(pred - tf.stop_gradient(y_hat)), name = 'Loss')
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.config.LR, name = 'Adam_opt').minimize(model_loss)

        # initialize variables
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def build_target_model(self):
        weights = {'t_w1' : tf.compat.v1.get_variable('T_w1', dtype = tf.float32, shape = (self.state_size, 32), initializer = tf.truncated_normal_initializer(stddev = 0.01)),
                   't_w2' : tf.compat.v1.get_variable('T_w2', dtype = tf.float32, shape = (32, 32), initializer = tf.truncated_normal_initializer(stddev = 0.01)),
                   't_wOut' : tf.compat.v1.get_variable('T_wOut', dtype = tf.float32, shape = (32, self.action_size), initializer = tf.truncated_normal_initializer(stddev = 0.01))}
        biases = {'t_b1' : tf.compat.v1.get_variable('T_b1', dtype = tf.float32, initializer = tf.constant(0., shape = (32, ), dtype = tf.float32)),
                  't_b2' : tf.compat.v1.get_variable('T_b2', dtype = tf.float32, initializer = tf.constant(0., shape = (32, ), dtype = tf.float32)),
                  't_bOut' : tf.compat.v1.get_variable('T_bOut', dtype = tf.float32, initializer = tf.constant(0., shape = (self.action_size, ), dtype = tf.float32))}

        # two fully-connected hidden layers
        fc_1 = tf.nn.relu(tf.add(tf.matmul(self.next_states, weights['t_w1']), biases['t_b1']))
        fc_2 = tf.nn.relu(tf.add(tf.matmul(fc_1, weights['t_w2']), biases['t_b2']))

        # output layer, q-values
        out = tf.add(tf.matmul(fc_2, weights['t_wOut']), biases['t_bOut'], name = 'Target')

        return out

    def build_model(self):
        weights = {'w1' : tf.compat.v1.get_variable('w1', dtype = tf.float32, shape = (self.state_size, 32), initializer = tf.truncated_normal_initializer(stddev = 0.01)),
                   'w2' : tf.compat.v1.get_variable('w2', dtype = tf.float32, shape = (32, 32), initializer = tf.truncated_normal_initializer(stddev = 0.01)),
                   'wOut' : tf.compat.v1.get_variable('wOut', dtype = tf.float32, shape = (32, self.action_size), initializer = tf.truncated_normal_initializer(stddev = 0.01))}
        biases = {'b1' : tf.compat.v1.get_variable('b1', dtype = tf.float32, initializer = tf.constant(0., shape = (32, ), dtype = tf.float32)),
                  'b2' : tf.compat.v1.get_variable('b2', dtype = tf.float32, initializer = tf.constant(0., shape = (32, ), dtype = tf.float32)),
                  'bOut' : tf.compat.v1.get_variable('bOut', dtype = tf.float32, initializer = tf.constant(0., shape = (self.action_size, ), dtype = tf.float32))}

        # two fully-connected hidden layers
        fc_1 = tf.nn.relu(tf.add(tf.matmul(self.states, weights['w1']), biases['b1']))
        fc_2 = tf.nn.relu(tf.add(tf.matmul(fc_1, weights['w2']), biases['b2']))

        # output layer, q-values
        out = tf.add(tf.matmul(fc_2, weights['wOut']), biases['bOut'], name = 'Primary')

        return out

    def act(self, state):
        # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a random action
        if np.random.rand() <= self.config.EPSILON:
            return random.randrange(self.action_size)
        
        # return the action with the highest rewards, exploitation
        else:
            return self.sess.run(tf.argmax(self.model, axis = 1), {self.states : np.reshape(state, [1, self.state_size])})[0]

    def remember(self, state, action, reward, next_state, done):
        # store in the replay experience queue
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        field_names = ['state', 'action', 'reward', 'next_state', 'done']
        batch_data = {}
        
        # randomly sample from the replay experience que
        replay_batch = random.sample(self.memory, self.config.BATCH_SIZE)
        for i in range(len(field_names)):
            batch_data[field_names[i]] = [data for data in list(zip(*replay_batch))[i]]
        batch_data.update({'done' : [int(bl) for bl in batch_data['done']]})

        # train the NN
        self.sess.run(self.optimizer, feed_dict = {self.states: batch_data['state'],
                                                   self.actions: batch_data['action'],
                                                   self.rewards: batch_data['reward'],
                                                   self.next_states: batch_data['next_state'],
                                                   self.done_flags: batch_data['done']})

        # if the exploration rate is greater than the set minimum, apply the decay rate
        if self.config.EPSILON > self.config.MIN_EPSILON:
            self.config.EPSILON *= self.config.EPSILON_DECAY

    def update_target_model(self):
        # obtain all the variables in the Q target network
        self.target_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Target')
        
        # obtain all the variables in the Q primary network
        self.primary_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Primary')

        # run the update process
        self.sess.run([var_t.assign(var) for var_t, var in zip(self.target_vars, self.primary_vars)])

    def load(self):
        # load saved model
        self.imported_graph = tf.train.import_meta_graph('model.meta')
        self.imported_graph.restore(self.sess, './model')

    def save(self):
        # save model weights
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, './model')
