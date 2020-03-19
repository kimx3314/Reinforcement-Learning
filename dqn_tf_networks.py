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

        # placeholders for inputs x (state size) and outputs y (action size)
        self.x = tf.compat.v1.placeholder(tf.float32, shape = (None, self.state_size), name = 'X')
        self.y = tf.compat.v1.placeholder(tf.int32, shape = (None, self.action_size), name = 'Y')

        # placeholders for actions, rewards, done_flags (used during training)
        self.next_states = tf.compat.v1.placeholder(tf.float32, shape = (self.config.BATCH_SIZE, self.state_size), name = 'Next_state')
        self.actions = tf.compat.v1.placeholder(tf.int32, shape = (self.config.BATCH_SIZE, ), name = 'Actions')
        self.rewards = tf.compat.v1.placeholder(tf.float32, shape = (self.config.BATCH_SIZE, ), name = 'Rewards')
        self.done_flags = tf.compat.v1.placeholder(tf.float32, shape = (self.config.BATCH_SIZE, ), name = 'Done_flags')

        self.target_model = self.build_target_model()
        self.model = self.build_model()

        # the Bellman equation
        action_one_hot = tf.one_hot(self.actions, self.action_size, 1.0, 0.0, name = 'Action_one_hot')
        pred = tf.reduce_sum(self.model * action_one_hot, reduction_indices = -1, name = 'q_acted')
        y_hat = self.rewards + (1. - self.done_flags) * self.config.GAMMA * tf.reduce_max(self.target_model, axis = -1)
        self.model_loss = tf.reduce_mean(tf.square(pred - tf.stop_gradient(y_hat)), name = 'Loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.config.LR, name = 'Adam_Opt').minimize(self.model_loss)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

    def build_target_model(self):
        
        weights = {'t_w_1' : tf.compat.v1.get_variable('T_W_1', dtype = tf.float32, shape = (self.state_size, 32), initializer = tf.truncated_normal_initializer(stddev = 0.01)),
                   't_w_2' : tf.compat.v1.get_variable('T_W_2', dtype = tf.float32, shape = (32, 32), initializer = tf.truncated_normal_initializer(stddev = 0.01)),
                   't_w_out' : tf.compat.v1.get_variable('T_W_out', dtype = tf.float32, shape = (32, self.action_size), initializer = tf.truncated_normal_initializer(stddev = 0.01))}
        biases = {'t_b_1' : tf.compat.v1.get_variable('T_b_1', dtype = tf.float32, initializer = tf.constant(0., shape = (32, ), dtype = tf.float32)),
                  't_b_2' : tf.compat.v1.get_variable('T_b_2', dtype = tf.float32, initializer = tf.constant(0., shape = (32, ), dtype = tf.float32)),
                  't_b_out' : tf.compat.v1.get_variable('T_b_out', dtype = tf.float32, initializer = tf.constant(0., shape = (self.action_size, ), dtype = tf.float32))}

        # two fully-connected hidden layers
        fc_1 = tf.nn.relu(tf.add(tf.matmul(self.x, weights['t_w_1']), biases['t_b_1']))
        fc_2 = tf.nn.relu(tf.add(tf.matmul(fc_1, weights['t_w_2']), biases['t_b_2']))

        # output layer
        out = tf.add(tf.matmul(fc_2, weights['t_w_out']), biases['t_b_out'], name = 'Target')

        return out

    def build_model(self):
        
        weights = {'w_1' : tf.compat.v1.get_variable('W_1', dtype = tf.float32, shape = (self.state_size, 32), initializer = tf.truncated_normal_initializer(stddev = 0.01)),
                   'w_2' : tf.compat.v1.get_variable('W_2', dtype = tf.float32, shape = (32, 32), initializer = tf.truncated_normal_initializer(stddev = 0.01)),
                   'w_out' : tf.compat.v1.get_variable('W_out', dtype = tf.float32, shape = (32, self.action_size), initializer = tf.truncated_normal_initializer(stddev = 0.01))}
        biases = {'b_1' : tf.compat.v1.get_variable('b_1', dtype = tf.float32, initializer = tf.constant(0., shape = (32, ), dtype = tf.float32)),
                  'b_2' : tf.compat.v1.get_variable('b_2', dtype = tf.float32, initializer = tf.constant(0., shape = (32, ), dtype = tf.float32)),
                  'b_out' : tf.compat.v1.get_variable('b_out', dtype = tf.float32, initializer = tf.constant(0., shape = (self.action_size, ), dtype = tf.float32))}

        # two fully-connected hidden layers
        fc_1 = tf.nn.relu(tf.add(tf.matmul(self.x, weights['w_1']), biases['b_1']))
        fc_2 = tf.nn.relu(tf.add(tf.matmul(fc_1, weights['w_2']), biases['b_2']))

        # output layer
        out = tf.add(tf.matmul(fc_2, weights['w_out']), biases['b_out'], name = 'Primary')

        return out

    def act(self, state):
        # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a random action
        if np.random.rand() <= self.config.EPSILON:
            return random.randrange(self.action_size)
        
        # return the action with the highest rewards, exploitation
        else:
            return self.sess.run(tf.argmax(self.model, 1), {self.x : np.reshape(state, [1, self.state_size])})[0]

    def remember(self, state, action, reward, next_state, done):
        # store in the replay experience queue
        self.memory.append((state, action, reward, next_state, done))

    def bellman(self, cur_reward, next_state):
        # return the bellman total return (t+1)
        return cur_reward + (self.config.GAMMA * np.amax(self.sess.run(self.target_model, {self.x : next_state})))

    def train(self):

        field_names = ['state', 'action', 'reward', 'next_state', 'done']
        batch_data = {}

        # randomly sample from the replay experience que
        # [(array([[-0.02105868,  0.04257037,  0.04981664,  0.0417123 ]]), 0, 1.0, array([[-0.02020727, -0.15322923,  0.05065088,  0.34968738]]), False)]
        replay_batch = random.sample(self.memory, self.config.BATCH_SIZE)
        for i in range(len(field_names)):
            batch_data[field_names[i]] = [data for data in list(zip(*replay_batch))[i]]
        batch_data.update({'done' : [int(bl) for bl in batch_data['done']]})
        #print(np.array(batch_data['state']).shape)

        self.sess.run(self.optimizer, feed_dict = {self.x: batch_data['state'],
                                                   self.actions: batch_data['action'],
                                                   self.rewards: batch_data['reward'],
                                                   self.next_states: batch_data['next_state'],
                                                   self.done_flags: batch_data['done']})

        # if the exploration rate is greater than the set minimum, apply the decay rate
        if self.config.EPSILON > self.config.MIN_EPSILON:
            self.config.EPSILON *= self.config.EPSILON_DECAY

    def update_target_model(self):
        
        # obtain all the variables in the Q target network
        self.target_vars = tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Target')
        
        # obtain all the variables in the Q primary network
        self.primary_vars = tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Primary')

        self.sess.run([var_t.assign(var) for var_t, var in zip(self.target_vars, self.primary_vars)])

    def load(self):

        # load saved model
        self.imported_graph = tf.train.import_meta_graph('model.meta')
        self.imported_graph.restore(self.sess, './model')

    def save(self):

        # save model weights
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, './model')