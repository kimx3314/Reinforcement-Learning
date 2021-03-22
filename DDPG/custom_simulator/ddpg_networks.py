# sean sungil kim

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from collections import deque
import random
import warnings
warnings.filterwarnings("ignore")


if os.path.isdir('./GRAPHS'):
    shutil.rmtree('./GRAPHS')
os.makedirs('./GRAPHS', exist_ok = True)

class DDPG(object):
    def __init__(self, sess, config, state_size, action_size, action_lower_bound, action_upper_bound):
        self.actor_loss_lst = []
        self.critic_loss_lst = []

        # initialize the parameters and the model
        self.config = config
        self.sess = sess
        self.memory = deque(maxlen = self.config['MEMORY_CAPACITY'])
        self.state_size = state_size
        self.action_size = action_size
        action_lower_bound[0] = -1
        action_upper_bound[0] = 1
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound

        # placeholders for inputs
        self.states= tf.compat.v1.placeholder(tf.float32, [None, state_size], 'States')
        self.next_states = tf.compat.v1.placeholder(tf.float32, [None, state_size], 'Next_states')
        self.rewards = tf.compat.v1.placeholder(tf.float32, [None, 1], 'Rewards')

        with tf.compat.v1.variable_scope('Actor'):
            # actions from actor
            self.actions = self.build_actor(self.states, scope='primary')
            actions_target = self.build_actor(self.next_states, scope='target', trainable=False)

        with tf.compat.v1.variable_scope('Critic'):
            # q-values from critic
            #q = self.build_critic(self.states, self.actions, scope='primary')
            self.q = self.build_critic(self.states, self.actions, scope='primary')  # by soh
            q_target = self.build_critic(self.next_states, actions_target, scope='target', trainable=False)

        # actor and critic parameters
        self.actor_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/primary')
        self.actor_t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.critic_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/primary')
        self.critic_t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # the Bellman equation
        # Q(s,a) = r + γ(max(Q(s’,a’))
        q_target_bellman = self.rewards + self.config['GAMMA'] * q_target

        # maximize the q
        #self.actor_loss = -tf.reduce_mean(q, name = 'Actor_loss')
        self.actor_loss = -tf.reduce_mean(self.q, name = 'Actor_loss')  # by soh
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config['ACTOR_LR'], name = 'Actor_Adam_opt').minimize(self.actor_loss, var_list=self.actor_params)

        # MSE loss, temporal difference error
        self.critic_loss = tf.compat.v1.losses.mean_squared_error(labels=q_target_bellman, predictions=self.q, scope='Critic_loss')
        self.critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config['CRITIC_LR'], name = 'Critic_Adam_opt').minimize(self.critic_loss, var_list=self.critic_params)

        # obtain all the variables in the primary and target networks and soft update (target net replacement)
        self.soft_update_variables = [tf.compat.v1.assign(var_t, (1 - self.config['TAU']) * var_t + self.config['TAU'] * var_e) for var_t, var_e in zip(self.actor_t_params + self.critic_t_params, self.actor_params + self.critic_params)]

        # initialize all variables
        self.sess.run(tf.compat.v1.global_variables_initializer())

        writer = tf.compat.v1.summary.FileWriter('./GRAPHS', sess.graph)

    def build_actor(self, states, scope, trainable=True):
        with tf.compat.v1.variable_scope(scope):
            # fully connected layers
            fc_1 = tf.compat.v1.layers.dense(states, 32, activation=tf.nn.relu, name='fc_1', trainable=trainable)
            fc_2 = tf.compat.v1.layers.dense(fc_1, 16, activation=tf.nn.relu, name='fc_2', trainable=trainable)

            # tanh activation puts the output in the range of [-1, 1]
            # e_prod_action is in the range of [-1, 1]
            fc_3_1 = tf.compat.v1.layers.dense(fc_2, 8, activation=tf.nn.relu, name='fc_3_1', trainable=trainable)
            e_prod_action = tf.compat.v1.layers.dense(fc_3_1, 1, activation=tf.nn.tanh, name='e_prod_action', trainable=trainable)

            # damper_action cannot be negative, hence relu activation
            fc_3_2 = tf.compat.v1.layers.dense(fc_2, 8, activation=tf.nn.relu, name='fc_3_2', trainable=trainable)
            damper_action = tf.clip_by_value(tf.compat.v1.layers.dense(fc_3_2, 1, activation=tf.nn.relu, trainable=trainable), self.action_lower_bound[1], self.action_upper_bound[1], name='damper_action')

            # concatenate to output
            out = tf.concat([e_prod_action, damper_action], axis=1, name='out')
            
            return out

            '''
            # weights and biases
            w1 = tf.compat.v1.get_variable('a_w1', (self.state_size, 32), trainable=trainable)
            b1 = tf.compat.v1.get_variable('a_b1', (1, 32), trainable=trainable)

            w2 = tf.compat.v1.get_variable('a_w2', (32, 16), trainable=trainable)
            b2 = tf.compat.v1.get_variable('a_b2', (1, 16), trainable=trainable)

            w3_1 = tf.compat.v1.get_variable('a_w3_1', (16, 1), trainable=trainable)
            b3_1 = tf.compat.v1.get_variable('a_b3_1', (1, 1), trainable=trainable)
            w3_2 = tf.compat.v1.get_variable('a_w3_2', (16, 1), trainable=trainable)
            b3_2 = tf.compat.v1.get_variable('a_b3_2', (1, 1), trainable=trainable)
            
            # fully connected layers
            fc_1 = tf.nn.relu(tf.matmul(states, w1) + b1)
            fc_2 = tf.nn.relu(tf.matmul(fc_1, w2) + b2)

            e_prod_action = tf.nn.tanh(tf.matmul(fc_2, w3_1) + b3_1)
            damper_action = tf.nn.relu(tf.matmul(fc_2, w3_2) + b3_2)

            # concatenate to output
            out = tf.concat([e_prod_action, damper_action], axis=1, name='out')
            
            return out
            '''

    def build_critic(self, states, actions, scope, trainable=True):
        with tf.compat.v1.variable_scope(scope):
            # fully connected layers
            fc_1_1 = tf.compat.v1.layers.dense(states, 32, activation=tf.nn.relu, name='fc_1_1', trainable=trainable)
            fc_1_2 = tf.compat.v1.layers.dense(actions, 32, activation=tf.nn.relu, name='fc_1_2', trainable=trainable)
            fc_1 = tf.concat([fc_1_1, fc_1_2], axis=1, name='fc_1')

            fc_2 = tf.compat.v1.layers.dense(fc_1, 16, activation=tf.nn.relu, name='fc_2', trainable=trainable)

            out = tf.compat.v1.layers.dense(fc_2, 1, name='out', trainable=trainable)
            
            return out

            '''
            # weights and biases
            w1_s = tf.compat.v1.get_variable('c_w1_s', (self.state_size, 32), trainable=trainable)
            w1_a = tf.compat.v1.get_variable('c_w1_a', (self.action_size, 32), trainable=trainable)
            b1 = tf.compat.v1.get_variable('c_b1', (1, 32), trainable=trainable)

            w2 = tf.compat.v1.get_variable('c_w2', (32, 16), trainable=trainable)
            b2 = tf.compat.v1.get_variable('c_b2', (1, 16), trainable=trainable)

            w3 = tf.compat.v1.get_variable('c_w3', (16, 1), trainable=trainable)
            b3 = tf.compat.v1.get_variable('c_b3', (1, 1), trainable=trainable)
            
            # fully connected layers
            fc_1 = tf.nn.relu(tf.matmul(states, w1_s) + tf.matmul(actions, w1_a) + b1)
            fc_2 = tf.nn.relu(tf.matmul(fc_1, w2) + b2)

            # output layer
            out = tf.matmul(fc_2, w3) + b3
            
            return out
            '''

    def act(self, state):
        # if the ddpg network did not begin the training phase, return random actions
        if self.config['COUNTER'] < self.config['MEMORY_CAPACITY']:
            random_action = np.array([random.uniform(self.action_lower_bound[0], self.action_upper_bound[0]), random.uniform(self.action_lower_bound[1], self.action_upper_bound[1])])

            return random_action

        # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a noisy action
        elif np.random.rand() <= self.config['EPSILON']:
            # add randomness to action using normal distribution, exploration
            noisy_action = np.array([np.clip(np.random.normal(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0, 0], self.config['E_PROD_STAND_DEV']), self.action_lower_bound[0], self.action_upper_bound[0]),\
                                     np.clip(np.random.normal(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0, 1], self.config['DAMP_STAND_DEV']),   self.action_lower_bound[1], self.action_upper_bound[1])])
            
            # decrease the epsilon and the standard deviation value
            self.config['EPSILON'] *= self.config['EPSILON_DECAY']
            self.config['E_PROD_STAND_DEV'] *= self.config['EPSILON_DECAY']
            self.config['DAMP_STAND_DEV'] *= self.config['EPSILON_DECAY']

            return noisy_action

        else:
            # return the action with the highest rewards, exploitation
            return self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0]

    def remember(self, state, action, reward, next_state):
        # store in the replay experience queue
        self.memory.append(np.concatenate((state, action, [reward], next_state)))

        # add 1 to the counter
        self.config['COUNTER'] += 1

    def train(self):
        # randomly sample from the replay experience que
        replay_batch = np.array(random.sample(self.memory, self.config['BATCH_SIZE']))
        
        # obtain the batch data for training
        batch_data_state = replay_batch[:, :self.state_size]
        batch_data_action = replay_batch[:, self.state_size: self.state_size + self.action_size]
        batch_data_reward = replay_batch[:, -self.state_size - 1: -self.state_size]
        batch_data_next_state = replay_batch[:, -self.state_size:]

        # train the actor and the critic
        _, current_actions = self.sess.run([self.actor_optimizer, self.actions], {self.states: batch_data_state}) # by soh
        #print("actions={}".format(current_actions))  # by soh

        _, current_q = self.sess.run([self.critic_optimizer, self.q], {self.states: batch_data_state,  # by soh
                                                                       self.actions: batch_data_action,
                                                                       self.rewards: batch_data_reward,
                                                                       self.next_states: batch_data_next_state})
        #print("q={}".format(current_q))  # by soh
        
        # document actor and critic loss
        current_actor_loss = self.sess.run(self.actor_loss, {self.states: batch_data_state})
        current_critic_loss = self.sess.run(self.critic_loss, {self.states: batch_data_state,
                                                               self.actions: batch_data_action,
                                                               self.rewards: batch_data_reward,
                                                               self.next_states: batch_data_next_state})
        self.actor_loss_lst.append(current_actor_loss)
        self.critic_loss_lst.append(current_critic_loss)

        # soft update
        self.soft_update()

    def save_actor_critic_result(self):
        plt.figure(figsize=(16, 4))
        plt.plot(self.actor_loss_lst, linewidth=0.3)
        plt.xlabel('$Steps$'), plt.ylabel('$Loss$')
        plt.title('$Actor$ $Loss$')
        plt.savefig('./RESULTS/actor_loss.png')

        plt.figure(figsize=(16, 4))
        plt.plot(self.critic_loss_lst, linewidth=0.3)
        plt.xlabel('$Steps$'), plt.ylabel('$Loss$')
        #plt.ylim((-3000, 16000))
        plt.title('$Critic$ $Loss$')
        plt.savefig('./RESULTS/critic_loss.png')

    def soft_update(self):
        # run the soft-update process
        self.sess.run(self.soft_update_variables)
        
    def load(self):
        # load saved model
        self.imported_graph = tf.compat.v1.train.import_meta_graph('model.meta')
        self.imported_graph.restore(self.sess, './model')

    def save(self):
        # save model weights
        self.saver = tf.compat.v1.train.Saver()
        self.saver.save(self.sess, './model')
