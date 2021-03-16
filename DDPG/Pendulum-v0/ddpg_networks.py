# sean sungil kim

import tensorflow as tf
import numpy as np
from collections import deque
import random


class DDPG(object):
    def __init__(self, sess, config, state_size, action_size, action_lower_bound, action_upper_bound):
        # initialize the parameters and the model
        self.config = config
        self.sess = sess
        self.memory = deque(maxlen = self.config.MEMORY_CAPACITY)
        self.state_size = state_size
        self.action_size = action_size
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
            q = self.build_critic(self.states, self.actions, scope='primary')
            q_target = self.build_critic(self.next_states, actions_target, scope='target', trainable=False)

        # actor and critic parameters
        self.actor_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/primary')
        self.actor_t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.critic_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/primary')
        self.critic_t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # the Bellman equation
        # Q(s,a) = r + γ(max(Q(s’,a’))
        q_target_bellman = self.rewards + self.config.GAMMA * q_target

        # maximize the q
        self.actor_loss = -tf.reduce_mean(q, name = 'Actor_loss')
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config.ACTOR_LR, name = 'Actor_Adam_opt').minimize(self.actor_loss, var_list=self.actor_params)

        # MSE loss, temporal difference error
        self.critic_loss = tf.compat.v1.losses.mean_squared_error(labels=q_target_bellman, predictions=q, scope='Critic_loss')
        self.critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config.CRITIC_LR, name = 'Critic_Adam_opt').minimize(self.critic_loss, var_list=self.critic_params)

        # obtain all the variables in the primary and target networks and soft update (target net replacement)
        self.soft_update_variables = [tf.compat.v1.assign(var_t, (1 - self.config.TAU) * var_t + self.config.TAU * var_e) for var_t, var_e in zip(self.actor_t_params + self.critic_t_params, self.actor_params + self.critic_params)]

        # initialize all variables
        self.sess.run(tf.compat.v1.global_variables_initializer())

        writer = tf.compat.v1.summary.FileWriter('./graphs', sess.graph)

    def build_actor(self, states, scope, trainable=True):
        with tf.compat.v1.variable_scope(scope):
            # fully connected layers
            fc_1 = tf.compat.v1.layers.dense(states, 32, activation=tf.nn.relu, name='fc_1', trainable=trainable)

            # output layer, tanh activation puts the out in the range of (-1, 1)
            # multiplying fc_2 by the action_upper_bound assists in faster convergence
            out = tf.multiply(tf.compat.v1.layers.dense(fc_1, self.action_size, activation=tf.nn.tanh, trainable=trainable), self.action_upper_bound, name='out')
            
            return out

    def build_critic(self, states, actions, scope, trainable=True):
        with tf.compat.v1.variable_scope(scope):
            # weights and biases
            w1_s = tf.compat.v1.get_variable('c_w1_s', (self.state_size, 32), trainable=trainable)
            w1_a = tf.compat.v1.get_variable('c_w1_a', (self.action_size, 32), trainable=trainable)
            b1 = tf.compat.v1.get_variable('c_b1', (1, 32), trainable=trainable)
            
            # fully connected layers
            fc_1 = tf.nn.relu(tf.matmul(states, w1_s) + tf.matmul(actions, w1_a) + b1)

            # output layer
            out = tf.compat.v1.layers.dense(fc_1, 1, name='out', trainable=trainable)
            
            return out

    def act(self, state):
        # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a noisy action
        if np.random.rand() <= self.config.EPSILON:
            # decrease the epsilon and the standard deviation value
            self.config.EPSILON *= self.config.EPSILON_DECAY
            self.config.STAND_DEV *= self.config.EPSILON_DECAY

            # add randomness to action using normal distribution, exploration
            return np.clip(np.random.normal(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0], self.config.STAND_DEV), self.action_lower_bound, self.action_upper_bound)
        else:
            # return the action with the highest rewards, exploitation
            return np.clip(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0], self.action_lower_bound, self.action_upper_bound)

    def remember(self, state, action, reward, next_state):
        # store in the replay experience queue
        self.memory.append(np.concatenate((state, action, [reward], next_state)))

        # add 1 to the counter
        self.config.COUNTER += 1

    def train(self):
        # randomly sample from the replay experience que
        replay_batch = np.array(random.sample(self.memory, self.config.BATCH_SIZE))
        
        # obtain the batch data for training
        batch_data_state = replay_batch[:, :self.state_size]
        batch_data_action = replay_batch[:, self.state_size: self.state_size + self.action_size]
        batch_data_reward = replay_batch[:, -self.state_size - 1: -self.state_size]
        batch_data_next_state = replay_batch[:, -self.state_size:]

        # train the actor and the critic
        self.sess.run(self.actor_optimizer, {self.states: batch_data_state})
        self.sess.run(self.critic_optimizer, {self.states: batch_data_state,
                                              self.actions: batch_data_action,
                                              self.rewards: batch_data_reward,
                                              self.next_states: batch_data_next_state})
        
        # soft update
        self.soft_update()

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
