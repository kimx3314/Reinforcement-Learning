# sean sungil kim

import tensorflow as tf
import numpy as np
from collections import deque
import random


class DDPG(object):
    def __init__(self, sess, config, state_size, action_size, action_bound):
        # initialize the parameters and the model
        self.config = config
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.memory = deque(maxlen = self.config.MEMORY_CAPACITY)

        # placeholders for inputs x (state size)
        self.states = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), 'State')
        self.next_states = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), 'Next_state')

        # placeholders for rewards and actions
        self.rewards = tf.compat.v1.placeholder(tf.float32, (None, ), 'Rewards')
        self.actions = tf.compat.v1.placeholder(tf.float32, (None,), 'Actions')

        # custom getter
        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        # actions from actor
        self.actor = self.build_actor('Actor', self.states)

        # q-values from critic
        pred = self.build_critic('Critic', self.states, self.actor)

        # actor and critic parameters
        actor_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope = 'Actor')
        critic_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope = 'Critic')

        # maximize the q
        self.actor_loss = -tf.reduce_mean(pred, name = 'Actor_loss')
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.config.ACTOR_LR, name = 'Actor_Adam_opt').minimize(self.actor_loss, var_list = actor_params)

        # soft update using ema
        # every timestep, the target network is 99% of the original target network weights and only 1% of the regular network weights
        # this soft update slowly mixes the regular network weights into the target network weights
        ema = tf.train.ExponentialMovingAverage(decay = 1 - self.config.TAU)

        # initialize the ema values on the actor and critic parameters
        target_update = [ema.apply(actor_params), ema.apply(critic_params)]

        # make sure to have ema initialized
        with tf.control_dependencies(target_update):
            # ema (soft update) is applied on target network (actor and critic) parameters
            actor_target = self.build_actor('Actor_target', self.next_states, reuse = True, trainable = False, custom_getter = ema_getter)
            q_target = self.build_critic('Critic_target', self.next_states, actor_target, reuse = True, trainable = False, custom_getter = ema_getter)

            # the Bellman equation
            y_hat = self.rewards + self.config.GAMMA * q_target

            # MSE loss
            critic_loss = tf.reduce_mean(tf.square(pred - tf.stop_gradient(y_hat)), name = 'Critic_loss')
            self.critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.config.CRITIC_LR, name = 'Critic_Adam_opt').minimize(critic_loss, var_list = critic_params)

        # initialize variables
        self.sess.run(tf.compat.v1.global_variables_initializer())
        writer = tf.summary.FileWriter('./graphs', sess.graph)

    def build_actor(self, name, states, reuse = None, trainable = True, custom_getter = None):
        with tf.compat.v1.variable_scope('Actor', reuse = reuse, custom_getter = custom_getter):
            weights = {'a_w1' : tf.compat.v1.get_variable('Actor_w1', dtype = tf.float32, shape = (self.state_size, 32), initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = trainable),
                    'a_wOut' : tf.compat.v1.get_variable('Actor_wOut', dtype = tf.float32, shape = (32, self.action_size), initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = trainable)}
            biases = {'a_b1' : tf.compat.v1.get_variable('Actor_b1', dtype = tf.float32, initializer = tf.constant(0., shape = (32, ), dtype = tf.float32), trainable = trainable),
                    'a_bOut' : tf.compat.v1.get_variable('Actor_bOut', dtype = tf.float32, initializer = tf.constant(0., shape = (self.action_size, ), dtype = tf.float32), trainable = trainable)}

            # two fully-connected hidden layers
            fc_1 = tf.nn.relu(tf.add(tf.matmul(states, weights['a_w1']), biases['a_b1']))

            # output layer, action value
            out = tf.add(tf.matmul(fc_1, weights['a_wOut']), biases['a_bOut'], name = name)

            return out

    def build_critic(self, name, states, actions, reuse = None, trainable = True, custom_getter = None):
        with tf.compat.v1.variable_scope('Critic', reuse = reuse, custom_getter = custom_getter):
            weights = {'c_w1_s' : tf.compat.v1.get_variable('Critic_w1_s', dtype = tf.float32, shape = (self.state_size, 32), initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = trainable),
                    'c_w1_a' : tf.compat.v1.get_variable('Critic_w1_a', dtype = tf.float32, shape = (self.action_size, 32), initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = trainable),
                    'c_wOut' : tf.compat.v1.get_variable('Critic_wOut', dtype = tf.float32, shape = (32, self.action_size), initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = trainable)}
            biases = {'c_b1' : tf.compat.v1.get_variable('Critic_b1', dtype = tf.float32, initializer = tf.constant(0., shape = (32, ), dtype = tf.float32), trainable = trainable),
                    'c_bOut' : tf.compat.v1.get_variable('Critic_bOut', dtype = tf.float32, initializer = tf.constant(0., shape = (self.action_size, ), dtype = tf.float32), trainable = trainable)}

            # two fully-connected hidden layers
            fc_1 = tf.nn.relu(tf.add(tf.add(tf.matmul(states, weights['c_w1_s']), tf.matmul(actions, weights['c_w1_a'])), biases['c_b1']))

            # output layer, q-values
            out = tf.add(tf.matmul(fc_1, weights['c_wOut']), biases['c_bOut'], name = name)

            return out

    def act(self, state):
        # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a noisy action
        if np.random.rand() <= self.config.EPSILON:
            # add randomness to action using normal distribution, exploration
            return np.clip(np.random.normal(self.sess.run(self.actor, {self.states : state[np.newaxis, :]})[0], self.config.STAND_DEV), -2, 2)

        else:
            # return the action with the highest rewards, exploitation
            return self.sess.run(self.actor, {self.states: state[np.newaxis, :]})[0]

    def remember(self, state, action, reward, next_state):
        # store in the replay experience queue
        self.memory.append((state, action, reward, next_state))

    def train(self):
        field_names = ['state', 'action', 'reward', 'next_state']
        batch_data = {}

        # randomly sample from the replay experience que
        replay_batch = random.sample(self.memory, self.config.BATCH_SIZE)
        for i in range(len(field_names)):
            batch_data[field_names[i]] = [data for data in list(zip(*replay_batch))[i]]

        self.sess.run(self.actor_optimizer, {self.states: batch_data['state']})
        self.sess.run(self.critic_optimizer, {self.states: batch_data['state'],
                                              self.actor: batch_data['action'],
                                              self.rewards: batch_data['reward'],
                                              self.next_states: batch_data['next_state']})

        # decrease the standard deviation value
        if self.config.STAND_DEV > self.config.MIN_STAND_DEV:
            self.config.STAND_DEV *=  self.config.EPSILON_DECAY

        # decrease the epsilon value
        if self.config.EPSILON > self.config.MIN_EPSILON:
            self.config.EPSILON *= self.config.EPSILON_DECAY

    def update_target_model(self):
        # obtain all the variables in the target networks
        self.actor_target_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor_target')
        self.critic_target_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic_target')
        
        # obtain all the variables in the primary networks
        self.actor_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor')
        self.critic_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic')

        # run the soft-update process
        self.sess.run([var_t.assign(self.config.TAU * var + (1.0 - self.config.TAU) * var_t) for var_t, var in zip(self.actor_target_vars, self.actor_vars)])
        self.sess.run([var_t.assign(self.config.TAU * var + (1.0 - self.config.TAU) * var_t) for var_t, var in zip(self.critic_target_vars, self.critic_vars)])

    def load(self):
        # load saved model
        self.imported_graph = tf.train.import_meta_graph('model.meta')
        self.imported_graph.restore(self.sess, './model')

    def save(self):
        # save model weights
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, './model')
