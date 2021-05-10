# sean sungil kim

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import sys
import warnings
warnings.filterwarnings("ignore")


class DDPG(object):
    def __init__(self,config, MODE, state_size, action_size, action_lower_bound, action_upper_bound):
        self.actor_loss_lst        = []
        self.critic_loss_lst       = []
        self.actor_loss_threshold  = None
        self.critic_loss_threshold = None

        # initialize the parameters and the model
        self.config                = config
        self.MODE                  = MODE
        self.iteration             = 0
        self.memory                = deque(maxlen = self.config['MEMORY_CAPACITY'])
        self.state_size            = state_size
        self.action_size           = action_size
        self.nChillers             = self.config["nChillers"] # nAC, nSTC, nLTC
        self.action_lower_bound    = action_lower_bound
        self.action_upper_bound    = action_upper_bound
        self.min_action            = 0.1

        # placeholders for inputs
        self.states                = tf.compat.v1.placeholder(tf.float32, [None, state_size], 'States')
        self.next_states           = tf.compat.v1.placeholder(tf.float32, [None, state_size], 'Next_states')
        self.rewards               = tf.compat.v1.placeholder(tf.float32, [None, 1], 'Rewards')

        with tf.compat.v1.variable_scope('Actor'):
            # actions from actor
            self.actions           = self.build_actor(self.states, scope='primary')
            actions_target         = self.build_actor(self.next_states, scope='target', trainable=False)

        with tf.compat.v1.variable_scope('Critic'):
            # q-values from critic
            self.q                 = self.build_critic(self.states, self.actions, scope='primary')
            q_target               = self.build_critic(self.next_states, actions_target, scope='target', trainable=False)

        # actor and critic parameters
        self.actor_params          = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/primary')
        self.actor_t_params        = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.critic_params         = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/primary')
        self.critic_t_params       = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # the Bellman equation
        # Q(s,a) = r + γ(max(Q(s’,a’))
        q_target_bellman           = self.rewards + self.config['GAMMA'] * q_target

        # maximize the q
        self.actor_loss            = -tf.reduce_mean(self.q, name = 'Actor_loss')
        self.actor_optimizer       = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config['ACTOR_LR'], name = 'Actor_Adam_opt').minimize(self.actor_loss, var_list=self.actor_params)

        # MSE loss, temporal difference error
        self.critic_loss           = tf.compat.v1.losses.mean_squared_error(labels=q_target_bellman, predictions=self.q, scope='Critic_loss')
        self.critic_optimizer      = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config['CRITIC_LR'], name = 'Critic_Adam_opt').minimize(self.critic_loss, var_list=self.critic_params)

        # obtain all the variables in the primary and target networks and soft update (target net replacement)
        self.soft_update_variables = [tf.compat.v1.assign(var_t, (1 - self.config['TAU']) * var_t + self.config['TAU'] * var_e) for var_t, var_e in zip(self.actor_t_params + self.critic_t_params, self.actor_params + self.critic_params)]

        # global variable initialization
        init_op                    = tf.compat.v1.global_variables_initializer()

        # savers
        self.actor_saver           = tf.compat.v1.train.Saver(var_list=self.actor_params)
        self.critic_saver          = tf.compat.v1.train.Saver(var_list=self.critic_params)
        
        # start the session and initialize the variables
        self.sess = tf.compat.v1.Session()
        self.sess.run(init_op)

        if self.MODE == 'test':
            # restore all variables
            self.load()

            # soft-update for the target networks
            self.soft_update()

    def build_actor(self, states, scope, trainable=True):
        with tf.compat.v1.variable_scope(scope):
            if self.config['SIMULATOR_VERSION'] == "ShinSaeGae_v0":
                # fully connected layers
                # temperature related states
                fc_1_1                   = tf.compat.v1.layers.dense(states[:, :4], 32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_1_1', trainable=trainable)
                fc_2_1                   = tf.compat.v1.layers.dense(fc_1_1, 16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_2_1', trainable=trainable)

                # time and cost related states
                fc_1_2                   = tf.compat.v1.layers.dense(states[:, 4:], 32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_1_2', trainable=trainable)
                fc_2_2                   = tf.compat.v1.layers.dense(fc_1_2, 16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_2_2', trainable=trainable)

                fc_3_concat              = tf.concat([fc_2_1, fc_2_2], axis=1, name='fc_3_concat')
                fc_3                     = tf.compat.v1.layers.dense(fc_3_concat, 16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_3', trainable=trainable)

                # define lower and upper bounds
                # tf.clip_by_value(t, clip_value_min, clip_value_max)
                raw_chiller_actions      = tf.clip_by_value(tf.compat.v1.layers.dense(fc_3, 3, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), trainable=trainable), \
                                                       tf.constant([[-self.nChillers[0], 0.0, 0.0]], tf.float32), tf.constant([[self.nChillers[0], self.nChillers[1], self.nChillers[2]]], tf.float32), name='raw_chiller_actions')

                # remove very small actions
                # tf.where(condition, x_true, y_false)
                AC_filtered              = tf.where(tf.concat([tf.abs(raw_chiller_actions[:, :1]) < self.min_action, tf.fill([tf.shape(raw_chiller_actions)[0], 2], False)], axis=1), \
                                                    tf.multiply(raw_chiller_actions, tf.fill(tf.shape(raw_chiller_actions), 0.0)), raw_chiller_actions, name='AC_filtered')
                STC_filtered             = tf.where(tf.concat([tf.fill([tf.shape(AC_filtered)[0], 1], False), AC_filtered[:, 1:2] < self.min_action, tf.fill([tf.shape(AC_filtered)[0], 1], False)], axis=1), \
                                                    tf.multiply(AC_filtered, tf.fill(tf.shape(AC_filtered), 0.0)), AC_filtered, name='STC_filtered')
                filtered_chiller_actions = tf.where(tf.concat([tf.fill([tf.shape(STC_filtered)[0], 2], False), STC_filtered[:, 2:] < self.min_action], axis=1), \
                                                    tf.multiply(STC_filtered, tf.fill(tf.shape(STC_filtered), 0.0)), STC_filtered, name='filtered_chiller_actions')

                # remove heating + cooling actions, element-wise multiplication
                out                      = tf.where(tf.concat([filtered_chiller_actions[:, :1] < 0, filtered_chiller_actions[:, :1] < 0, filtered_chiller_actions[:, :1] < 0], axis=1), \
                                                    tf.multiply(filtered_chiller_actions, tf.concat([tf.fill([tf.shape(filtered_chiller_actions)[0], 1], 1.0), tf.fill([tf.shape(filtered_chiller_actions)[0], 2], 0.0)], axis=1)), filtered_chiller_actions, name='out')

            elif self.config['SIMULATOR_VERSION'] == "Inha_NewBuilding_v0":
                # fully connected layers
                fc_1           = tf.compat.v1.layers.dense(states, 32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_1', trainable=trainable)
                fc_2           = tf.compat.v1.layers.dense(fc_1, 16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_2', trainable=trainable)

                # Q_mc
                fc_3_1         = tf.compat.v1.layers.dense(fc_2, 8, activation=tf.nn.tanh, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_3_1', trainable=trainable)
                Q_mc           = tf.clip_by_value(tf.compat.v1.layers.dense(fc_3_1, 1, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), trainable=trainable), -1, 1, name='Q_mc')

                # damper_action cannot be negative, hence relu activation
                fc_3_2         = tf.compat.v1.layers.dense(fc_2, 8, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_3_2', trainable=trainable)
                dor            = tf.clip_by_value(tf.compat.v1.layers.dense(fc_3_2, 1, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), trainable=trainable), self.action_lower_bound[1], self.action_upper_bound[1], name='damper_action')

                # concatenate to output
                out            = tf.concat([Q_mc, dor], axis=1, name='out')

            return out

    def build_critic(self, states, actions, scope, trainable=True):
        with tf.compat.v1.variable_scope(scope):
            # fully connected layers
            fc_1_1 = tf.compat.v1.layers.dense(states, 32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_1_1', trainable=trainable)
            fc_1_2 = tf.compat.v1.layers.dense(actions, 32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_1_2', trainable=trainable)
            fc_1   = tf.concat([fc_1_1, fc_1_2], axis=1, name='fc_1')

            fc_2   = tf.compat.v1.layers.dense(fc_1, 16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_2', trainable=trainable)

            out    = tf.compat.v1.layers.dense(fc_2, 1, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='out', trainable=trainable)
            
            return out

    def act(self, state):
        if self.MODE == 'test':
            # return the action with the highest rewards, exploitation
            return self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0]

        elif self.MODE == 'train':
            # if the ddpg network did not begin the training phase, return random actions
            if self.config['COUNTER'] < self.config['MEMORY_CAPACITY']:
                if self.config['SIMULATOR_VERSION'] == "ShinSaeGae_v0":
                    random_action = np.array([random.uniform(self.action_lower_bound[0], self.action_upper_bound[0]), \
                                              random.uniform(self.action_lower_bound[1], self.action_upper_bound[1]), \
                                              random.uniform(self.action_lower_bound[2], self.action_upper_bound[2])])
                elif self.config['SIMULATOR_VERSION'] == "Inha_NewBuilding_v0":
                    random_action = np.array([random.uniform(-1, 1), \
                                              random.uniform(self.action_lower_bound[1], self.action_upper_bound[1])])
                
                return random_action

            # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a noisy action
            elif np.random.rand() <= self.config['EPSILON']:
                # add randomness to action using normal distribution, exploration
                if self.config['SIMULATOR_VERSION'] == "ShinSaeGae_v0":
                    noisy_action = np.array([np.clip(np.random.normal(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0, 0], self.config['CHILLER_STAND_DEV']), self.action_lower_bound[0], self.action_upper_bound[0]), \
                                             np.clip(np.random.normal(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0, 1], self.config['CHILLER_STAND_DEV']), self.action_lower_bound[1], self.action_upper_bound[1]), \
                                             np.clip(np.random.normal(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0, 2], self.config['CHILLER_STAND_DEV']), self.action_lower_bound[2], self.action_upper_bound[2])])
                elif self.config['SIMULATOR_VERSION'] == "Inha_NewBuilding_v0":
                    noisy_action = np.array([np.clip(np.random.normal(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0, 0], self.config['CHILLER_STAND_DEV']), -1, 1), \
                                             np.clip(np.random.normal(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0, 1], self.config['DAMP_STAND_DEV']), self.action_lower_bound[1], self.action_upper_bound[1])])

                # decrease the epsilon and the standard deviation value
                self.config['EPSILON']          *= self.config['EPSILON_DECAY']
                self.config['CHILLER_STAND_DEV'] *= self.config['EPSILON_DECAY']
                self.config['DAMP_STAND_DEV']   *= self.config['EPSILON_DECAY']

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
        if self.iteration == 0:
            print('\t\t\t\t\t\t\t\t----------- DDPG TRAINING HAS STARTED -----------\t\t\t\t\t\t\t\t')

        # randomly sample from the replay experience que
        replay_batch = np.array(random.sample(self.memory, self.config['BATCH_SIZE']))

        # obtain the batch data for training
        batch_data_state      = replay_batch[:, :self.state_size]
        batch_data_action     = replay_batch[:, self.state_size: self.state_size + self.action_size]
        batch_data_reward     = replay_batch[:, -self.state_size - 1: -self.state_size]
        batch_data_next_state = replay_batch[:, -self.state_size:]

        # train the actor and the critic
        self.sess.run(self.actor_optimizer, {self.states: batch_data_state})
        self.sess.run(self.critic_optimizer, {self.states: batch_data_state,
                                              self.actions: batch_data_action,
                                              self.rewards: batch_data_reward,
                                              self.next_states: batch_data_next_state})
        
        # document actor and critic loss
        current_actor_loss  = self.sess.run(self.actor_loss, {self.states: batch_data_state})
        current_critic_loss = self.sess.run(self.critic_loss, {self.states: batch_data_state,
                                                               self.actions: batch_data_action,
                                                               self.rewards: batch_data_reward,
                                                               self.next_states: batch_data_next_state})
        self.actor_loss_lst.append(current_actor_loss)
        self.critic_loss_lst.append(current_critic_loss)

        # soft update
        self.soft_update()

        if len(self.actor_loss_lst) == 10000:
            print('\t\t\t\t\t\t\t\t----------- DDPG MIN THRESHOLD SET -----------\t\t\t\t\t\t\t\t')
            self.actor_loss_threshold  = current_actor_loss
            self.critic_loss_threshold = current_critic_loss
            
        elif len(self.actor_loss_lst) > 10000:
            if current_actor_loss <= self.actor_loss_threshold:
                self.save_actor()
                self.actor_loss_threshold = current_actor_loss
                print('\t\t\t\t\t\t\t\t----------- ACTOR MODEL SAVED -----------\t\t\t\t\t\t\t\t')

            if current_critic_loss <= self.critic_loss_threshold:
                self.save_critic()
                self.critic_loss_threshold = current_critic_loss
                print('\t\t\t\t\t\t\t\t----------- CRITIC MODEL SAVED -----------\t\t\t\t\t\t\t\t')

        self.iteration +=1

        return self.actor_loss_threshold, self.critic_loss_threshold

    def save_actor_critic_result(self):
        plt.figure(figsize=(16, 4))
        plt.plot(self.actor_loss_lst, linewidth=0.3)
        plt.xlabel('$Steps$'), plt.ylabel('$Loss$')
        plt.title('$Actor$ $Loss$'), plt.tight_layout()
        plt.savefig('./RESULTS/'+self.MODE.upper()+'/actor_loss.png')

        plt.figure(figsize=(16, 4))
        plt.plot(self.critic_loss_lst, linewidth=0.3)
        plt.xlabel('$Steps$'), plt.ylabel('$Loss$')
        plt.title('$Critic$ $Loss$'), plt.tight_layout()
        plt.savefig('./RESULTS/'+self.MODE.upper()+'/critic_loss.png')

    def soft_update(self):
        # run the soft-update process
        self.sess.run(self.soft_update_variables)

    def save_actor(self):
        # save actor weights
        self.actor_saver.save(self.sess, './GRAPHS/actor')

    def save_critic(self):
        # save critic weights
        self.critic_saver.save(self.sess, './GRAPHS/critic')

    def load(self):
        if self.config['LOAD_CHECK']:
            stdoutOrigin=sys.stdout
            sys.stdout = open("./actor_init.txt", "w")
            print('==============INITIALIZED==============')
            for acttor_param in self.actor_params:
                print(str(acttor_param) +'\n'+ str(acttor_param.eval(session=self.sess)), '\n\n')
            sys.stdout.close()
            sys.stdout=stdoutOrigin

        self.actor_saver = tf.compat.v1.train.import_meta_graph('./GRAPHS/actor.meta')
        self.critic_saver = tf.compat.v1.train.import_meta_graph('./GRAPHS/critic.meta')

        self.actor_saver.restore(self.sess, './GRAPHS/actor')
        self.critic_saver.restore(self.sess, './GRAPHS/critic')

        if self.config['LOAD_CHECK']:
            stdoutOrigin=sys.stdout 
            sys.stdout = open("./actor_load.txt", "w")
            print('==============LOADED==============')
            for acttor_param in self.actor_params:
                print(str(acttor_param) +'\n'+ str(acttor_param.eval(session=self.sess)), '\n\n')
            sys.stdout.close()
            sys.stdout=stdoutOrigin
            