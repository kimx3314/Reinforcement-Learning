"""BOLTZMANN RL

.. moduleauthor:: Sean Sungil Kim <sungilkim3314@gmail.com>

Deals with everything related to the rl agent.

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import sys
import warnings
warnings.filterwarnings("ignore")


class DDPG(object):
    def __init__(self, config, MODE, state_size, action_size, action_lower_bound, action_upper_bound):
        """Initializes all the tensorflow placeholders, empty lists necessary for saving the result, rl parameters, simulator parameters, and the rl algorithm tensorflow graphs.

        Args:
            config (dict): configuration file in JSON format.
            MODE (str): train vs. test phase.
            state_size (int): total number of observational states.
            action_size (int): total number of actions.
            action_lower_bound (list): lowest possible value for each action.
            action_upper_bound (list): highest possible value for each action.

        """
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
        #self.action_lower_bound    = action_lower_bound
        #self.action_upper_bound    = action_upper_bound
        self.action_lower_bound    = [-1.0]
        self.action_upper_bound    = [1.0]

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
        """Defines the nn actor model in tensorflow.

        Args:
            states (arr): the state parameters defined in the Simulator class.
            scope (str): the variable scope name.
            
        Kwargs:
            trainable (bool): The primary networks should be trainable, but the target networks should not be trainable. The target networks' weights are updated via self.soft_update().

        Returns:
            The built/defined actor model.

        """
        with tf.compat.v1.variable_scope(scope):
            if self.config['SIMULATOR_VERSION'] == "ShinSaeGae_v0" and (self.config['SEASON'] == 'all' or self.config['SEASON'] == 'summer-winter'):
                # fully connected layers
                fc_1 = tf.compat.v1.layers.dense(states, 32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_1', trainable=trainable)
                fc_2 = tf.compat.v1.layers.dense(fc_1, 16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_2', trainable=trainable)
                bn = tf.compat.v1.layers.batch_normalization(fc_2, name='bn', trainable=trainable)

                fc_3 = tf.compat.v1.layers.dense(bn, 8, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_3', trainable=trainable)
                out = tf.compat.v1.layers.dense(fc_3, 1, activation=tf.nn.tanh, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='out', trainable=trainable)

                # tf.clip_by_value(t, clip_value_min, clip_value_max)
                #clipped  = tf.clip_by_value(tf.compat.v1.layers.dense(fc_4, 1, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), trainable=trainable), \
                #                            tf.constant([[self.action_lower_bound[0]]], tf.float32), tf.constant([[self.action_upper_bound[0]]], tf.float32), name='clipped')

                # tf.where(condition, x_true, y_false)
                #out = tf.where(tf.math.abs(clipped[0, 0]) < 100, tf.multiply(clipped, tf.fill([1, 1], 0.0)), clipped, name='out')

            elif self.config['SIMULATOR_VERSION'] == "ShinSaeGae_v0" and self.config['SEASON'] == 'summer':
                # fully connected layers
                fc_1 = tf.compat.v1.layers.dense(states, 32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_1', trainable=trainable)
                fc_2 = tf.compat.v1.layers.dense(fc_1, 16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_2', trainable=trainable)
                
                out  = tf.clip_by_value(tf.compat.v1.layers.dense(fc_2, 1, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), trainable=trainable), \
                                        tf.constant([[0]], tf.float32), tf.constant([[self.action_upper_bound[0]]], tf.float32), name='out')

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
        """Defines the nn critic model in tensorflow

        Args:
            states (arr): the state parameters defined in the Simulator class.
            actions (arr): the actions from the actor.
            scope (str): the variable scope name.
            
        Kwargs:
            trainable (bool): The primary networks should be trainable, but the target networks should not be trainable. The target networks' weights are updated via self.soft_update().

        Returns:
            The built/defined critic model.

        """
        with tf.compat.v1.variable_scope(scope):
            # fully connected layers
            fc_1_1 = tf.compat.v1.layers.dense(states, 32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_1_1', trainable=trainable)
            fc_1_2 = tf.compat.v1.layers.dense(actions, 16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_1_2', trainable=trainable)
            fc_1   = tf.concat([fc_1_1, fc_1_2], axis=1, name='fc_1')

            fc_2   = tf.compat.v1.layers.dense(fc_1, 16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='fc_2', trainable=trainable)

            out    = tf.compat.v1.layers.dense(fc_2, 1, kernel_regularizer=tf.keras.regularizers.l2(l=0.01), name='out', trainable=trainable)
            
            return out

    def act(self, state):
        """When the current state is inputted to this function, the current state is fed into the rl agent andthe action is outputted.

        This function takes into consideration the concept of exploration and exploitation.

        Args:
            state (arr): the state parameter defined in the Simulator class.
            
        Returns:
            Either a randomly sampled (exploration) or actor-based (exploitation) actions.

        """
        if self.MODE == 'test':
            # return the action with the highest rewards, exploitation
            return self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0]

        elif self.MODE == 'train':
            # if the ddpg network did not begin the training phase, return random actions
            if self.config['COUNTER'] < self.config['MEMORY_CAPACITY']:
                if self.config['SIMULATOR_VERSION'] == "ShinSaeGae_v0":
                    random_action = np.array([random.uniform(self.action_lower_bound[0], self.action_upper_bound[0])])
                elif self.config['SIMULATOR_VERSION'] == "Inha_NewBuilding_v0":
                    random_action = np.array([random.uniform(-1, 1), \
                                              random.uniform(self.action_lower_bound[1], self.action_upper_bound[1])])
                
                return random_action

            # if the exploration rate is below or equal to the random sample from a uniform distribution over [0, 1), return a noisy action
            elif np.random.rand() <= self.config['EPSILON']:
                # add randomness to action using normal distribution, exploration
                if self.config['SIMULATOR_VERSION'] == "ShinSaeGae_v0":
                    noisy_action = np.array([np.clip(np.random.normal(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0, 0], self.config['STAND_DEV']), self.action_lower_bound[0], self.action_upper_bound[0])])
                elif self.config['SIMULATOR_VERSION'] == "Inha_NewBuilding_v0":
                    noisy_action = np.array([np.clip(np.random.normal(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0, 0], self.config['STAND_DEV']), -1, 1), \
                                             np.clip(np.random.normal(self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0, 1], self.config['STAND_DEV']), self.action_lower_bound[1], self.action_upper_bound[1])])

                # decrease the epsilon and the standard deviation value
                if self.config['EPSILON'] > self.config['MIN_EPSILON']:
                    self.config['EPSILON'] *= self.config['EPSILON_DECAY']
                if self.config['STAND_DEV'] > self.config['MIN_STAND_DEV']:
                    self.config['STAND_DEV'] *= self.config['EPSILON_DECAY']

                return noisy_action

            else:
                # return the action with the highest rewards, exploitation
                return self.sess.run(self.actions, {self.states: state[np.newaxis, :]})[0]

    def remember(self, state, action, reward, next_state):
        """Adds the (state, action, reward, next_state) pairs to the memory buffer.

        Args:
            state (arr): the state parameter defined in the simulator class.
            action (list): the actions defined in the Simulator class.
            reward (list): the reward calculated based on the states, from the simulator.
            next_state (arr): the next timestep state parameters.

        """
        # store in the replay experience queue
        self.memory.append(np.concatenate((state, action, reward, next_state)))

        # add 1 to the counter
        self.config['COUNTER'] += 1

    def train(self):
        """Performs the tranining of the actor and the critic according to the graphs initialized in the__init__() function above. 

        Currently utilizes random sampling from the memory buffer, andsoft update procedure.
        Saves the actor and critic separately according to the corresponding losses.
        
        """
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
        if self.iteration%100 == 0:
            self.soft_update()

        if self.iteration%3000 == 0:
            print('\t\t\t\t\t\t\t\t----------- ACTOR-CRITIC SAVED -----------\t\t\t\t\t\t\t\t')
            self.save_actor()
            self.save_critic()
            #self.actor_loss_threshold  = current_actor_loss
            #self.critic_loss_threshold = current_critic_loss

        '''
        if len(self.actor_loss_lst) == 5000:
            print('\t\t\t\t\t\t\t\t----------- DDPG MIN THRESHOLD SET -----------\t\t\t\t\t\t\t\t')
            self.actor_loss_threshold  = current_actor_loss
            self.critic_loss_threshold = current_critic_loss
            
        elif len(self.actor_loss_lst) > 5000:
            if current_actor_loss <= self.actor_loss_threshold:
                self.save_actor()
                self.actor_loss_threshold = current_actor_loss
                print('\t\t\t\t\t\t\t\t----------- ACTOR MODEL SAVED -----------\t\t\t\t\t\t\t\t')

            if current_critic_loss <= self.critic_loss_threshold:
                self.save_critic()
                self.critic_loss_threshold = current_critic_loss
                print('\t\t\t\t\t\t\t\t----------- CRITIC MODEL SAVED -----------\t\t\t\t\t\t\t\t')
        '''
        self.iteration +=1

        #return self.actor_loss_threshold, self.critic_loss_threshold

    def save_actor_critic_result(self):
        """Saves the loss vs iterations graph for both the actor and the critic.
        
        """
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
        """Performs the soft update procedure defined in the __init__() function above.
        
        """
        # run the soft-update process
        self.sess.run(self.soft_update_variables)

    def save_actor(self):
        """Saves the actor-related graphs.
        
        """
        # save actor weights
        self.actor_saver.save(self.sess, './GRAPHS/actor')

    def save_critic(self):
        """Saves the critic-related graphs.
        
        """
        # save critic weights
        self.critic_saver.save(self.sess, './GRAPHS/critic')

    def load(self):
        """Loads all the actor-critic related graphs.
        
        """
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
