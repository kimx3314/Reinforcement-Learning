# sean sungil kim

import tensorflow as tf
import numpy as np


class Q_Networks_Agent:
    def __init__(self, config, sess, state_size, action_size):
        self.config = config
        self.sess = sess

        self.state = tf.placeholder(tf.float32, [1, state_size], 'States')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'Rewards')
        self.next_state = tf.placeholder(tf.float32, [1, state_size], 'Next_States')

        self.q_networks = self.build_q_networks(self.state)
        next_max_Q = np.max(self.build_q_networks(self.next_state))

        # return the bellman total return
        # Q(s,a) = r + γ(max(Q(s’,a’))
        bellman_Qs = self.reward + (self.config['GAMMA'] * next_max_Q)

        loss = tf.reduce_sum(tf.square(self.q_networks - bellman_Qs))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def build_q_networks(self, states):
        # initialize the Q-network weights
        weights = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
        
        # obtain the Q-values
        Qs = tf.matmul(states, weights)

        return Qs

    def act(self, env, state):
        if np.random.rand(1) < 0.1:
            return env.action_space.sample()
        else:
            return self.sess.run(tf.argmax(self.q_networks, 1), feed_dict={self.state : np.identity(16)[state:state+1]})[0]

    def learn(self, state, action, reward, next_state):
        # train the q-networks with the updated Qs with the bellamn equation
        self.sess.run(self.optimizer, feed_dict={self.state : np.identity(16)[state:state+1],
                                                 self.reward : [[reward]],
                                                 self.next_state : np.identity(16)[next_state:next_state+1]})
