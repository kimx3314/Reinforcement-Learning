# sean sungil kim

import tensorflow as tf
import numpy as np


class Q_Networks_Agent:
    def __init__(self, config, sess, state_size, action_size):
        self.config = config
        self.sess = sess
        
        self.states = tf.placeholder(tf.float32, [1, state_size], 'States')
        self.next_Qs = tf.placeholder(tf.float32, [1, action_size], 'Next_Qs')
        self.q_networks = self.build_q_networks(self.states)

        loss = tf.reduce_sum(tf.square(self.next_Qs - self.q_networks))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def build_q_networks(self, states):
        # initialize the Q-network weights
        weights = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
        
        # obtain the Q-values
        Qs = tf.matmul(states, weights)

        return Qs

    def act(self, state):
        return self.sess.run(tf.argmax(self.q_networks, 1), feed_dict={self.states:np.identity(16)[state:state+1]})[0]

    def learn(self, state, action, reward, next_state):
        next_max_Q = np.max(self.sess.run(self.q_networks, feed_dict={self.states:np.identity(16)[next_state:next_state+1]}))

        # return the bellman total return
        # Q(s,a) = r + γ(max(Q(s’,a’))
        bellman_Qs = self.sess.run(self.q_networks, feed_dict={self.states:np.identity(16)[state:state+1]})
        bellman_Qs[0, action] = reward + (self.config['GAMMA'] * next_max_Q)

        #Train our network using target and predicted Q values
        self.sess.run(self.optimizer, feed_dict={self.states:np.identity(16)[state:state+1], \
                                                 self.next_Qs:bellman_Qs})

