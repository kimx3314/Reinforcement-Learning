# sean sungil kim

import numpy as np


class Q_Table_Agent:
    def __init__(self, config, state_size, action_size):
        # initializing the q-table
        self.config = config
        self.action_size = action_size
        self.Q = np.zeros([state_size, action_size])

    def act(self, state, episode):
        # add noise from the standard normal distribution
        return np.argmax(self.Q[state, :] + np.random.randn(1, self.action_size) * (1 / (episode + 1)))

    def learn(self, state, action, reward, next_state):
        # return the bellman total return
        # Q(s,a) = r + γ(max(Q(s’,a’))
        self.Q[state, action] = reward + (self.config['GAMMA'] * np.max(self.Q[next_state, :]))
