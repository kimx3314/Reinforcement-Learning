# sean sungil kim

class Config:
    def __init__(self):
        self.BATCH_SIZE = 64
        self.EPISODES = 500
        self.TARGET_UPDATE_STEP = 100
        self.GAMMA = 0.95                     # discount rate for the bellman equation
        self.LEARNING_RATE = 0.002            # dqn learning rate
        self.EPSILON = 1.0                    # starting exploration rate
        self.EPSILON_MIN = 0.01               # minimum exploration rate
        self.EPSILON_DECAY = 0.99             # the rate at which the exploration rate decays
        self.MEMORY_CAPACITY = 1000           # maximum experience queue size
        self.COUNTER = 0                      # replay buffer counter
        self.RENDER = True                    # render toggle
        