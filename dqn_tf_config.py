# sean sungil kim

class Config(object):
    def __init__(self):
        # parameters
        self.BATCH_SIZE = 256
        self.EPISODES = 2000
        self.LR = 0.001                       # learning rate for the NN optimizer
        self.GAMMA = 0.98                     # discount rate for the bellman equation
        self.EPSILON = 1.0                    # exploration rate, starting value set at 100% chance of exploring
        self.MIN_EPSILON = 0.001              # minimum exploration rate
        self.EPSILON_DECAY = 0.999            # the rate at which the exploration rate decays
        self.MEMORY_CAPACITY = 10000          # maximum experience queue size
        self.RENDER = True                    # render toggle