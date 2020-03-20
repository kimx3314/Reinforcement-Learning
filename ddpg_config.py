# sean sungil kim

class Config(object):
    def __init__(self):
        # parameters
        self.BATCH_SIZE = 32
        self.EPISODES = 2000
        self.MAX_EP_STEPS = 2000
        self.ACTOR_LR = 0.001                 # actor learning rate
        self.CRITIC_LR = 0.002                # critic learning rate
        self.GAMMA = 0.9                      # discount rate for the bellman equation
        self.TAU = 0.01                       # soft target update
        self.EPSILON = 1.0                    # exploration rate, starting value set at 100% chance of exploring
        self.MIN_EPSILON = 0.01               # minimum exploration rate
        self.EPSILON_DECAY = 0.999            # the rate at which the exploration rate decays
        self.MIN_STAND_DEV = 0.1              # minimum standard deviation value
        self.STAND_DEV = 3                    # standard deviation, when adding noise to the action output
        self.MEMORY_CAPACITY = 10000          # maximum experience queue size
        self.RENDER = True                    # render toggle