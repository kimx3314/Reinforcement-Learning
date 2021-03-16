# sean sungil kim

class Config(object):
    def __init__(self):
        # parameters
        self.BATCH_SIZE = 32
        self.EPISODES = 300
        self.ACTOR_LR = 0.001                 # actor learning rate
        self.CRITIC_LR = 0.002                # critic learning rate
        self.GAMMA = 0.9                      # discount rate for the bellman equation
        self.TAU = 0.01                       # soft target update
        self.EPSILON = 1.0                    # exploration rate, starting value set at 100% chance of exploring
        self.EPSILON_DECAY = 0.99             # the rate at which the exploration rate decays
        self.STAND_DEV = 2.0                  # standard deviation, when adding noise to the action output
        self.COUNTER = 0                      # replay buffer counter
        self.MEMORY_CAPACITY = 8000           # maximum experience queue size
        self.RENDER = True                    # render toggle
