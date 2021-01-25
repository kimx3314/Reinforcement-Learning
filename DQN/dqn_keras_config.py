# sean sungil kim


class Config:
    def __init__(self):
        self.GAMMA = 0.95
        self.LEARNING_RATE = 0.001
        self.MEMORY_SIZE = 100000
        self.BATCH_SIZE = 64
        self.EXPLORATION_RATE = 1.0
        self.EXPLORATION_MIN = 0.01
        self.EXPLORATION_DECAY = 0.995
