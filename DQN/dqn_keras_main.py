# sean sungil kim

import gym
import numpy as np
from dqn_keras_networks import DQNAgent
from dqn_keras_config import Config

# set the config class
config = Config()

def cartpole():
    # environment parameters
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQNAgent(config, state_size, action_size)
    
    while True:
        total_return = 0

        # for each episode, reset the environment
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        while True:
            # rendering the environment
            env.render()
            
            # t
            # retrieve the action from the dqn model
            action = dqn.act(state)
            
            # input the action to the environment, and obtain the following
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # if the episode is finished, reward is -
            reward = reward if not done else -reward
            total_return += reward

            # store in the replay experience queue and go to the next state
            dqn.remember(state, action, reward, next_state, done)

            # t + 1
            # go to the next state
            state = next_state

            # if the episode is finished, go to the next episode
            if done:
                print("Score: " + str(total_return) + ", Exploration Rate: " + str(round(config.EXPLORATION_RATE, 3)))
                break

            # DQN agent training
            dqn.train()

if __name__ == "__main__":
    cartpole()
