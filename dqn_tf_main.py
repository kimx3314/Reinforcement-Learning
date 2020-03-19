# sean sungil kim

import numpy as np
import tensorflow as tf
import gym
from dqn_networks_tf import DQNAgent
from dqn_config_tf import Config


# set the config class
config = Config()

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    sess = tf.compat.v1.Session()
    dqn = DQNAgent(sess, state_size, action_size, config)

    for episode in range(config.EPISODES):
        # for each episode, reset the environment
        state = env.reset()

        score = 0
        while True:
            if config.RENDER:
                env.render()
            
            # t
            action = dqn.act(state)

            # t + 1
            next_state, reward, done, _ = env.step(action)

            # if done, reward is -10
            reward = reward if not done else -10
            score += reward
            
            # store in the replay experience queue and go to the next state
            dqn.remember(state, action, reward, next_state, done)
            state = next_state
            
            # if the episode is finished, update the target_model and go to the next episode
            if done:
                #dqn.update_target_model()
                print("Episode: %i / %i,\tScore: %i,\tExploration Rate: %.4f" % (episode + 1, config.EPISODES, score, config.EPSILON))
                break

            # if there are enough data in the replay experience queue, fit the NN using past experience
            if len(dqn.memory) > config.BATCH_SIZE:
                dqn.train()
