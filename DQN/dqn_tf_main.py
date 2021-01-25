# sean sungil kim

import tensorflow as tf
import gym
import numpy as np
from dqn_tf_networks import DQNAgent
from dqn_tf_config import Config


# set the config class
config = Config()

def cartpole():
    # environment parameters
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    sess = tf.compat.v1.Session()
    dqn = DQNAgent(sess, state_size, action_size, config)
    
    step = 0
    for episode in range(config.EPISODES):
        total_return = 0

        # for each episode, reset the environment
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        while True:
            # rendering the environment
            if config.RENDER:
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
            step += 1

            # DQN agent training
            if config.COUNTER > config.MEMORY_CAPACITY:
                dqn.train()

                # update the target_model every N steps
                if step % config.TARGET_UPDATE_STEP == 0:
                    dqn.soft_update()

            # if the episode is finished, go to the next episode
            if done:
                print("Episode: %i / %i,\tReturn: %i,\tCounter: %i,\t\tExploration Rate: %.4f" % (episode, config.EPISODES, total_return, config.COUNTER, config.EPSILON))
                break


if __name__ == "__main__":
    cartpole()
