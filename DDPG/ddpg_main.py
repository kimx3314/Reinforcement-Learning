# sean sungil kim

import gym
import tensorflow as tf
from ddpg_networks import DDPG
from ddpg_config import Config


# set the config class
config = Config()

episode_rewards = []
if __name__ == "__main__":
    tf.reset_default_graph()
    env = gym.make('Pendulum-v0')

    # define the state and the action space size
    state_size = env.observation_space.shape[0]       #3
    action_size = env.action_space.shape[0]           #1
    
    # upper bound of the action space
    action_bound = env.action_space.high

    # Actor and Critic
    sess = tf.Session()
    ddpg = DDPG(sess, config, state_size, action_size, action_bound)

    for episode in range(config.EPISODES):
        # for each episode, reset the environment
        state = env.reset()

        ep_reward = 0
        while True:
            if config.RENDER:
                env.render()

            # t
            # retrieve the action from the ddpg model
            action = ddpg.act(state)

            # input the action to the environment, and obtain the following
            next_state, reward, done, _ = env.step(action)

            # store it in the replay experience queue and go to the next state
            ddpg.remember(state, action, reward, next_state)

            # t + 1
            # go to the next state
            state = next_state
            ep_reward += reward

            # if the episode is finished, update the target_model and go to the next episode
            if done:
                print("Episode: %i / %i,\tScore: %i,\tStandard Deviation: %.4f, \tExploration Rate: %.4f" % (episode, config.EPISODES, ep_reward, config.STAND_DEV, config.EPSILON))
                episode_rewards.append(ep_reward)
                break

            # if there are enough instances in the replay experience queue, start the training
            if len(ddpg.memory) > config.BATCH_SIZE:
                ddpg.train()
