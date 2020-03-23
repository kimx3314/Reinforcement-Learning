# sean sungil kim

import gym
import tensorflow as tf
from ddpg_networks import DDPG
from ddpg_config import Config


# set the config class
config = Config()

if __name__ == "__main__":
    tf.compat.v1.reset_default_graph()
    env = gym.make('Pendulum-v0')

    # define the state and the action space size
    state_size = env.observation_space.shape[0]         #3
    action_size = env.action_space.shape[0]             #1
    step = 1

    # upper bound of the action space
    action_bound = env.action_space.high

    # ddpg is specifically adapted for environments with continuous action spaces
    sess = tf.compat.v1.Session()
    ddpg = DDPG(sess, config, state_size, action_size, action_bound)

    for episode in range(config.EPISODES):
        # for each episode, reset the environment
        state = env.reset()

        score = 0
        while True:
            if config.RENDER:
                env.render()

            # t
            # retrieve the action from the ddpg model
            action = ddpg.act(state)

            # input the action to the environment, and obtain the following
            next_state, reward, done, _ = env.step(action)
            step += 1

            # store it in the replay experience queue and go to the next state
            ddpg.remember(state, action, reward, next_state)

            # t + 1
            # go to the next state
            state = next_state
            score += reward

            # if the episode is finished, go to the next episode
            if done:
                print("Episode: %i / %i,\tScore: %i,\tStandard Deviation: %.4f, \tExploration Rate: %.4f" % (episode, config.EPISODES, score, config.STAND_DEV, config.EPSILON))
                break

            # if there are enough instances in the replay experience queue, start the training
            if len(ddpg.memory) > config.BATCH_SIZE:
                ddpg.train()
            
            # update the target_model every N steps
            if step % config.TARGET_UPDATE_STEP == 0:
                ddpg.update_target_model()
