# sean sungil kim

import tensorflow as tf
import gym
from dqn_tf_networks import DQNAgent
from dqn_tf_config import Config


# set the config class
config = Config()

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    
    # define the state and the action space size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    step = 1

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
            # retrieve the action from the dqn model
            action = dqn.act(state)

            # input the action to the environment, and obtain the following
            next_state, reward, done, _ = env.step(action)
            step += 1
            
            # if the episode is finished, reward is -10
            reward = -10 if done else reward
            
            # store in the replay experience queue and go to the next state
            dqn.remember(state, action, reward, next_state, done)
            
            # t + 1
            # go to the next state
            state = next_state
            score += reward

            # if the episode is finished, go to the next episode
            if done:
                print("Episode: %i / %i,\tScore: %i,\tExploration Rate: %.4f" % (episode + 1, config.EPISODES, score, config.EPSILON))
                break

            # if there are enough instances in the replay experience queue, fit the NN using past experience
            if len(dqn.memory) > config.BATCH_SIZE:
                dqn.train()

            # update the target_model every N steps
            if step % config.TARGET_UPDATE_STEP == 0:
                dqn.update_target_model()
