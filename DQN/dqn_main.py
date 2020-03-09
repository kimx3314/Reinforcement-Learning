# sean sungil kim

import numpy as np
import gym
from dqn_networks import networks


BATCH_SIZE = 32
EPISODES = 4000


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = networks(state_size, action_size)

    for episode in range(EPISODES):
        # for each episode, reset the environment
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        score = 0
        while True:
            #env.render()

            # the longer it stays in the while loop, bigger the score (for CartPole env only)
            score += 1
            
            # t
            action = dqn.act(state)

            # t + 1
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # if done, reward is -10
            reward = reward if not done else -10

            # store in the replay experience queue and go to the next state
            dqn.remember(state, action, reward, next_state, done)
            state = next_state
            
            # if the episode is finished, update the target_model and go to the next episode
            if done:
                dqn.update_target_model()
                print("Episode: %i / %i,\tScore: %i,\tExploration Rate: %.4f" % (episode + 1, EPISODES, score, dqn.epsilon))
                break

            # if there are enough data in the replay experience queue, fit the NN using past experience
            if len(dqn.memory) > BATCH_SIZE:
                dqn.train(BATCH_SIZE)
                #print("Episode: %i / %i,\tScore: %i,\t\tLoss: %.4f" % (episode + 1, EPISODES, score, loss))
