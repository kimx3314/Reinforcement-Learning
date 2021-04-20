# sean sungil kim

import gym
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from q_learning_q_networks import Q_Networks_Agent


def visualize_return(return_lst):
    plt.figure(figsize=(12, 4))
    plt.plot(return_lst, linewidth=0.6, label="Return")
    plt.xlabel('$Episodes$'), plt.ylabel('$Return$')
    plt.title('$Return$ $vs.$ $Episodes$'), plt.tight_layout()
    plt.savefig('./RESULTS/return_vs_episodes.png')

def FrozenLake(config):
    tf.reset_default_graph()

    env = gym.make('FrozenLake-v0')
    state_size = env.observation_space.n
    action_size = env.action_space.n

    with tf.Session() as sess:
        q_networks = Q_Networks_Agent(config, sess, state_size, action_size)

        return_lst = []
        for episode in range(config['EPISODES']):
            total_return = 0

            # for each episode, reset the environment
            state = env.reset()

            while True:
                # rendering the environment
                if config['RENDER']:
                    env.render()

                # t
                # obtain a greedy action from the q-table with slight noise
                action = q_networks.act(env, state)

                # input the action to the environment, and obtain the following
                next_state, reward, done, _ = env.step(action)

                # update the q-table with the (state, action, reward, next_state) pair
                q_networks.learn(state, action, reward, next_state)
                
                # t + 1
                # go to the next state
                state = next_state
                total_return += reward

                if done:
                    print("Episode: %i / %i,\tReturn: %i" % (episode, config['EPISODES'], total_return))
                    break

            return_lst.append(total_return)
        visualize_return(return_lst)


if __name__ == "__main__":
    with open(os.getcwd()+'/config.json') as f:
        config = json.load(f)
    
    FrozenLake(config)
