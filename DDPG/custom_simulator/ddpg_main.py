# sean sungil kim

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ddpg_networks import DDPG
from asi_bems import asi_bems
import json
import os
import math
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


def fill_na_days(data, year):
    month_31 = [1, 3, 5, 7, 8, 10, 12]

    for num_month in range(1, 13):
        sub_data = data[data['Month'] == num_month]
        if num_month in month_31:
            for num_day in range(1, 32):
                if num_day not in sub_data['Day'].tolist():
                    data = data.append(pd.DataFrame({'Year':[year], 'Month':[num_month], 'Day':[num_day], 'Return':[np.nan]}))
        elif num_month == 2:
            if year % 4 == 0:
                max_days = 29
            else:
                max_days = 28
            for num_day in range(1, max_days+1):
                if num_day not in sub_data['Day'].tolist():
                    data = data.append(pd.DataFrame({'Year':[year], 'Month':[num_month], 'Day':[num_day], 'Return':[np.nan]}))
        else:
            for num_day in range(1, 31):
                if num_day not in sub_data['Day'].tolist():
                    data = data.append(pd.DataFrame({'Year':[year], 'Month':[num_month], 'Day':[num_day], 'Return':[np.nan]}))

    return data

def conv_to_calamp(data):
    data = np.concatenate((data[0::7], data[1::7], data[2::7], data[3::7], data[4::7], data[5::7], data[6::7]))

    return data

def daily_heatmap(vis_data):
    daily_data_2020 = vis_data[vis_data['Year'] == 2020]
    daily_data_2019 = vis_data[vis_data['Year'] == 2019]
    daily_data_2018 = vis_data[vis_data['Year'] == 2018]
    daily_data_2017 = vis_data[vis_data['Year'] == 2017]
    
    month_label_lst = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    ai_cs_2020 = conv_to_calamp(np.concatenate((np.full((2, ), np.nan), np.array(daily_data_2020['Return']), np.full((3, ), np.nan))))
    ai_cs_2019 = conv_to_calamp(np.concatenate((np.full((1, ), np.nan), np.array(daily_data_2019['Return']), np.full((5, ), np.nan))))
    ai_cs_2018 = conv_to_calamp(np.concatenate((np.array(daily_data_2018['Return']), np.full((6, ), np.nan))))
    ai_cs_2017 = conv_to_calamp(np.concatenate((np.full((6, ), np.nan), np.array(daily_data_2017['Return']))))
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (22, 14), gridspec_kw = {'hspace':0.5})
    ax1 = sns.heatmap(ai_cs_2020.reshape(7, -1), ax = ax1, vmax = 0, cmap = sns.diverging_palette(10, 240, n = 30), cbar = False)
    ax1.set_xticks(np.array([math.floor(val) for val in np.linspace(0, 53, 13)][:-1]))
    ax1.set_xticklabels(month_label_lst, rotation = 90, size = 16), ax1.set_yticklabels(days, rotation = 0, size = 16)
    ax1.set_ylabel('2020', size = 25, labelpad = 20), ax1.set_facecolor('gray')
    ax2 = sns.heatmap(ai_cs_2019.reshape(7, -1), ax = ax2, vmax = 0, cmap = sns.diverging_palette(10, 240, n = 30), cbar = False)
    ax2.set_xticks(np.array([math.floor(val) for val in np.linspace(0, 53, 13)][:-1]))
    ax2.set_xticklabels(month_label_lst, rotation = 90, size = 16), ax2.set_yticklabels(days, rotation = 0, size = 16)
    ax2.set_ylabel('2019', size = 25, labelpad = 20), ax2.set_facecolor('gray')
    ax3 = sns.heatmap(ai_cs_2018.reshape(7, -1), ax = ax3, vmax = 0, cmap = sns.diverging_palette(10, 240, n = 30), cbar = False)
    ax3.set_xticks(np.array([math.floor(val) for val in np.linspace(0, 53, 13)][:-1]))
    ax3.set_xticklabels(month_label_lst, rotation = 90, size = 16), ax3.set_yticklabels(days, rotation = 0, size = 16)
    ax3.set_ylabel('2018', size = 25, labelpad = 20), ax3.set_facecolor('gray')
    ax4 = sns.heatmap(ai_cs_2017.reshape(7, -1), ax = ax4, vmax = 0, cmap = sns.diverging_palette(10, 240, n = 30), cbar = False)
    ax4.set_xticks(np.array([math.floor(val) for val in np.linspace(0, 53, 13)][:-1]))
    ax4.set_xticklabels(month_label_lst, rotation = 90, size = 16), ax4.set_yticklabels(days, rotation = 0, size = 16)
    ax4.set_ylabel('2017', size = 25, labelpad = 20), ax4.set_facecolor('gray')

    mappable = ax1.get_children()[0]
    cbar = plt.colorbar(mappable, ax = [ax1, ax2, ax3, ax4], orientation = 'vertical', pad = 0.03)
    cbar.ax.set_ylabel('$Return$', rotation = 270, size = 16, labelpad = 25)
    cbar.ax.tick_params(labelsize = 13)
    plt.suptitle('Heatmap of Return per Episode', size = 30, x = 0.44, y = 0.94)
    plt.savefig('./RESULTS/return_per_episode_heatmap.png')

def return_linegraph(result_data):
    plt.figure(figsize=(12, 4))
    plt.plot(result_data['Return'], linewidth=0.7)
    plt.xlabel('$Episodes$'), plt.ylabel('$Return$')
    plt.title('$Return$ $vs.$ $Episodes$')
    plt.savefig('./RESULTS/episodes_vs_return.png')


with open(os.getcwd()+'/config.json') as f:
    config = json.load(f)

date_lst, return_lst = [], []
if __name__ == "__main__":
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    env = asi_bems().make(config['SIMULATOR_VERSION'])
    print('\n=================================================================================================================')
    print('=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\t\t\tACTIVATING REINFORCEMENT LEARNING\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    print('=\t\t\tSimulator Version \t\t: \t', config['SIMULATOR_VERSION'], '\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')

    # define the state and the action space size
    state_size = env.observation_space()['shape'][0]         #4, [OA, RA, Month, Time]
    action_size = env.action_space()['shape'][0]             #2, [e_prod, damper]
    print('=\t\t\tState Size \t\t\t: \t', state_size, '\t\t\t\t\t\t=\n=\t\t\tAction Size \t\t\t: \t', action_size, '\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    
    # upper and lower bounds of the observation space
    observation_upper_bound = env.observation_space()['high']
    observation_lower_bound = env.observation_space()['low']
    print('=\t\t\tObservation Upper Bound \t: \t', observation_upper_bound, '\t\t\t=\n=\t\t\tObservation Lower Bound \t: \t', observation_lower_bound, '\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')

    # upper and lower bounds of the action space
    action_upper_bound = env.action_space()['high']
    action_lower_bound = env.action_space()['low']
    print('=\t\t\tAction Upper Bound \t\t: \t', action_upper_bound, '\t\t\t\t=\n=\t\t\tAction Lower Bound \t\t: \t', action_lower_bound, '\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    print('=================================================================================================================\n')
    
    # ddpg is specifically adapted for environments with continuous action spaces
    sess = tf.compat.v1.Session()
    ddpg = DDPG(sess, config, state_size, action_size, action_lower_bound, action_upper_bound)

    step = 0
    for episode in range(config['EPISODES']):
        # for each episode, reset the environment
        state, date = env.reset()
        date_lst.append(date)

        # initialize the total return
        total_return = 0
        
        while True:
            # t
            # retrieve the action from the ddpg model
            action = ddpg.act(state)

            # input the action to the environment, and obtain the following
            next_state, reward, done = env.step(action)

            # store it in the replay experience queue and go to the next state
            ddpg.remember(state, action, reward, next_state)
            #print('State :\t', state, 'Action :\t', action, 'Reward :\t', reward, 'Next State :\t',next_state)

            # if there are enough instances in the replay experience queue, start the training
            if config['COUNTER'] > config['MEMORY_CAPACITY']:
                ddpg.train()
                step += 1

            # t + 1
            # go to the next state
            state = next_state

            # add to the return
            total_return += reward

            # if the episode is finished, go to the next episode
            if done:
                print("Episode: %i / %s,\tReturn: %.4f,\tCounter: %i,\t\tE Prod Stand Dev: %.4f,\tDamper Stand Dev: %.4f,\tExploration Rate: %.4f" % \
                     (episode, date, total_return, config['COUNTER'], config['E_PROD_STAND_DEV'], config['DAMP_STAND_DEV'], config['EPSILON']))
                return_lst.append(total_return)
                break

    ddpg.save_actor_critic_result()
    env.save_exp_result()
    result_data = pd.DataFrame({'Date' : date_lst, 'Return' : return_lst})
    result_data['Date'] = pd.to_datetime(result_data['Date'])
    result_data.to_csv('./RESULTS/rl_result.csv', index=False)
    return_linegraph(result_data)
    
    result_data = result_data.sort_values(by = 'Date').reset_index(drop=True)
    result_data['Year'] = result_data['Date'].dt.year
    result_data['Month'] = result_data['Date'].dt.month
    result_data['Day'] = result_data['Date'].dt.day
    result_data = result_data[['Year', 'Month', 'Day', 'Return']]

    vis_data = pd.DataFrame()
    for unique_year in np.unique(result_data['Year']):
        sub_data = result_data[result_data['Year'] == unique_year]
        filtered_data = fill_na_days(sub_data, unique_year)
        vis_data = vis_data.append(filtered_data)

    vis_data = vis_data.sort_values(by = ['Year', 'Month', 'Day']).reset_index(drop=True)
    #daily_heatmap(vis_data)
    vis_data.to_csv('./RESULTS/heatmap_result.csv', index=False)
