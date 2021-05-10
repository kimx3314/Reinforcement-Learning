# sean sungil kim

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import seaborn as sns
from ddpg_networks import DDPG
from asi_bems import asi_bems
import estimator_cls
from datetime import timedelta
import json
import os
import shutil
import math
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


def fill_na_days(data, col_name, year):
    month_31 = [1, 3, 5, 7, 8, 10, 12]

    for num_month in range(1, 13):
        sub_data = data[data['Month'] == num_month]
        if num_month in month_31:
            for num_day in range(1, 32):
                if num_day not in sub_data['Day'].tolist():
                    data = data.append(pd.DataFrame({'Year':[year], 'Month':[num_month], 'Day':[num_day], col_name:[np.nan]}))
        elif num_month == 2:
            if year % 4 == 0:
                max_days = 29
            else:
                max_days = 28
            for num_day in range(1, max_days+1):
                if num_day not in sub_data['Day'].tolist():
                    data = data.append(pd.DataFrame({'Year':[year], 'Month':[num_month], 'Day':[num_day], col_name:[np.nan]}))
        else:
            for num_day in range(1, 31):
                if num_day not in sub_data['Day'].tolist():
                    data = data.append(pd.DataFrame({'Year':[year], 'Month':[num_month], 'Day':[num_day], col_name:[np.nan]}))

    return data

def conv_to_calamp(data):
    data = np.concatenate((data[0::7], data[1::7], data[2::7], data[3::7], data[4::7], data[5::7], data[6::7]))

    return data

def daily_heatmap(vis_data, simulator_version, col_name, MODE, filename):
    if col_name == 'Return':
        vmin = vis_data[col_name].min()
        vmax = vis_data[col_name].max()
        cmap = "Blues"
    elif col_name == 'cost_saved_percent':
        vmin = -100
        vmax = 100
        cmap = "coolwarm_r"
    elif col_name == 'e_prod_saved_percent':
        vmin = -100
        vmax = 100
        cmap = "coolwarm_r"

    month_label_lst = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize = (22, 14), gridspec_kw = {'hspace':0.5})

    if simulator_version == 'ShinSaeGae_v0':
        daily_data_2020 = vis_data[vis_data['Year'] == 2020]
        ai_cs_2020 = conv_to_calamp(np.concatenate((np.full((2, ), np.nan), np.array(daily_data_2020[col_name]), np.full((3, ), np.nan))))
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (22, 14), gridspec_kw = {'hspace':0.5})
        ax1 = sns.heatmap(ai_cs_2020.reshape(7, -1), ax = ax1, vmin=vmin, vmax=vmax, cmap=cmap, cbar = False)
        ax1.set_xticks(np.array([math.floor(val) for val in np.linspace(0, 53, 13)][:-1]))
        ax1.set_xticklabels(month_label_lst, rotation = 90, size = 16), ax1.set_yticklabels(days, rotation = 0, size = 16)
        ax1.set_ylabel('2020', size = 25, labelpad = 20), ax1.set_facecolor('gray')
    
    daily_data_2019 = vis_data[vis_data['Year'] == 2019]
    daily_data_2018 = vis_data[vis_data['Year'] == 2018]
    daily_data_2017 = vis_data[vis_data['Year'] == 2017]
    
    ai_cs_2019 = conv_to_calamp(np.concatenate((np.full((1, ), np.nan), np.array(daily_data_2019[col_name]), np.full((5, ), np.nan))))
    ai_cs_2018 = conv_to_calamp(np.concatenate((np.array(daily_data_2018[col_name]), np.full((6, ), np.nan))))
    ai_cs_2017 = conv_to_calamp(np.concatenate((np.full((6, ), np.nan), np.array(daily_data_2017[col_name]))))
    
    ax2 = sns.heatmap(ai_cs_2019.reshape(7, -1), ax = ax2, vmin=vmin, vmax=vmax, cmap=cmap, cbar = False)
    ax2.set_xticks(np.array([math.floor(val) for val in np.linspace(0, 53, 13)][:-1]))
    ax2.set_xticklabels(month_label_lst, rotation = 90, size = 16), ax2.set_yticklabels(days, rotation = 0, size = 16)
    ax2.set_ylabel('2019', size = 25, labelpad = 20), ax2.set_facecolor('gray')
    ax3 = sns.heatmap(ai_cs_2018.reshape(7, -1), ax = ax3, vmin=vmin, vmax=vmax, cmap=cmap, cbar = False)
    ax3.set_xticks(np.array([math.floor(val) for val in np.linspace(0, 53, 13)][:-1]))
    ax3.set_xticklabels(month_label_lst, rotation = 90, size = 16), ax3.set_yticklabels(days, rotation = 0, size = 16)
    ax3.set_ylabel('2018', size = 25, labelpad = 20), ax3.set_facecolor('gray')
    ax4 = sns.heatmap(ai_cs_2017.reshape(7, -1), ax = ax4, vmin=vmin, vmax=vmax, cmap=cmap, cbar = False)
    ax4.set_xticks(np.array([math.floor(val) for val in np.linspace(0, 53, 13)][:-1]))
    ax4.set_xticklabels(month_label_lst, rotation = 90, size = 16), ax4.set_yticklabels(days, rotation = 0, size = 16)
    ax4.set_ylabel('2017', size = 25, labelpad = 20), ax4.set_facecolor('gray')

    if simulator_version == 'ShinSaeGae_v0':
        mappable = ax1.get_children()[0]
        cbar = plt.colorbar(mappable, ax = [ax1, ax2, ax3, ax4], orientation = 'vertical', pad = 0.03)
    elif simulator_version == 'Inha_NewBuilding_v0':
        mappable = ax2.get_children()[0]
        cbar = plt.colorbar(mappable, ax = [ax2, ax3, ax4], orientation = 'vertical', pad = 0.03)
    cbar.ax.tick_params(labelsize = 13)

    if col_name == 'Return':
        cbar.ax.set_ylabel('$Return$', rotation = 270, size = 16, labelpad = 25)
        plt.suptitle('$Heatmap$ $of$ $Return$ $Trend$', size = 30, x = 0.44, y = 0.94)
    elif col_name == 'cost_saved_percent':
        cbar.ax.set_ylabel('$Cost$ $Saved$ (%)', rotation = 270, size = 16, labelpad = 25)
        plt.suptitle('$Heatmap$ $of$ $Cost$ $Saved$ $Trend$', size = 30, x = 0.44, y = 0.94)
    elif col_name == 'e_prod_saved_percent':
        cbar.ax.set_ylabel('$Energy$ $Production$ $Saved$ (%)', rotation = 270, size = 16, labelpad = 25)
        plt.suptitle('$Heatmap$ $of$ $Energy$ $Production$ $Saved$ $Trend$', size = 30, x = 0.44, y = 0.94)

    plt.tight_layout(), plt.savefig('./RESULTS/'+MODE.upper()+'/'+filename)

def save_heatmap_result(result_data, simulator_version, col_name, MODE):
    vis_data = pd.DataFrame()
    for unique_year in np.unique(result_data['Year']):
        sub_data = result_data[result_data['Year'] == unique_year]
        filtered_data = fill_na_days(sub_data, col_name, unique_year)
        vis_data = vis_data.append(filtered_data)

    vis_data = vis_data.sort_values(by = ['Year', 'Month', 'Day']).reset_index(drop=True)
    daily_heatmap(vis_data, simulator_version, col_name, MODE, col_name.lower() + '_trend_heatmap.png')
    vis_data.to_csv('./RESULTS/' + MODE.upper() + '/' + col_name.lower() + '_heatmap_result.csv', index=False)

def return_linegraph(result_data, MODE, training_started=None):
    if MODE == 'train':
        plt.figure(figsize=(12, 4))
        plt.plot(result_data['Return'], linewidth=0.6, label='Daily Return')
        plt.axvline(training_started, linewidth=2, color="r", label='Training Phase Began')
        plt.xlabel('$Episodes$'), plt.ylabel('$Return$'), plt.legend()
        plt.title('$Return$ $vs.$ $Episodes$'), plt.tight_layout()
        plt.savefig('./RESULTS/'+MODE.upper()+'/return_vs_episodes.png')
    elif MODE == 'test':
        dates = date2num(result_data['Date'])

        plt.figure(figsize=(12, 4))
        plt.plot_date(dates, result_data['Return'], '-', linewidth=0.6)
        plt.xlabel('$Date$'), plt.ylabel('$Return$'), plt.xticks(rotation=45)
        plt.title('$Return$ $Trend$'), plt.tight_layout()
        plt.savefig('./RESULTS/'+MODE.upper()+'/return_trend.png')

def run_ShinSaeGae_v0_rl(MODE, config):
    if os.path.isdir('./RESULTS/TRAIN/') and MODE == 'train':
        shutil.rmtree('./RESULTS/TRAIN/')
    if os.path.isdir('./RESULTS/TEST/') and MODE == 'test':
        shutil.rmtree('./RESULTS/TEST/')
    if os.path.isdir('./GRAPHS') and MODE == 'train':
        shutil.rmtree('./GRAPHS')

    os.makedirs('./RESULTS/TRAIN/', exist_ok = True)
    os.makedirs('./RESULTS/TEST/', exist_ok = True)
    os.makedirs('./GRAPHS', exist_ok = True)

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    estimator = estimator_cls.Estimator(config)
    env = asi_bems().make(config['SIMULATOR_VERSION'], MODE, estimator)

    print('\n=================================================================================================================')
    print('=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\tACTIVATING REINFORCEMENT LEARNING\t\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    print('=\tSimulator Version \t\t: \t', config['SIMULATOR_VERSION'], '\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')

    # define the state and the action space size
    state_size = env.observation_space()['shape'][0]         #6, [OA, RA, Q_g, TT-RA, OA-RA, Month, Time]
    action_size = env.action_space()['shape'][0]             #3, [nAC, nSTC, nLTC] [6, 2, 3]
    print('=\tState Size \t\t\t: \t', state_size, '\t\t\t\t\t\t\t\t=\n=\tAction Size \t\t\t: \t', action_size, '\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    
    # upper and lower bounds of the observation space
    observation_upper_bound = env.observation_space()['high']
    observation_lower_bound = env.observation_space()['low']
    print('=\tObservation Upper Bound \t: \t', observation_upper_bound, '\t\t=\n=\tObservation Lower Bound \t: \t', observation_lower_bound, '\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')

    # upper and lower bounds of the action space
    action_upper_bound = env.action_space()['high']
    action_lower_bound = env.action_space()['low']
    print('=\tAction Upper Bound \t\t: \t', action_upper_bound, '\t\t\t\t\t\t\t=\n=\tAction Lower Bound \t\t: \t', action_lower_bound, '\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    print('=================================================================================================================\n')
    
    if MODE == 'test':
        config['EPISODES'] = 1041

    # ddpg is specifically adapted for environments with continuous action spaces
    ddpg = DDPG(config, MODE, state_size, action_size, action_lower_bound, action_upper_bound)

    training_started = None
    date_lst, return_lst = [], []
    for episode in range(config['EPISODES']):
        # for each episode, reset the environment
        state, date = env.reset()
        date_lst.append(date)
        
        # initialize the total return
        total_return = 0
        preAction = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        while True:
            # t
            # retrieve the action from the ddpg model
            action = ddpg.act(state)
            
            # convert the action to the e_prod value
            status = 'heat' if action[0] < 0 else 'cool'
            ind_e_prod, e_prod = estimator.action_to_energy(action[0], action[1], action[2], preAction[-1], status)
            
            # input the action to the environment, and obtain the following
            next_state, reward, done = env.step(action, e_prod, ind_e_prod, status)
            
            if MODE == 'train':
                # store it in the replay experience queue and go to the next state
                ddpg.remember(state, action, reward, next_state)
                #print('State :\t', state, 'Action :\t', action, 'Reward :\t', reward, 'Next State :\t',next_state)

                # if there are enough instances in the replay experience queue, start the training
                if config['COUNTER'] > config['MEMORY_CAPACITY']:
                    if training_started == None:
                        training_started = episode
                    min_actor_loss, min_critic_loss = ddpg.train()

            # t + 1
            # go to the next state
            state = next_state
            preAction[-3] = preAction[-2]
            preAction[-2] = preAction[-1]
            preAction[-1] = action.copy()

            # add to the return
            total_return += reward

            # if the episode is finished, go to the next episode
            if done:
                if MODE == 'train':
                    print("Episode: %i / %s\tReturn: %.1f\t\tCounter: %i\t\tE Prod Stand Dev: %.2f\t\tDamper Stand Dev: %.2f\t\tExploration Rate: %.2f" % \
                        (episode, date, total_return, config['COUNTER'], config['CHILLER_STAND_DEV'], config['DAMP_STAND_DEV'], config['EPSILON']))
                elif MODE == 'test':
                    print("Episode: %i / %s,\tReturn: %.1f" % (episode, date, total_return))
                    
                return_lst.append(total_return)
                break
    
    env.save_exp_result()
    rl_result_data = pd.DataFrame({'Date' : date_lst, 'Return' : return_lst})
    rl_result_data['Date'] = pd.to_datetime(rl_result_data['Date'])
    rl_result_data.to_csv('./RESULTS/'+MODE.upper()+'/rl_result.csv', index=False)
    
    if MODE == 'train':
        print('\n\nMin Actor Loss = %.4f\nMin Critic Loss = %.4f\n\n' % (min_actor_loss, min_critic_loss))
        ddpg.save_actor_critic_result()
        return_linegraph(rl_result_data, MODE, training_started)
    elif MODE == 'test':
        return_linegraph(rl_result_data, MODE)

        rl_result_data = rl_result_data.sort_values(by = 'Date').reset_index(drop=True)
        rl_result_data['Year'] = rl_result_data['Date'].dt.year
        rl_result_data['Month'] = rl_result_data['Date'].dt.month
        rl_result_data['Day'] = rl_result_data['Date'].dt.day
        rl_result_data = rl_result_data[['Year', 'Month', 'Day', 'Return']]

        save_heatmap_result(rl_result_data, config['SIMULATOR_VERSION'], 'Return', MODE)

        result_data = pd.read_csv('./RESULTS/TEST/result.csv')
        result_data['Korean DateTime'] = pd.to_datetime(result_data['Korean DateTime'])-timedelta(hours=6) # day starts from 6:00am
        result_data['day'] = result_data['Korean DateTime'].dt.date
        daily_data = result_data[['day', 'Pred e_prod', 'Org e_prod', 'Pred Cost', 'Org Cost']].groupby('day').apply(lambda c: c.abs().sum()).reset_index()
        daily_data['e_prod_saved_percent'] = [((daily_data['Org e_prod'].iloc[i]-daily_data['Pred e_prod'].iloc[i]) / daily_data['Org e_prod'].iloc[i])*100 if daily_data['Org e_prod'].iloc[i] != 0 else -100 for i in range(len(daily_data['Org e_prod']))]
        daily_data['cost_saved_percent'] = [((daily_data['Org Cost'].iloc[i]-daily_data['Pred Cost'].iloc[i]) / daily_data['Org Cost'].iloc[i])*100 if daily_data['Org Cost'].iloc[i] != 0 else -100 for i in range(len(daily_data['Org Cost']))]
        daily_data['day'] = pd.to_datetime(daily_data['day'])
        daily_data['Year'] = daily_data['day'].dt.year
        daily_data['Month'] = daily_data['day'].dt.month
        daily_data['Day'] = daily_data['day'].dt.day
        daily_data.to_csv('./RESULTS/'+MODE.upper()+'/daily_summary.csv', index=False)

        save_heatmap_result(daily_data[['Year', 'Month', 'Day', 'e_prod_saved_percent']], config['SIMULATOR_VERSION'], 'e_prod_saved_percent', MODE)
        save_heatmap_result(daily_data[['Year', 'Month', 'Day', 'cost_saved_percent']], config['SIMULATOR_VERSION'], 'cost_saved_percent', MODE)

        monthly_data = daily_data[['Pred e_prod', 'Org e_prod', 'Pred Cost', 'Org Cost', 'Year', 'Month']]
        monthly_data = monthly_data.groupby(['Year', 'Month']).sum().reset_index()
        monthly_data['e_prod_saved_percent'] = [((monthly_data['Org e_prod'].iloc[i]-monthly_data['Pred e_prod'].iloc[i])/monthly_data['Org e_prod'].iloc[i])*100 if monthly_data['Org e_prod'].iloc[i] != 0 else -100 for i in range(len(monthly_data['Org e_prod']))]
        monthly_data['cost_saved_percent'] = [((monthly_data['Org Cost'].iloc[i]-monthly_data['Pred Cost'].iloc[i])/monthly_data['Org Cost'].iloc[i])*100 if monthly_data['Org Cost'].iloc[i] != 0 else -100 for i in range(len(monthly_data['Org Cost']))]
        monthly_data.to_csv('./RESULTS/'+MODE.upper()+'/monthly_summary.csv', index=False)

        yearly_data = daily_data[['Pred e_prod', 'Org e_prod', 'Pred Cost', 'Org Cost', 'Year']]
        yearly_data = yearly_data.groupby(['Year']).sum().reset_index()
        yearly_data['e_prod_saved_percent'] = [((yearly_data['Org e_prod'].iloc[i]-yearly_data['Pred e_prod'].iloc[i])/yearly_data['Org e_prod'].iloc[i])*100 if yearly_data['Org e_prod'].iloc[i] != 0 else -100 for i in range(len(yearly_data['Org e_prod']))]
        yearly_data['cost_saved_percent'] = [((yearly_data['Org Cost'].iloc[i]-yearly_data['Pred Cost'].iloc[i])/yearly_data['Org Cost'].iloc[i])*100 if yearly_data['Org Cost'].iloc[i] != 0 else -100 for i in range(len(yearly_data['Org Cost']))]
        yearly_data.to_csv('./RESULTS/'+MODE.upper()+'/yearly_summary.csv', index=False)

def run_Inha_NewBuilding_v0_rl(MODE, config):
    if os.path.isdir('./RESULTS/TRAIN/') and MODE == 'train':
        shutil.rmtree('./RESULTS/TRAIN/')
    if os.path.isdir('./RESULTS/TEST/') and MODE == 'test':
        shutil.rmtree('./RESULTS/TEST/')
    if os.path.isdir('./GRAPHS') and MODE == 'train':
        shutil.rmtree('./GRAPHS')

    os.makedirs('./RESULTS/TRAIN/', exist_ok = True)
    os.makedirs('./RESULTS/TEST/', exist_ok = True)
    os.makedirs('./GRAPHS', exist_ok = True)

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    estimator = estimator_cls.Estimator(config)
    env = asi_bems().make(config['SIMULATOR_VERSION'], MODE, estimator)

    print('\n=========================================================================================================================')
    print('=\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\t\t\tACTIVATING REINFORCEMENT LEARNING\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    print('=\t\t\tSimulator Version \t\t: \t', config['SIMULATOR_VERSION'], '\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')

    # define the state and the action space size
    state_size = env.observation_space()['shape'][0]         #6, [OA, RA, Q_g, TT-RA, OA-RA, Month, Time]
    action_size = env.action_space()['shape'][0]             #3, [nAC, nSTC, nLTC] [6, 2, 3]
    print('=\t\t\tState Size \t\t\t: \t', state_size, '\t\t\t\t\t\t\t=\n=\t\t\tAction Size \t\t\t: \t', action_size, '\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    
    # upper and lower bounds of the observation space
    observation_upper_bound = env.observation_space()['high']
    observation_lower_bound = env.observation_space()['low']
    print('=\t\t\tObservation Upper Bound \t: \t', observation_upper_bound, '\t\t\t=\n=\t\t\tObservation Lower Bound \t: \t', observation_lower_bound, '\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')

    # upper and lower bounds of the action space
    action_upper_bound = env.action_space()['high']
    action_lower_bound = env.action_space()['low']
    print('=\t\t\tAction Upper Bound \t\t: \t', action_upper_bound, '\t\t\t\t\t=\n=\t\t\tAction Lower Bound \t\t: \t', action_lower_bound, '\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    print('=========================================================================================================================\n')
    
    if MODE == 'test':
        config['EPISODES'] = 1094

    # ddpg is specifically adapted for environments with continuous action spaces
    ddpg = DDPG(config, MODE, state_size, action_size, action_lower_bound, action_upper_bound)

    training_started = None
    date_lst, return_lst = [], []
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
            
            if MODE == 'train':
                # store it in the replay experience queue and go to the next state
                ddpg.remember(state, action, reward, next_state)
                #print('State :\t', state, 'Action :\t', action, 'Reward :\t', reward, 'Next State :\t',next_state)

                # if there are enough instances in the replay experience queue, start the training
                if config['COUNTER'] > config['MEMORY_CAPACITY']:
                    if training_started == None:
                        training_started = episode
                    min_actor_loss, min_critic_loss = ddpg.train()

            # t + 1
            # go to the next state
            state = next_state

            # add to the return
            total_return += reward

            # if the episode is finished, go to the next episode
            if done:
                if MODE == 'train':
                    print("Episode: %i / %s\tReturn: %.1f\t\tCounter: %i\t\tE Prod Stand Dev: %.2f\t\tDamper Stand Dev: %.2f\t\tExploration Rate: %.2f" % \
                        (episode, date, total_return, config['COUNTER'], config['E_PROD_STAND_DEV'], config['DAMP_STAND_DEV'], config['EPSILON']))
                elif MODE == 'test':
                    print("Episode: %i / %s,\tReturn: %.1f" % (episode, date, total_return))
                    
                return_lst.append(total_return)
                break
    
    env.save_exp_result()
    rl_result_data = pd.DataFrame({'Date' : date_lst, 'Return' : return_lst})
    rl_result_data['Date'] = pd.to_datetime(rl_result_data['Date'])
    rl_result_data.to_csv('./RESULTS/'+MODE.upper()+'/rl_result.csv', index=False)
    
    if MODE == 'train':
        print('\n\nMin Actor Loss = %.4f\nMin Critic Loss = %.4f\n\n' % (min_actor_loss, min_critic_loss))
        ddpg.save_actor_critic_result()
        return_linegraph(rl_result_data, MODE, training_started)
    elif MODE == 'test':
        return_linegraph(rl_result_data, MODE)

        rl_result_data = rl_result_data.sort_values(by = 'Date').reset_index(drop=True)
        rl_result_data['Year'] = rl_result_data['Date'].dt.year
        rl_result_data['Month'] = rl_result_data['Date'].dt.month
        rl_result_data['Day'] = rl_result_data['Date'].dt.day
        rl_result_data = rl_result_data[['Year', 'Month', 'Day', 'Return']]

        save_heatmap_result(rl_result_data, config['SIMULATOR_VERSION'], 'Return', MODE)


if __name__ == "__main__":
    # ShinSaeGae_v0 OR Inha_NewBuilding_v0
    with open(os.getcwd()+'/config.json') as f:
        config = json.load(f)

    if config['SIMULATOR_VERSION'] == "ShinSaeGae_v0":
        run_ShinSaeGae_v0_rl('train', config)
        run_ShinSaeGae_v0_rl('test', config)
    elif config['SIMULATOR_VERSION'] == "Inha_NewBuilding_v0":
        run_Inha_NewBuilding_v0_rl('train', config)
        run_Inha_NewBuilding_v0_rl('test', config)

    print('\n=================================================================================================================')
    print('=\t\t\tREINFORCEMENT LEARNING TRAINING & TESTING PROCESSES ARE FINISHED\t\t\t=')
    print('=================================================================================================================\n')
