"""BOLTZMANN RL

.. moduleauthor:: Sean Sungil Kim <sungilkim3314@gmail.com>

Controls the overall flow of reinforcement learning. There are some functions for saving results in .csv format and for visualization purposes as well.

"""

import tensorflow as tf
import numpy as np
import pandas as pd
from networks import DDPG
from sac_networks import SAC
from simulator import Simulator
from visualize import Visualize
from datetime import timedelta
import json
import os
import shutil
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


#"SIMULATOR_VERSION" : ["ShinSaeGae_v0", "Inha_NewBuilding_v0"]
#"SEASON"            : ["summer", "winter", 'summer-winter', "all"]
#"RL_VERSION"        : ["DDPG", "SAC"]

def fill_na_days(data, col_name, year):
    """Performs inputting np.nan values for missing days, so that the heatmap code can detect them and mark them grey.

    Args:
        data (pd.DataFrame): daily result data; it should contain the column name of interest.
        col_name (str): column name of interest.
        year (int): 4 digit year number.

    Returns:
        A pandas DataFrame containing original and missing daily data.

    """
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

def save_heatmap_result(result_data, simulator_version, col_name, MODE):
    """Uses fill_na_days, conv_to_calmap, and daily_heatmap functions to generate/save the heatmap and the corresponding csvs.

    Args:
        result_data (pd.DataFrame): daily result data; it should contain the column name of interest.
        simulator_version (str): version of the simulator.
        col_name (str): column name of interest.
        MODE (str): train vs. test phase.

    """
    vis_data = pd.DataFrame()
    for unique_year in np.unique(result_data['Year']):
        sub_data = result_data[result_data['Year'] == unique_year]
        filtered_data = fill_na_days(sub_data, col_name, unique_year)
        vis_data = vis_data.append(filtered_data)

    vis_data = vis_data.sort_values(by = ['Year', 'Month', 'Day']).reset_index(drop=True)
    Visualize.daily_heatmap(vis_data, simulator_version, col_name, MODE, col_name.lower() + '_trend_heatmap.png')
    vis_data.to_csv('./RESULTS/' + MODE.upper() + '/' + col_name.lower() + '_heatmap_result.csv', index=False)

def run_ShinSaeGae_v0_rl(MODE, config):
    """Controls the overall flow of reinforcement learning.
    
    Utilizes the ShinSaeGae_v0 simulator environment. Therefore some variables are specific to the environment.
    There are some functions for saving results in .csv format and for visualization purposes as well.

    Args:
        MODE (str): train vs. test phase.
        config (dict): configuration file in JSON format.

    """
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
    env = Simulator().make(config['SIMULATOR_VERSION'], config["SEASON"], MODE)

    print('\n=================================================================================================================')
    print('=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\tACTIVATING REINFORCEMENT LEARNING\t\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    print('=\tSimulator Version \t\t: \t', config['SIMULATOR_VERSION'], '\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')

    # define the state and the action space size
    state_size = env.observation_space()['shape'][0]         #5, [TT_min-RA, TT_max-RA, OA-RA, Month, Time]
    action_size = env.action_space()['shape'][0]             #1, [e_prod]
    print('=\tState Size \t\t\t: \t', state_size, '\t\t\t\t\t\t\t\t=\n=\tAction Size \t\t\t: \t', action_size, '\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    
    # upper and lower bounds of the observation space
    observation_upper_bound = env.observation_space()['high']
    observation_lower_bound = env.observation_space()['low']
    print('=\tObservation Upper Bound \t: \t', observation_upper_bound, '\t\t\t\t\t=\n=\tObservation Lower Bound \t: \t', observation_lower_bound, '\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')

    # upper and lower bounds of the action space
    action_upper_bound = env.action_space()['high']
    action_lower_bound = env.action_space()['low']
    print('=\tAction Upper Bound \t\t: \t', action_upper_bound, '\t\t\t\t\t\t\t=\n=\tAction Lower Bound \t\t: \t', action_lower_bound, '\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    print('=================================================================================================================\n')
    
    if MODE == 'test':
        config['EPISODES'] = len(env.test_idx)

    # ddpg is specifically adapted for environments with continuous action spaces
    if config['RL_VERSION'] == "DDPG":
        agent = DDPG(config, MODE, state_size, action_size, action_lower_bound, action_upper_bound)
    elif config['RL_VERSION'] == "SAC":
        agent =  SAC(config, MODE, state_size, action_size, action_lower_bound, action_upper_bound)

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
            action = agent.act(state)

            # input the action to the environment, and obtain the following
            next_state, reward, done = env.step(action, agent, MODE)

            if MODE == 'train':
                # store it in the replay experience queue and go to the next state
                agent.remember(state, action, [reward], next_state)
                #print('State :\t', state, 'Action :\t', action, 'Reward :\t', reward, 'Next State :\t',next_state)

                # if there are enough instances in the replay experience queue, start the training
                if config['COUNTER'] > config['MEMORY_CAPACITY']*0.7:
                    if training_started == None:
                        training_started = episode
                    #min_actor_loss, min_critic_loss = agent.train()
                    agent.train()
            
            # t + 1
            # go to the next state
            state = next_state

            # add to the return
            total_return += reward

            # if the episode is finished, go to the next episode
            if done:
                if MODE == 'train':
                    print("Episode: %i / %s\tReturn: %.1f\t\tCounter: %i\t\tStand Dev: %.2f\t\tExploration Rate: %.2f" % \
                        (episode+1, date, total_return, config['COUNTER'], config['STAND_DEV'], config['EPSILON']))
                elif MODE == 'test':
                    print("Episode: %i / %s,\tReturn: %.1f" % (episode+1, date, total_return))
                    
                return_lst.append(total_return)
                break

    env.save_exp_result(MODE)

    # saving daily rl return result
    rl_result_data = pd.DataFrame({'Date' : date_lst, 'Return' : return_lst})
    rl_result_data['Date'] = pd.to_datetime(rl_result_data['Date'])
    rl_result_data.to_csv('./RESULTS/'+MODE.upper()+'/rl_result.csv', index=False)
    
    if MODE == 'train':
        #print('\n\nMin Actor Loss = %.4f\nMin Critic Loss = %.4f\n\n' % (min_actor_loss, min_critic_loss))
        agent.save_actor_critic_result()
        Visualize.return_linegraph(rl_result_data, MODE, training_started)
    elif MODE == 'test':
        Visualize.return_linegraph(rl_result_data, MODE)
        
        rl_result_data = rl_result_data.sort_values(by = 'Date').reset_index(drop=True)
        rl_result_data['Year'] = rl_result_data['Date'].dt.year
        rl_result_data['Month'] = rl_result_data['Date'].dt.month
        rl_result_data['Day'] = rl_result_data['Date'].dt.day
        rl_result_data = rl_result_data[['Year', 'Month', 'Day', 'Return']]

        save_heatmap_result(rl_result_data, config['SIMULATOR_VERSION'], 'Return', MODE)

        result_data = pd.read_csv('./RESULTS/TEST/result.csv')
        result_data['Korean DateTime'] = pd.to_datetime(result_data['Korean DateTime'])-timedelta(hours=6) # day starts from 6:00am
        result_data['Date'] = result_data['Korean DateTime'].dt.date
        result_data['Month'] = result_data['Korean DateTime'].dt.year.astype(str)+'-'+result_data['Korean DateTime'].dt.month.astype(str)
        result_data['Year'] = result_data['Korean DateTime'].dt.year.astype(str)

        open_data = pd.read_csv('./RESULTS/TEST/open_hours_result.csv')
        open_data['Korean DateTime'] = pd.to_datetime(open_data['Korean DateTime'])-timedelta(hours=6) # day starts from 6:00am
        open_data['Date'] = open_data['Korean DateTime'].dt.date
        open_data['Month'] = open_data['Korean DateTime'].dt.year.astype(str)+'-'+open_data['Korean DateTime'].dt.month.astype(str)
        open_data['Year'] = open_data['Korean DateTime'].dt.year.astype(str)
        
        daily_data = result_data[['Date', 'Pred e_prod', 'Org e_prod', 'Pred Cost', 'Org Cost', 'Pred CO2_emissions', 'Org CO2_emissions']].groupby(['Date']).apply(lambda c: c.abs().sum()).reset_index()
        #daily_data = result_data[['Date', 'Pred e_prod', 'Org e_prod']].groupby(['Date']).apply(lambda c: c.abs().sum()).reset_index()
        daily_data['temp_metric'] = open_data[['Date', 'within_temp_range']].groupby(['Date']).apply(lambda c: c.sum()/len(c)).reset_index()['within_temp_range']
        daily_data['e_prod_saved_percent'] = [((daily_data['Org e_prod'].iloc[i]-daily_data['Pred e_prod'].iloc[i]) / daily_data['Org e_prod'].iloc[i])*100 if daily_data['Org e_prod'].iloc[i] != 0 else -100 for i in range(len(daily_data['Org e_prod']))]
        daily_data['cost_saved_percent'] = [((daily_data['Org Cost'].iloc[i]-daily_data['Pred Cost'].iloc[i]) / daily_data['Org Cost'].iloc[i])*100 if daily_data['Org Cost'].iloc[i] != 0 else -100 for i in range(len(daily_data['Org Cost']))]
        daily_data['CO2_saved_percent'] = [((daily_data['Org CO2_emissions'].iloc[i]-daily_data['Pred CO2_emissions'].iloc[i]) / daily_data['Org CO2_emissions'].iloc[i])*100 if daily_data['Org CO2_emissions'].iloc[i] != 0 else -100 for i in range(len(daily_data['Org CO2_emissions']))]
        daily_data['Date'] = pd.to_datetime(daily_data['Date'])
        daily_data['Year'] = daily_data['Date'].dt.year
        daily_data['Month'] = daily_data['Date'].dt.month
        daily_data['Day'] = daily_data['Date'].dt.day
        daily_data.to_csv('./RESULTS/'+MODE.upper()+'/daily_summary.csv', index=False)

        save_heatmap_result(daily_data[['Year', 'Month', 'Day', 'e_prod_saved_percent']], config['SIMULATOR_VERSION'], 'e_prod_saved_percent', MODE)
        save_heatmap_result(daily_data[['Year', 'Month', 'Day', 'cost_saved_percent']], config['SIMULATOR_VERSION'], 'cost_saved_percent', MODE)
        save_heatmap_result(daily_data[['Year', 'Month', 'Day', 'CO2_saved_percent']], config['SIMULATOR_VERSION'], 'CO2_saved_percent', MODE)

        monthly_data = result_data[['Month', 'Pred e_prod', 'Org e_prod', 'Pred Cost', 'Org Cost', 'Pred CO2_emissions', 'Org CO2_emissions']].groupby(['Month']).apply(lambda c: c.abs().sum()).reset_index()
        #monthly_data = result_data[['Month', 'Pred e_prod', 'Org e_prod']].groupby(['Month']).apply(lambda c: c.abs().sum()).reset_index()
        monthly_data['temp_metric'] = open_data[['Month', 'within_temp_range']].groupby(['Month']).apply(lambda c: c.sum()/len(c)).reset_index()['within_temp_range']
        monthly_data['e_prod_saved_percent'] = [((monthly_data['Org e_prod'].iloc[i]-monthly_data['Pred e_prod'].iloc[i])/monthly_data['Org e_prod'].iloc[i])*100 if monthly_data['Org e_prod'].iloc[i] != 0 else -100 for i in range(len(monthly_data['Org e_prod']))]
        monthly_data['cost_saved_percent'] = [((monthly_data['Org Cost'].iloc[i]-monthly_data['Pred Cost'].iloc[i])/monthly_data['Org Cost'].iloc[i])*100 if monthly_data['Org Cost'].iloc[i] != 0 else -100 for i in range(len(monthly_data['Org Cost']))]
        monthly_data['CO2_saved_percent'] = [((monthly_data['Org CO2_emissions'].iloc[i]-monthly_data['Pred CO2_emissions'].iloc[i])/monthly_data['Org CO2_emissions'].iloc[i])*100 if monthly_data['Org CO2_emissions'].iloc[i] != 0 else -100 for i in range(len(monthly_data['Org CO2_emissions']))]
        monthly_data.to_csv('./RESULTS/'+MODE.upper()+'/monthly_summary.csv', index=False)

        yearly_data = result_data[['Year', 'Pred e_prod', 'Org e_prod', 'Pred Cost', 'Org Cost', 'Pred CO2_emissions', 'Org CO2_emissions']].groupby(['Year']).apply(lambda c: c.abs().sum()).reset_index()
        #yearly_data = result_data[['Year', 'Pred e_prod', 'Org e_prod']].groupby(['Year']).apply(lambda c: c.abs().sum()).reset_index()
        yearly_data['temp_metric'] = open_data[['Year', 'within_temp_range']].groupby('Year').apply(lambda c: c.sum()/len(c)).reset_index()['within_temp_range']
        yearly_data['e_prod_saved_percent'] = [((yearly_data['Org e_prod'].iloc[i]-yearly_data['Pred e_prod'].iloc[i])/yearly_data['Org e_prod'].iloc[i])*100 if yearly_data['Org e_prod'].iloc[i] != 0 else -100 for i in range(len(yearly_data['Org e_prod']))]
        yearly_data['cost_saved_percent'] = [((yearly_data['Org Cost'].iloc[i]-yearly_data['Pred Cost'].iloc[i])/yearly_data['Org Cost'].iloc[i])*100 if yearly_data['Org Cost'].iloc[i] != 0 else -100 for i in range(len(yearly_data['Org Cost']))]
        yearly_data['CO2_saved_percent'] = [((yearly_data['Org CO2_emissions'].iloc[i]-yearly_data['Pred CO2_emissions'].iloc[i])/yearly_data['Org CO2_emissions'].iloc[i])*100 if yearly_data['Org CO2_emissions'].iloc[i] != 0 else -100 for i in range(len(yearly_data['Org CO2_emissions']))]
        yearly_data.to_csv('./RESULTS/'+MODE.upper()+'/yearly_summary.csv', index=False)

def run_Inha_NewBuilding_v0_rl(MODE, config):
    """Controls the overall flow of reinforcement learning.
    
    Utilizes the Inha_NewBuilding_v0 simulator environment. Therefore some variables are specific to the environment.
    There are some functions for saving results in .csv format and for visualization purposes as well.

    Args:
        MODE (str): train vs. test phase.
        config (dict): configuration file in JSON format.
    
    """
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
    env = Simulator().make(config, MODE)

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
    print('=\t\t\tObservation Upper Bound \t: \t', observation_upper_bound, '\t\t\t\t\t\t=\n=\t\t\tObservation Lower Bound \t: \t', observation_lower_bound, '\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')

    # upper and lower bounds of the action space
    action_upper_bound = env.action_space()['high']
    action_lower_bound = env.action_space()['low']
    print('=\t\t\tAction Upper Bound \t\t: \t', action_upper_bound, '\t\t\t\t\t=\n=\t\t\tAction Lower Bound \t\t: \t', action_lower_bound, '\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t=\n=\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t=')
    print('=========================================================================================================================\n')
    
    if MODE == 'test':
        config['EPISODES'] = 1094

    # ddpg is specifically adapted for environments with continuous action spaces
    agent = DDPG(config, MODE, state_size, action_size, action_lower_bound, action_upper_bound)

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
            action = agent.act(state)
            
            # input the action to the environment, and obtain the following
            next_state, reward, done = env.step(action)
            
            if MODE == 'train':
                # store it in the replay experience queue and go to the next state
                agent.remember(state, action, reward, next_state)
                #print('State :\t', state, 'Action :\t', action, 'Reward :\t', reward, 'Next State :\t',next_state)

                # if there are enough instances in the replay experience queue, start the training
                if config['COUNTER'] > config['MEMORY_CAPACITY']:
                    if training_started == None:
                        training_started = episode
                    min_actor_loss, min_critic_loss = agent.train()

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
        agent.save_actor_critic_result()
        Visualize.return_linegraph(rl_result_data, MODE, training_started)
    elif MODE == 'test':
        Visualize.return_linegraph(rl_result_data, MODE)

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
