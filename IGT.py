import pandas as pd
import numpy as np
import torch
import pickle
from statsmodels.iolib.summary import summary
from LSTM import LSTM_Fitting, encode_trial_type, find_best_choice, find_trial_type, read_pickle
from utils.ComputationalModeling import ComputationalModels, dict_generator


def IGT_data_reader(file_path, trial_prefix='Choice_', value_name='choice'):
    """
    Reads the IGT data from a CSV file and returns a DataFrame.

    Parameters:
    file_path (str): The path to the CSV file containing IGT data.
    trial_prefix (str): The prefix used in the trial column names.
    value_name (str): The name of the column to hold the values (e.g., 'choice', 'gains', 'losses').

    Returns:
    pd.DataFrame: A DataFrame containing the IGT data.
    """
    df = pd.read_csv(file_path)
    df_long = (df.reset_index().melt(id_vars='index', var_name='trial', value_name=value_name)).rename(columns={'index': 'Subnum'})
    df_long['trial'] = df_long['trial'].str.replace(trial_prefix, '').astype(int)
    df_long = df_long.sort_values(by=['Subnum', 'trial'])
    return df_long


def summary_df_generator(choice_path, gains_path, losses_path):
    choice_df = IGT_data_reader(file_path=choice_path, trial_prefix='Choice_', value_name='choice')
    gains_df = IGT_data_reader(file_path=gains_path, trial_prefix='Wins_', value_name='gains')
    losses_df = IGT_data_reader(file_path=losses_path, trial_prefix='Losses_', value_name='losses')

    # Combine the dataframes
    df = pd.merge(choice_df, gains_df, on=['Subnum', 'trial'], how='left')
    df = pd.merge(df, losses_df, on=['Subnum', 'trial'], how='left')
    df['outcome'] = df['gains'] + df['losses']
    df['Subnum'] = df['Subnum'].str.replace('Subj_', '').astype(int)
    df = df.sort_values(by=['Subnum', 'trial']).reset_index(drop=True)

    return df


# =============================================================================
# Load the data
# =============================================================================
if __name__ == '__main__':
    # Read the IGT many labs data for included experiments that have at least 100 trials
    IGT_95 = summary_df_generator('./Data/raw_data/IGTdataSteingroever2014/choice_95.csv',
                                   './Data/raw_data/IGTdataSteingroever2014/wi_95.csv',
                                   './Data/raw_data/IGTdataSteingroever2014/lo_95.csv') # We will not use 95-trial IGT
    IGT_100 = summary_df_generator('./Data/raw_data/IGTdataSteingroever2014/choice_100.csv',
                                   './Data/raw_data/IGTdataSteingroever2014/wi_100.csv',
                                   './Data/raw_data/IGTdataSteingroever2014/lo_100.csv')
    IGT_150 = summary_df_generator('./Data/raw_data/IGTdataSteingroever2014/choice_150.csv',
                                   './Data/raw_data/IGTdataSteingroever2014/wi_150.csv',
                                   './Data/raw_data/IGTdataSteingroever2014/lo_150.csv')
    print(IGT_95.groupby('Subnum').size().value_counts())
    print(IGT_100.groupby('Subnum').size().value_counts())
    print(IGT_150.groupby('Subnum').size().value_counts())

    # Remove the last 50 trials from the 150 trials dataset
    IGT_150 = IGT_150[IGT_150['trial'] <= 100].reset_index(drop=True)
    print(IGT_150.groupby('Subnum').size().value_counts())

    # Combine the datasets
    IGT_manylabs = pd.concat([IGT_100, IGT_150], axis=0).reset_index(drop=True)

    # Reset the index and subject number
    IGT_manylabs['Subnum'] = (IGT_manylabs.index // 100) + 1
    print(IGT_manylabs.groupby('Subnum').size().value_counts())

    # define the kept variables
    kept_var = ['Subnum', 'trial', 'choice', 'outcome']
    IGT = IGT_manylabs[kept_var]
    print(IGT.groupby('Subnum').size().value_counts())
    IGT.to_csv('./Data/processed_data/IGT_total.csv', index=False)
