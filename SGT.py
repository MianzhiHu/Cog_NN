import numpy as np
import pandas as pd
import torch
import pickle
from sympy.codegen import Print
from xarray.util.generate_ops import inplace
from LSTM import LSTM_Fitting, encode_trial_type, find_best_choice, find_trial_type, read_pickle
from utils.ComputationalModeling import ComputationalModels, dict_generator, best_param_generator

# =============================================================================
# Load the data
# =============================================================================
if __name__ == '__main__':
    sgt_2022 = pd.read_csv("./Data/raw_data/SGT_Don_2022.csv")
    sgt_2024 = pd.read_csv("./Data/raw_data/SGTControlExp1.csv")

    sgt_2022.rename(columns={'subjID': 'Subnum', 'rt': 'RT'}, inplace=True)
    sgt_2024.rename(columns={'Reward': 'outcome', 'React': 'RT', 'keyResponse': 'choice'}, inplace=True)
    print(sgt_2022['choice'].value_counts())
    print(sgt_2024['choice'].value_counts())
    print(sgt_2022.columns)
    print(sgt_2024.columns)

    sgt = pd.concat([sgt_2022, sgt_2024], axis=0)
    print(sgt_2022.groupby('Subnum').size().value_counts())
    print(sgt_2024.groupby('Subnum').size().value_counts())
    print(sgt.groupby('Subnum').size().value_counts())

    # =============================================================================
    # Prepare the SGT data
    # =============================================================================
    sgt = sgt.reset_index(drop=True)
    sgt['Subnum'] = (sgt.index // 100) + 1
    sgt['trial'] = sgt.index % 100 + 1
    sgt['choice'] = sgt['choice'].astype(int)

    # if the participant failed to respond, set the outcome to 0
    print(sgt['choice'].value_counts())
    print(sgt['outcome'].value_counts())
    sgt.loc[sgt['choice'] == -1, 'outcome'] = 0
    sgt['missed'] = sgt['choice'].apply(lambda x: 1 if x == -1 else 0)
    sgt['choice'] = sgt['choice'].replace(-1, np.nan)

    print(max(sgt['Subnum']))

    kept_var = ['Subnum', 'trial', 'RT', 'choice', 'outcome', 'missed']
    sgt = sgt[kept_var]
    sgt.to_csv('./Data/processed_data/SGT_total.csv', index=False)


    # now create dummy variables for the choice
    sgt_dummy = pd.get_dummies(sgt, columns=['choice'])

    # =============================================================================
    # Transform the data
    # =============================================================================
    # define the variables
    var = ['Subnum', 'outcome', 'missed', 'choice_1.0', 'choice_2.0', 'choice_3.0', 'choice_4.0']
    sgt_dummy = sgt_dummy[var]

    # Group df by subject
    grouped = sgt_dummy.groupby('Subnum').apply(lambda x: x.values.tolist(), include_groups=False)

    sequences = list(grouped.values)
    num_participants = len(sequences)
    max_len = max([len(x) for x in sequences])
    padded_sequences = torch.zeros((len(sequences), max_len, len(var) - 1))

    for i, seq in enumerate(sequences):
        for j, step in enumerate(seq):
            padded_sequences[i, j] = torch.tensor(step)

    sgt_features = padded_sequences[:, :, :]
    sgt_targets = padded_sequences[:, :, -4:]

    # create a pseudo mask for the SGT data which is always 1
    sgt_mask = torch.ones(sgt_targets.shape[0], sgt_targets.shape[1], sgt_targets.shape[2])

    # check if there are any NaN values
    if torch.isnan(sgt_features).any() or torch.isnan(sgt_targets).any():
        raise ValueError('There are NaN values in the data!')

    # check for infinite values (very unlikely)
    if torch.isinf(sgt_features).any() or torch.isinf(sgt_targets).any():
        raise ValueError('There are infinite values in the data!')

    print(f'Data preparation has been completed!')
    print(f'We have {num_participants} participants, {max_len} trials per participant, '
          f'and {sgt_features.shape[2]} features.')
    #
    # # =============================================================================
    # # The LSTM model fitting starts here
    # # =============================================================================
    # # Define the model
    # model = LSTM_Fitting(n_layers=[1, 2, 3, 4, 5], n_nodes=[5, 10, 20, 50, 100],
    #                      n_epochs=[100, 200, 400, 600, 800, 1000, 1200],
    #                      batch_size=[8], task='IGTSGT')
    #
    # model.fit(sgt_features, sgt_targets, sgt_mask, './Results/AllResults/SGT')

    # results = read_pickle('./Results/AllResults/SGTresults.pickle')

    # # =============================================================================
    # # Now fit traditional delta and decay models
    # # =============================================================================
    # # Define the models
    # model_delta = ComputationalModels(model_type='delta', task='IGT_SGT', condition='Both')
    # model_decay = ComputationalModels(model_type='decay', task='IGT_SGT', condition='Both')
    #
    # # prepare the data
    # sgt = sgt.dropna(subset=['choice'])
    # sgt['choice'] = (sgt['choice'] - 1).astype(int)
    #
    # sgt_dict = dict_generator(sgt, task='IGT_SGT')
    #
    # # Fit the models
    # delta_results = model_delta.fit(sgt_dict, num_iterations=200, num_feedback=999)
    # decay_results = model_decay.fit(sgt_dict, num_iterations=200, num_feedback=999)
    #
    # # Save the results
    # delta_results.to_csv('./Results/RLResults/SGT_delta_results.csv', index=False)
    # decay_results.to_csv('./Results/RLResults/SGT_decay_results.csv', index=False)
    #
    # # =============================================================================
    # # Post-hoc simulation starts here
    # # =============================================================================
    # # Load the results
    # delta_results = pd.read_csv('./Results/RLResults/SGT_delta_results.csv')
    # decay_results = pd.read_csv('./Results/RLResults/SGT_decay_results.csv')



