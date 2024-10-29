import numpy as np
import pandas as pd
import torch
import pickle
from sympy.codegen import Print
from xarray.util.generate_ops import inplace
from LSTM import LSTM_Fitting, encode_trial_type, find_best_choice, find_trial_type, read_pickle

# =============================================================================
# Load the data
# =============================================================================
if __name__ == '__main__':
    sgt_2022 = pd.read_csv("./Data/SGT_Don_2022.csv")
    sgt_ordered = pd.read_csv("./Data/IGTSGT_OrderData.csv")
    sgt_ordered = sgt_ordered[sgt_ordered['FileName'].str.contains('SCGT')]
    sgt_2022.rename(columns={'subjID': 'Subnum'}, inplace=True)
    sgt = pd.concat([sgt_2022, sgt_ordered], axis=0)

    # =============================================================================
    # Prepare the SGT data
    # =============================================================================
    sgt = sgt.reset_index(drop=True)
    sgt['Subnum'] = (sgt.index // 100) + 1
    sgt['choice'] = sgt['choice'].astype(int)

    # if the participant failed to respond, set the outcome to 0
    sgt.loc[sgt['choice'] == -1, 'outcome'] = 0
    sgt['missed'] = sgt['choice'].apply(lambda x: 0 if x == -1 else 1)
    sgt['choice'] = sgt['choice'].replace(-1, np.nan)

    # now create dummy variables for the choice
    sgt = pd.get_dummies(sgt, columns=['choice'])

    # =============================================================================
    # Transform the data
    # =============================================================================
    # define the variables
    var = ['Subnum', 'outcome', 'missed', 'choice_1.0', 'choice_2.0', 'choice_3.0', 'choice_4.0']
    sgt = sgt[var]

    # Group df by subject
    grouped = sgt.groupby('Subnum').apply(lambda x: x.values.tolist(), include_groups=False)

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

    # =============================================================================
    # The LSTM model fitting starts here
    # =============================================================================
    # Define the model
    # model = LSTM_Fitting(n_layers=[1, 2, 3, 4, 5], n_nodes=[5, 10, 20, 50, 100],
    #                      n_epochs=[100, 200, 400, 600, 800, 1000, 1200],
    #                      batch_size=[8])
    model = LSTM_Fitting(n_layers=[1, 2], n_nodes=[5],
                         n_epochs=[10],
                         batch_size=[8], task='IGTSGT')


    model.fit(sgt_features, sgt_targets, sgt_mask, './Results/AllResults/SGT', max_workers=24)

    results = read_pickle('./Results/AllResults/SGTresults.pickle')