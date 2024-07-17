import pandas as pd
import torch
import pickle
from LSTM import LSTM_Fitting, encode_trial_type, find_best_choice, find_trial_type

# =============================================================================
# Load the data
# =============================================================================
if __name__ == '__main__':
    cont_rewards = pd.read_csv("./Data/ABCD_ContRewards.csv")
    ABCD_2019_1 = pd.read_csv("./Data/ABCD_2019_1.csv")
    ABCD_2019_2 = pd.read_csv("./Data/ABCD_2019_2.csv")
    ABCD_2019_3 = pd.read_csv("./Data/ABCD_2019_3.csv")
    ABCD_2022_1 = pd.read_csv("./Data/ABCD_2022_1.csv")
    ABCD_2022_2 = pd.read_csv("./Data/ABCD_2022_2.csv")

    LV = cont_rewards[cont_rewards['Condition'] == 'LV']
    MV = cont_rewards[cont_rewards['Condition'] == 'MV']
    HV = cont_rewards[cont_rewards['Condition'] == 'HV']

    # =============================================================================
    # Prepare the ABCD continuous rewards data initially used for the Dual-Process Model
    # =============================================================================
    dataframes = [LV, MV, HV]
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].reset_index(drop=True)
        dataframes[i].iloc[:, 1] = (dataframes[i].index // 250) + 1
        dataframes[i].rename(
            columns={'X': 'Subnum', 'setSeen': 'SetSeen.', 'choice': 'KeyResponse', 'points': 'Reward'}, inplace=True)
        dataframes[i]['KeyResponse'] = dataframes[i]['KeyResponse'] - 1
        dataframes[i]['SetSeen.'] = dataframes[i]['SetSeen.'] - 1
        dataframes[i]['trial_index'] = dataframes[i].groupby('Subnum').cumcount() + 1

        # standardize the reward using min-max scaling
        dataframes[i]['Reward'] = (dataframes[i]['Reward'] - dataframes[i]['Reward'].min()) / (
                    dataframes[i]['Reward'].max() - dataframes[i]['Reward'].min())

        # set the reward to 0 after 150 trials
        dataframes[i].loc[dataframes[i]['trial_index'] > 150, 'Reward'] = 0

        # add a column to indicate whether the reward is seen by the participant
        dataframes[i]['RewardSeen'] = 1
        dataframes[i].loc[dataframes[i]['trial_index'] > 150, 'RewardSeen'] = 0

        # if reward is not seen, set the reward to NaN
        dataframes[i].loc[dataframes[i]['RewardSeen'] == 0, 'Reward'] = 0

        # Function to encode pairs
        encode_map = {
            'AB': ['A', 'B'],
            'CD': ['C', 'D'],
            'CA': ['C', 'A'],
            'CB': ['C', 'B'],
            'BD': ['B', 'D'],
            'AD': ['A', 'D']
        }

        # Apply the encoding function
        dataframes[i] = encode_trial_type(dataframes[i])

        # use dummy variables for the all the categorical variables
        dataframes[i] = pd.get_dummies(dataframes[i], columns=['KeyResponse'])
        dataframes[i][['KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']] = dataframes[i][
            ['KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']].astype(int)

    # =============================================================================
    # Prepare the ABCD data initially published in Don et al. (2019)
    # =============================================================================
    dataframes_2019 = [ABCD_2019_1, ABCD_2019_2, ABCD_2019_3]

    for i in range(len(dataframes_2019)):
        dataframes_2019[i]['trial_index'] = dataframes_2019[i].groupby('subnum').cumcount() + 1
        dataframes_2019[i]['RewardSeen'] = 1
        dataframes_2019[i].rename(columns={'subnum': 'Subnum', 'optSeen': 'TrialType', 'response': 'KeyResponse',
                                           'reward': 'Reward'}, inplace=True)
        dataframes_2019[i]['KeyResponse'] = (dataframes_2019[i]['KeyResponse'] - 1).astype(int)
        dataframes_2019[i] = encode_trial_type(dataframes_2019[i], letters=False)
        dataframes_2019[i] = pd.get_dummies(dataframes_2019[i], columns=['KeyResponse'])
        dataframes_2019[i][['KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']] = dataframes_2019[i][[
            'KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']].astype(int)

        if i == 2:
            # participants in the 3rd dataset did not see the reward after 150 trials
            dataframes_2019[i].loc[dataframes_2019[i]['trial_index'] > 150, 'RewardSeen'] = 0
            dataframes_2019[i].loc[dataframes_2019[i]['RewardSeen'] == 0, 'reward'] = 0

        if i > 0:
            # participants in the 1st and 2nd dataset completed 50 trials in which they can pick from all 4 options
            # these trials are not of our interest
            dataframes_2019[i] = dataframes_2019[i][dataframes_2019[i]['TrialType'] != 7]

    # =============================================================================
    # Prepare the ABCD data initially published in Don et al. (2022)
    # IMPORTANT: This dataset is not being used because the ABCD task was significantly modified
    # =============================================================================
    # ABCD_2022 = ABCD_2022_1[(ABCD_2022_1['group'] == 1) & (ABCD_2022_1['trialType'] != 7) & (ABCD_2022_1['phase'] != 4)]
    # ABCD_2022.rename(columns={'subject': 'Subnum', 'trialType': 'TrialType', 'choice': 'KeyResponse', 'outcome': 'Reward'}, inplace=True)
    # ABCD_2022['RewardSeen'] = 1
    # ABCD_2022[ABCD_2022['phase'] == 2]['RewardSeen'] = 0
    # ABCD_2022[ABCD_2022['RewardSeen'] == 0]['Reward'] = 0
    # # in addition, we need to delete the AB and CD trials in the transfer phase
    # # ABCD_2022 = ABCD_2022.drop(ABCD_2022[(ABCD_2022['phase'] == 2) & (
    # #         (ABCD_2022['TrialType'] == 1) | (ABCD_2022['TrialType'] == 2))].index)
    # print(ABCD_2022.groupby('Subnum').size().value_counts())
    # ABCD_2022['KeyResponse'] = (ABCD_2022['KeyResponse'] - 1).astype(int)
    # ABCD_2022 = encode_trial_type(ABCD_2022, letters=False, dict=2)
    # ABCD_2022 = pd.get_dummies(ABCD_2022, columns=['KeyResponse'])
    # ABCD_2022[['KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']] = ABCD_2022[[
    #     'KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']].astype(int)

    # =============================================================================
    # Concatenate the dataframes
    # =============================================================================
    # define the variables
    var = ['Subnum', 'Reward', 'RewardSeen', 'Option_A', 'Option_B', 'Option_C', 'Option_D',
           'KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']

    # concatenate the dataframes
    dataframes_250 = dataframes + dataframes_2019
    for i in range(len(dataframes_250)):
        dataframes_250[i] = dataframes_250[i][var]
    df = pd.concat(dataframes_250, ignore_index=True)

    # recalculate the subject numbers
    df['Subnum'] = (df.index // 250) + 1

    # # only needed when using the ABCD_2022 data
    # # find the largest subject number
    # max_subnum = df['Subnum'].max()
    #
    # # concatenate the ABCD_2022 data
    # ABCD_2022 = ABCD_2022[var].reset_index(drop=True)
    # ABCD_2022.loc[:, 'Subnum'] = (ABCD_2022.index // 270) + 1 + max_subnum
    #
    # df = pd.concat([df, ABCD_2022], ignore_index=True)

    # =============================================================================
    # Transform the data to a format that can be used for the LSTM model
    # =============================================================================
    # Group df by subject
    grouped = df.groupby('Subnum').apply(lambda x: x.values.tolist(), include_groups=False)

    sequences = list(grouped.values)
    num_participants = len(sequences)
    max_len = max([len(x) for x in sequences])
    padded_sequences = torch.zeros((len(sequences), max_len, len(var)-1))

    for i, seq in enumerate(sequences):
        for j, step in enumerate(seq):
            padded_sequences[i, j] = torch.tensor(step)

    ABCD_features = padded_sequences[:, :, :]
    ABCD_targets = padded_sequences[:, :, -4:]
    ABCD_mask = padded_sequences[:, :, 2:6]

    # check if there are any NaN values
    if torch.isnan(ABCD_features).any() or torch.isnan(ABCD_targets).any() or torch.isnan(ABCD_mask).any():
        raise ValueError('There are NaN values in the data!')

    # check for infinite values (very unlikely)
    if torch.isinf(ABCD_features).any() or torch.isinf(ABCD_targets).any() or torch.isinf(ABCD_mask).any():
        raise ValueError('There are infinite values in the data!')

    print(f'Data preparation has been completed!')
    print(f'We have {num_participants} participants, {max_len} trials per participant, and {ABCD_features.shape[2]} features.')

    # =============================================================================
    # The LSTM model fitting starts here
    # =============================================================================

    # Define the model
    model = LSTM_Fitting(n_layers=[1, 2, 3, 4, 5], n_nodes=[5, 10, 20, 50, 100],
                         n_epochs=[100, 200, 400, 600, 800, 1000, 1200],
                         batch_size=[2, 4, 8, 16, 32, 64])

    result, MSE, weights = model.fit(ABCD_features, ABCD_targets, ABCD_mask)
    best_result, _, _, _, _ = model.find_best_configuration(result=result, standard='MSE')

    result_path = './Results/AllResults/'
    best_result_path = './Results/BestResults/'

    with open(result_path + 'ABCD.pkl', 'wb') as f:
        pickle.dump(result, f)

    with open(best_result_path + 'ABCD_best.pkl', 'wb') as f:
        pickle.dump(best_result, f)
