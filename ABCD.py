import pandas as pd
import numpy as np
import torch
import pickle
from statsmodels.iolib.summary import summary
from LSTM import LSTM_Fitting, encode_trial_type, find_best_choice, find_trial_type, read_pickle
from utils.ComputationalModeling import ComputationalModels, dict_generator
pd.options.mode.copy_on_write = True

# =============================================================================
# Load the data
# =============================================================================
if __name__ == '__main__':
    cont_rewards = pd.read_csv("./Data/raw_data/ABCD_ContRewards.csv")
    ABCD_2019_1 = pd.read_csv("./Data/raw_data/ABCD_2019_1.csv")
    ABCD_2019_2 = pd.read_csv("./Data/raw_data/ABCD_2019_2.csv")
    ABCD_2019_3 = pd.read_csv("./Data/raw_data/ABCD_2019_3.csv")
    ABCD_2022_1 = pd.read_csv("./Data/raw_data/ABCD_2022_1.csv")
    ABCD_2022_2 = pd.read_csv("./Data/raw_data/ABCD_2022_2.csv")
    ABCD_2025 = pd.read_csv("./Data/raw_data/ABCD_2025_ID.csv")

    LV = cont_rewards[cont_rewards['Condition'] == 'LV']
    MV = cont_rewards[cont_rewards['Condition'] == 'MV']
    HV = cont_rewards[cont_rewards['Condition'] == 'HV']

    # =============================================================================
    # Prepare the ABCD continuous rewards data initially used for the Dual-Process Model
    # =============================================================================
    dataframes = [HV]
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].reset_index(drop=True)
        dataframes[i].iloc[:, 1] = (dataframes[i].index // 250) + 1
        dataframes[i].rename(
            columns={'X': 'Subnum', 'setSeen': 'SetSeen.', 'choice': 'KeyResponse', 'points': 'Reward'}, inplace=True)
        dataframes[i]['KeyResponse'] = dataframes[i]['KeyResponse'] - 1
        dataframes[i]['SetSeen.'] = dataframes[i]['SetSeen.'] - 1
        dataframes[i]['trial_index'] = dataframes[i].groupby('Subnum').cumcount() + 1
        dataframes[i]['source'] = 'dp'

        # add a column to indicate whether the reward is seen by the participant
        dataframes[i]['RewardSeen'] = 1
        dataframes[i].loc[dataframes[i]['trial_index'] > 150, 'RewardSeen'] = 0

        # if reward is not seen, set the reward to NaN
        dataframes[i].loc[dataframes[i]['RewardSeen'] == 0, 'Reward'] = 0

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
        dataframes_2019[i]['RewardSeen'] = 1
        dataframes_2019[i].rename(columns={'subnum': 'Subnum', 'optSeen': 'TrialType', 'response': 'KeyResponse',
                                           'reward': 'Reward'}, inplace=True)
        # participants in the 1st and 2nd dataset completed 50 trials in which they can pick from all 4 options,
        # but these trials are not of our interest
        dataframes_2019[i] = dataframes_2019[i][dataframes_2019[i]['TrialType'] != 7]
        dataframes_2019[i]['trial_index'] = dataframes_2019[i].groupby('Subnum').cumcount() + 1
        dataframes_2019[i]['source'] = f'2019_{i+1}'
        dataframes_2019[i]['KeyResponse'] = (dataframes_2019[i]['KeyResponse'] - 1).astype(int)
        dataframes_2019[i] = encode_trial_type(dataframes_2019[i], letters=False)
        dataframes_2019[i] = pd.get_dummies(dataframes_2019[i], columns=['KeyResponse'])
        dataframes_2019[i][['KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']] = dataframes_2019[i][[
            'KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']].astype(int)

        if i == 2:
            # participants in the 3rd dataset did not see the reward after 150 trials
            dataframes_2019[i].loc[dataframes_2019[i]['trial_index'] > 150, 'RewardSeen'] = 0
            dataframes_2019[i].loc[dataframes_2019[i]['RewardSeen'] == 0, 'Reward'] = 0


    # =============================================================================
    # Prepare the ABCD data from the individual difference study (unpublished 2025)
    # =============================================================================
    # This is a 200 trial version of the ABCD task with 120 training trials and 80 transfer trials
    ABCD_2025 = ABCD_2025[ABCD_2025['Condition'] == 'Frequency'].reset_index() # only keep the frequency condition
    ABCD_2025.rename(columns={'ReactTime': 'RT'}, inplace=True)
    ABCD_2025['Subnum'] = (ABCD_2025.index // 200) + 1

    # participants did not see the reward after 120 trials
    ABCD_2025['trial_index'] = ABCD_2025.groupby('Subnum').cumcount() + 1
    ABCD_2025['RewardSeen'] = 1
    ABCD_2025['source'] = '2025_ID'
    ABCD_2025.loc[ABCD_2025['trial_index'] > 120, 'RewardSeen'] = 0
    ABCD_2025.loc[ABCD_2025['RewardSeen'] == 0, 'Reward'] = 0

    # Recode the trial types
    ABCD_2025 = encode_trial_type(ABCD_2025, letters=True)
    ABCD_2025['KeyResponse'] = (ABCD_2025['KeyResponse'] - 1).astype(int)
    ABCD_2025 = pd.get_dummies(ABCD_2025, columns=['KeyResponse'])
    ABCD_2025[['KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']] = ABCD_2025[[
        'KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']].astype(int)

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
    var = ['Subnum', 'trial_index', 'RT', 'Reward', 'RewardSeen', 'Option_A', 'Option_B', 'Option_C', 'Option_D',
           'KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3', 'source']

    # concatenate the dataframes
    dataframes_250 = dataframes + dataframes_2019
    for i in range(len(dataframes_250)):
        dataframes_250[i] = dataframes_250[i][var]
    df = pd.concat(dataframes_250, ignore_index=True).reset_index(drop=True)

    # recalculate the subject numbers
    df['Subnum'] = (df.index // 250) + 1

    # record the maximum participant number
    n_participant = max(df['Subnum'])
    print(df.groupby('Subnum').size().value_counts())

    # now add the 2025 data
    ABCD_2025['Subnum'] = ABCD_2025['Subnum'] + n_participant
    df = pd.concat([df, ABCD_2025[var]], ignore_index=True)
    print(f'We have {df["Subnum"].nunique()} participants in total.')

    # Verify RewardSeen=0 implies Reward=0 for all participants
    invalid_rows = df[(df['RewardSeen'] == 0) & (df['Reward'] != 0)]
    if len(invalid_rows) > 0:
        raise ValueError(f'Found {len(invalid_rows)} rows where RewardSeen=0 but Reward≠0')
    print('Verified: All rows with RewardSeen=0 have Reward=0')

    # Check if RewardSeen is all 1 in 2019 Experiment 1 and 2
    print(df.groupby('source')['RewardSeen'].value_counts())

    # Check the maximum trial numbers per source
    print(df.groupby('source')['trial_index'].max())

    df.to_csv('./Data/processed_data/ABCD_total.csv', index=False)

    # # only needed when using the ABCD_2022 data
    # # find the largest subject number
    # max_subnum = df['Subnum'].max()
    #
    # # concatenate the ABCD_2022 data
    # ABCD_2022 = ABCD_2022[var].reset_index(drop=True)
    # ABCD_2022.loc[:, 'Subnum'] = (ABCD_2022.index // 270) + 1 + max_subnum
    #
    # df = pd.concat([df, ABCD_2022], ignore_index=True)

    # # =============================================================================
    # # Transform the data to a format that can be used for the LSTM model
    # # =============================================================================
    # # Group df by subject
    # grouped = df.groupby('Subnum').apply(lambda x: x.values.tolist(), include_groups=False)
    #
    # sequences = list(grouped.values)
    # num_participants = len(sequences)
    # max_len = max([len(x) for x in sequences])
    # padded_sequences = torch.zeros((len(sequences), max_len, len(var) - 1))
    #
    # for i, seq in enumerate(sequences):
    #     for j, step in enumerate(seq):
    #         padded_sequences[i, j] = torch.tensor(step)
    #
    # ABCD_features = padded_sequences[:, :, :]
    # ABCD_targets = padded_sequences[:, :, -4:]
    # ABCD_mask = padded_sequences[:, :, 2:6]
    #
    # # check if there are any NaN values
    # if torch.isnan(ABCD_features).any() or torch.isnan(ABCD_targets).any() or torch.isnan(ABCD_mask).any():
    #     raise ValueError('There are NaN values in the data!')
    #
    # # check for infinite values (very unlikely)
    # if torch.isinf(ABCD_features).any() or torch.isinf(ABCD_targets).any() or torch.isinf(ABCD_mask).any():
    #     raise ValueError('There are infinite values in the data!')
    #
    # print(f'Data preparation has been completed!')
    # print(
    #     f'We have {num_participants} participants, {max_len} trials per participant, and {ABCD_features.shape[2]} features.')

    # # =============================================================================
    # # The LSTM model fitting starts here
    # # =============================================================================
    # # Define the model
    # model = LSTM_Fitting(n_layers=[1, 2, 3, 4, 5], n_nodes=[5, 10, 20, 50, 100],
    #                      n_epochs=[100, 200, 400, 600, 800, 1000, 1200],
    #                      batch_size=[8], task='ABCD')
    #
    #
    # model.fit(ABCD_features, ABCD_targets, ABCD_mask, './Results/AllResults/ABCD')
    #
    # # =============================================================================
    # # Read the results
    # # =============================================================================
    # results = read_pickle('./Results/AllResults/ABCDresults.pickle')

    # # =============================================================================
    # # Now fit traditional delta and decay models
    # # =============================================================================
    # # Define the models
    # model_delta = ComputationalModels(model_type='delta')
    # model_decay = ComputationalModels(model_type='decay')
    #
    # # With feedback
    # ABCD_with_feedback = pd.concat(dataframes_2019[0:2], ignore_index=True, axis=0).reset_index(drop=True)
    # ABCD_without_feedback = dataframes_2019[2].reset_index(drop=True)
    #
    # # reverse the one-hot encoding
    # for i, data in enumerate([ABCD_with_feedback, ABCD_without_feedback]):
    #     data['KeyResponse'] = data[['KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2',
    #                                 'KeyResponse_3']].idxmax(axis=1).str[-1].astype(int)
    #     data.rename(columns={'TrialType': 'SetSeen.'}, inplace=True)
    #     data['SetSeen.'] = data['SetSeen.'] - 1
    #     data['Subnum'] = (data.index // 250) + 1
    #
    # ABCD_with_feedback_dict = dict_generator(ABCD_with_feedback)
    # ABCD_without_feedback_dict = dict_generator(ABCD_without_feedback)

    # # Fit the models
    # delta_results_fb = model_delta.fit(ABCD_with_feedback_dict, num_iterations=200, num_feedback=250)
    # decay_results_fb = model_decay.fit(ABCD_with_feedback_dict, num_iterations=200, num_feedback=250)
    # delta_results_nofb = model_delta.fit(ABCD_without_feedback_dict, num_iterations=200)
    # decay_results_nofb = model_decay.fit(ABCD_without_feedback_dict, num_iterations=200)
    #
    # # add the condition to the results
    # delta_results_fb['Condition'] = 'FB'
    # decay_results_fb['Condition'] = 'FB'
    # delta_results_nofb['Condition'] = 'NoFB'
    # decay_results_nofb['Condition'] = 'NoFB'
    #
    # delta_results = pd.concat([delta_results_fb, delta_results_nofb], axis=0).reset_index(drop=True)
    # decay_results = pd.concat([decay_results_fb, decay_results_nofb], axis=0).reset_index(drop=True)
    #
    # # read in the fitting results from Hu et al.(2024)
    # delta_HV = pd.read_csv('./Results/RLResults/delta_HV_results.csv')
    # decay_HV = pd.read_csv('./Results/RLResults/decay_HV_results.csv')
    # delta_MV = pd.read_csv('./Results/RLResults/delta_MV_results.csv')
    # decay_MV = pd.read_csv('./Results/RLResults/decay_MV_results.csv')
    # delta_LV = pd.read_csv('./Results/RLResults/delta_LV_results.csv')
    # decay_LV = pd.read_csv('./Results/RLResults/decay_LV_results.csv')
    #
    # # add the condition to the results
    # for df in [delta_HV, decay_HV]:
    #     df['Condition'] = 'HV'
    # for df in [delta_MV, decay_MV]:
    #     df['Condition'] = 'MV'
    # for df in [delta_LV, decay_LV]:
    #     df['Condition'] = 'LV'
    #
    # # concatenate the results
    # delta_results = pd.concat([delta_LV, delta_MV, delta_HV, delta_results], axis=0).reset_index(drop=True)
    # decay_results = pd.concat([decay_LV, decay_MV, decay_HV, decay_results], axis=0).reset_index(drop=True)
    #
    # # reset the participant numbers
    # delta_results['participant_id'] = delta_results.index + 1
    # decay_results['participant_id'] = decay_results.index + 1
    #
    # # save the results
    # delta_results.to_csv('./Results/RLResults/delta_results.csv', index=False)
    # decay_results.to_csv('./Results/RLResults/decay_results.csv', index=False)

    # # =============================================================================
    # # Post-hoc simulation starts here
    # # =============================================================================
    # # Load the results
    # delta_results = pd.read_csv('./Results/RLResults/delta_results.csv')
    # decay_results = pd.read_csv('./Results/RLResults/decay_results.csv')
    #
    # # Separate the results again into conditions
    # results = {'delta': delta_results, 'decay': decay_results}
    # conditions = ['LV', 'MV', 'HV', 'FB', 'NoFB']
    # result_dfs = {}
    # for key, result in results.items():
    #     for condition in conditions:
    #         result_dfs[f'{key}_{condition}'] = result[result['Condition'] == condition].reset_index(drop=True)
    #         result_dfs[f'{key}_{condition}']['participant_id'] = result_dfs[f'{key}_{condition}'].index + 1
    #
    # # define means and standard deviations
    # reward_means = [0.65, 0.35, 0.75, 0.25]
    # reward_var_HV = [0.48, 0.48, 0.43, 0.43]
    # reward_var_MV = [0.24, 0.24, 0.22, 0.22]
    # reward_var_LV = [0.12, 0.12, 0.11, 0.11]
    #
    # # post-hoc simulations for the delta model
    # delta_posthoc_LV = model_delta.post_hoc_simulation(result_dfs['delta_LV'], dataframes[0], reward_means,
    #                                                    reward_var_LV, trial_sequence_option='ori',
    #                                                    reward_sampling='normal', summary=True, num_iterations=1000)
    # delta_posthoc_MV = model_delta.post_hoc_simulation(result_dfs['delta_MV'], dataframes[1], reward_means,
    #                                                    reward_var_MV, trial_sequence_option='ori',
    #                                                    reward_sampling='normal', summary=True, num_iterations=1000)
    # delta_posthoc_HV = model_delta.post_hoc_simulation(result_dfs['delta_HV'], dataframes[2], reward_means,
    #                                                    reward_var_HV, trial_sequence_option='ori',
    #                                                    reward_sampling='normal', summary=True, num_iterations=1000)
    # delta_posthoc_FB = model_delta.post_hoc_simulation(result_dfs['delta_FB'], ABCD_with_feedback, reward_means,
    #                                                    reward_var_HV, trial_sequence_option='ori', num_feedback=250,
    #                                                    reward_sampling='binary', summary=True, num_iterations=1000)
    # delta_posthoc_NoFB = model_delta.post_hoc_simulation(result_dfs['delta_NoFB'], ABCD_without_feedback, reward_means,
    #                                                      reward_var_HV, trial_sequence_option='ori',
    #                                                      reward_sampling='binary', summary=True, num_iterations=1000)
    #
    # # post-hoc simulations for the decay model
    # decay_posthoc_LV = model_decay.post_hoc_simulation(result_dfs['decay_LV'], dataframes[0], reward_means,
    #                                                    reward_var_LV, trial_sequence_option='ori',
    #                                                    reward_sampling='normal', summary=True, num_iterations=1000)
    # decay_posthoc_MV = model_decay.post_hoc_simulation(result_dfs['decay_MV'], dataframes[1], reward_means,
    #                                                    reward_var_MV, trial_sequence_option='ori',
    #                                                    reward_sampling='normal', summary=True, num_iterations=1000)
    # decay_posthoc_HV = model_decay.post_hoc_simulation(result_dfs['decay_HV'], dataframes[2], reward_means,
    #                                                    reward_var_HV, trial_sequence_option='ori',
    #                                                    reward_sampling='normal', summary=True, num_iterations=1000)
    # decay_posthoc_FB = model_decay.post_hoc_simulation(result_dfs['decay_FB'], ABCD_with_feedback, reward_means,
    #                                                    reward_var_HV, trial_sequence_option='ori', num_feedback=250,
    #                                                    reward_sampling='binary', summary=True, num_iterations=1000)
    # decay_posthoc_NoFB = model_decay.post_hoc_simulation(result_dfs['decay_NoFB'], ABCD_without_feedback, reward_means,
    #                                                      reward_var_HV, trial_sequence_option='ori',
    #                                                      reward_sampling='binary', summary=True, num_iterations=1000)
    #
    # # concatenate the results
    # delta_posthoc = pd.concat([delta_posthoc_LV, delta_posthoc_MV, delta_posthoc_HV, delta_posthoc_FB,
    #                            delta_posthoc_NoFB], axis=0).reset_index(drop=True)
    # delta_posthoc['Subnum'] = delta_posthoc.index // 250 + 1
    #
    #
    # decay_posthoc = pd.concat([decay_posthoc_LV, decay_posthoc_MV, decay_posthoc_HV, decay_posthoc_FB,
    #                            decay_posthoc_NoFB], axis=0).reset_index(drop=True)
    # decay_posthoc['Subnum'] = decay_posthoc.index // 250 + 1
    #
    # # save the results
    # delta_posthoc.to_csv('./Results/RLResults/delta_posthoc_results.csv', index=False)
    # decay_posthoc.to_csv('./Results/RLResults/decay_posthoc_results.csv', index=False)
    #
    # # =============================================================================
    # # Concatenate all the results
    # # =============================================================================
    # delta_results = pd.read_csv('./Results/RLResults/delta_results.csv')
    # decay_results = pd.read_csv('./Results/RLResults/decay_results.csv')
    # delta_posthoc = pd.read_csv('./Results/RLResults/delta_posthoc_results.csv')
    # decay_posthoc = pd.read_csv('./Results/RLResults/decay_posthoc_results.csv')




