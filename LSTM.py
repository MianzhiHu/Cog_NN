import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from itertools import product


# define the LSTM model
class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num):
        super().__init__()
        self.lstmLayer = nn.LSTM(in_dim, hidden_dim, layer_num)
        self.relu = nn.ReLU()
        self.fcLayer = nn.Linear(hidden_dim, out_dim)
        self.weightInit = np.sqrt(1.0 / hidden_dim)

    def forward(self, x, mask):
        out, _ = self.lstmLayer(x)
        out = self.relu(out)
        out = self.fcLayer(out)
        # set the unavailable options to -inf so that the softmax function will ignore them
        out = torch.where(mask == 1, out, torch.tensor(float('-inf')))
        out = nn.Softmax(dim=-1)(out)
        return out


class LSTM_Fitting():
    def __init__(self, n_layers, n_nodes, n_epochs, batch_size):
        self.param_grid = {
            'n_layers': n_layers,
            'n_nodes': n_nodes,
            'n_epochs': n_epochs,
            'batch_size': batch_size
        }

    def fit(self, features, targets, mask, n_folds=5, output_dim=4):
        # define unchangeable parameters
        lag = 1  # the model prediction starts from the second trial
        n_folds = n_folds
        iteration = 0

        # initialize dictionaries to store the results
        results_dict = {}
        MSE_dict = {}
        weights_dict = {}

        # define the model
        for n_nodes, n_layers, n_epochs, batch_size in product(self.param_grid['n_nodes'],
                                                               self.param_grid['n_layers'],
                                                               self.param_grid['n_epochs'],
                                                               self.param_grid['batch_size']):
            iteration += 1

            print('Iteration [{}/{}]'.format(iteration, len(list(product(self.param_grid['n_nodes'],
                                                                         self.param_grid['n_layers'],
                                                                         self.param_grid['n_epochs'],
                                                                         self.param_grid['batch_size'])))))
            print('Current Configuration:')
            print(f'Number of Layers: {n_layers}, Number of Nodes: {n_nodes}, Number of Epochs: {n_epochs}, '
                  f'Batch Size: {batch_size}')

            model = LSTM(features.shape[2], n_nodes, output_dim, n_layers)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

            # Temporary DataFrame for current configuration
            LSTM_result = pd.DataFrame()

            # split the data into n_folds
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            # iterate over the folds
            for fold, (train_index, test_index) in enumerate(kf.split(features)):
                # split the data
                X_train, X_test = features[train_index], features[test_index]
                y_train, y_test = targets[train_index], targets[test_index]
                mask_train, mask_test = mask[train_index], mask[test_index]

                losses = []

                for epoch in np.arange(n_epochs):
                    # randomize the order of the participants
                    participant_indices = np.random.permutation(X_train.shape[0])

                    for i in range(0, len(participant_indices), batch_size):
                        batch_indices = participant_indices[i:i + batch_size]

                        # Ensure to not break the sequence for each participant
                        X_batch = X_train[batch_indices].float()
                        y_batch = y_train[batch_indices].float()
                        mask_batch = mask_train[batch_indices].float()

                        optimizer.zero_grad()
                        output = model(X_batch, mask_batch)
                        loss = criterion(output[:, :-lag], y_batch[:, lag:])
                        loss.backward()

                        on_policy_loss = loss.item()
                        losses.append(on_policy_loss)

                        optimizer.step()

                        # print the number of folds
                        if i % 10 == 0:
                            print(f'Fold: {fold + 1}/{n_folds}')
                            print('Epoch[{}/{}], Batch[{}/{}], Loss: {:.5f}'.format(
                                epoch + 1, n_epochs, (i / 10) + 1, X_train.shape[0] / batch_size, on_policy_loss))

                # evaluate
                model_eval = model.eval()
                y_pred = model_eval(X_test, mask_test).data.cpu().numpy()
                weights = weight_storing(model)

                if fold == 0:
                    test_set_full = y_test
                    pred_set_full = y_pred
                    MSEloss = losses
                    weights_full = weights
                else:
                    test_set_full = np.concatenate((test_set_full, y_test), axis=0)
                    pred_set_full = np.concatenate((pred_set_full, y_pred), axis=0)
                    MSEloss = np.concatenate((MSEloss, losses), axis=0)
                    weights_full = pd.concat([weights_full, weights], axis=0)

            # =============================================================================
            # The model fitting process ends here
            # =============================================================================

            # convert the results back to the original format
            for results in [test_set_full, pred_set_full]:
                participant = []
                trial_index = []
                outcome = []
                for i in range(results.shape[0]):
                    for j in range(results.shape[1]):
                        participant.append(i)
                        trial_index.append(j)
                        outcome.append(results[i, j])

                # create a df for the current fold
                fold_results = pd.DataFrame(
                    {'Subnum': participant, 'trial_index': trial_index, 'real_y': outcome})

                if LSTM_result.empty:
                    LSTM_result = fold_results
                else:
                    LSTM_result = pd.merge(LSTM_result,
                                           pd.DataFrame({'Subnum': participant, 'trial_index': trial_index,
                                                         'pred_y': outcome}), on=['Subnum', 'trial_index'])

            # now, get the trial type first
            LSTM_result['TrialType'] = LSTM_result['pred_y'].apply(find_trial_type)

            # get people's actual choices, as well as the predicted choices
            LSTM_result['bestOption'] = LSTM_result.apply(find_best_choice, axis=1, target_variable='real_y')
            LSTM_result['pred_bestOption'] = LSTM_result.apply(find_best_choice, axis=1,
                                                               target_variable='pred_y')

            results_dict[(n_layers, n_nodes, n_epochs, batch_size)] = LSTM_result
            MSE_dict[(n_layers, n_nodes, n_epochs, batch_size)] = MSEloss
            weights_dict[(n_layers, n_nodes, n_epochs, batch_size)] = weights_full

        return results_dict, MSE_dict, weights_dict, participant_indices

    def find_best_configuration(self, result, standard='MSE', print_results=True):
        # initialize the best configuration
        best_MSE = 1000
        best_MAE = 1000
        best_percent_correct = 0
        best_config = None

        # calculate the metrics for each configuration
        for key, value in result.items():
            value['squared_error'] = (value['bestOption'] - value['pred_bestOption']) ** 2
            value['absolute_error'] = np.abs(value['bestOption'] - value['pred_bestOption'])
            MSE = np.mean(value['squared_error'])
            MAE = np.mean(value['absolute_error'])
            value['binary_pred'] = np.where(value['pred_bestOption'] > 0.5, 1, 0)
            value['correct_pred'] = np.where(value['bestOption'] == value['binary_pred'], 1, 0)
            percent_correct = value['correct_pred'].mean()
            if print_results:
                print(
                    f'Number of Layers: {key[0]}, Number of Nodes: {key[1]}, Number of Epochs: {key[2]}, Batch Size: {key[3]}')
                print(f'MSE: {MSE}')
                print(f'MAE: {MAE}')
                print(f'Percent Correct: {percent_correct}')
                print('------------------------------------')
                print('\n')

            if standard == 'MSE':
                if MSE < best_MSE:
                    best_MSE = MSE
                    best_MAE = MAE
                    best_percent_correct = percent_correct
                    best_config = key

            elif standard == 'MAE':
                if MAE < best_MAE:
                    best_MSE = MSE
                    best_MAE = MAE
                    best_percent_correct = percent_correct
                    best_config = key

            elif standard == 'percent_correct':
                if percent_correct > best_percent_correct:
                    best_MSE = MSE
                    best_MAE = MAE
                    best_percent_correct = percent_correct
                    best_config = key

        print(f'The best configuration is: {best_config}')
        print(f'The best MSE is: {best_MSE}')
        print(f'The best MAE is: {best_MAE}')
        print(f'The best percent correct is: {best_percent_correct}')

        # find the best configuration
        best_result = result[best_config]

        return best_result, best_config, best_MSE, best_MAE, best_percent_correct


# define some needed functions
def encode_trial_type(df, letters=True, dict=1):
    # Create dummy columns for each option
    df.loc[:, 'Option_A'] = 0
    df.loc[:, 'Option_B'] = 0
    df.loc[:, 'Option_C'] = 0
    df.loc[:, 'Option_D'] = 0

    if letters:
        # Iterate through the DataFrame to set dummies
        for index, row in df.iterrows():
            if 'A' in row['TrialType']:
                df.at[index, 'Option_A'] = 1
            if 'B' in row['TrialType']:
                df.at[index, 'Option_B'] = 1
            if 'C' in row['TrialType']:
                df.at[index, 'Option_C'] = 1
            if 'D' in row['TrialType']:
                df.at[index, 'Option_D'] = 1

    if not letters:
        for index, row in df.iterrows():
            if dict == 1:
                if row['TrialType'] == 1:
                    df.at[index, 'Option_A'] = 1
                    df.at[index, 'Option_B'] = 1
                elif row['TrialType'] == 2:
                    df.at[index, 'Option_C'] = 1
                    df.at[index, 'Option_D'] = 1
                elif row['TrialType'] == 3:
                    df.at[index, 'Option_A'] = 1
                    df.at[index, 'Option_C'] = 1
                elif row['TrialType'] == 4:
                    df.at[index, 'Option_B'] = 1
                    df.at[index, 'Option_C'] = 1
                elif row['TrialType'] == 5:
                    df.at[index, 'Option_A'] = 1
                    df.at[index, 'Option_D'] = 1
                elif row['TrialType'] == 6:
                    df.at[index, 'Option_B'] = 1
                    df.at[index, 'Option_D'] = 1

            if dict == 2:
                if row['TrialType'] == 1:
                    df.at[index, 'Option_A'] = 1
                    df.at[index, 'Option_B'] = 1
                elif row['TrialType'] == 2:
                    df.at[index, 'Option_C'] = 1
                    df.at[index, 'Option_D'] = 1
                elif row['TrialType'] == 3:
                    df.at[index, 'Option_A'] = 1
                    df.at[index, 'Option_C'] = 1
                elif row['TrialType'] == 4:
                    df.at[index, 'Option_A'] = 1
                    df.at[index, 'Option_D'] = 1
                elif row['TrialType'] == 5:
                    df.at[index, 'Option_B'] = 1
                    df.at[index, 'Option_C'] = 1
                elif row['TrialType'] == 6:
                    df.at[index, 'Option_B'] = 1
                    df.at[index, 'Option_D'] = 1

    return df


def find_trial_type(row):  # Function to convert the one-hot encoding back to the trial type
    options = ['A', 'B', 'C', 'D']
    trial_type = ''.join([options[i] for i in range(len(row)) if row[i] > 0])
    return trial_type


def find_best_choice(row, target_variable):  # Function to extract the value of the best choice from the numpy array

    best_choice_mapping = {
        'AB': 0,
        'CD': 2,
        'AC': 2,
        'AD': 0,
        'BC': 2,
        'BD': 1,
    }

    # extract the value from the position to indicate the best choice
    best_choice_position = best_choice_mapping[row['TrialType']]
    best_choice = row[target_variable][best_choice_position]
    return best_choice


def weight_storing(current_model):
    # preallocate a dictionary to store the weights
    temp_weights = {
        'Parameter Name': [],
        'Size': [],
        'Values': []
    }

    # Iterate over state_dict to collect parameter names and values
    for param_name, param_values in current_model.state_dict().items():
        temp_weights['Parameter Name'].append(param_name)
        temp_weights['Size'].append(param_values.size())
        # Flatten the tensor values and convert them to a list
        temp_weights['Values'].append(param_values.cpu().numpy().flatten().tolist())

    return pd.DataFrame(temp_weights)


def average_weights(values):
    # convert the list of weights to a dataframe
    expanded = values['Values'].apply(pd.Series)
    expanded['Parameter Name'] = values['Parameter Name']
    mean_df = expanded.groupby('Parameter Name').mean().reset_index()
    return mean_df