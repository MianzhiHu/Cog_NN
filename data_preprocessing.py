import os
import numpy as np
import pandas as pd

data_dir = f'{os.getcwd()}/ds004636/'
possible_sessions = ["ses-2", "ses-1"]
# ======================================================================================================================
# ANT (N = 103)
# ======================================================================================================================
ant_data = []
ant_col = ['correct', 'cue', 'flanker_location', 'flanker_middle_direction', 'response_time', 'flanker_type', 'worker_id']
i = 0

for sub_dir in os.listdir(data_dir):
    if not sub_dir.startswith("sub-"):
        continue

    found_ant = False

    for ses in possible_sessions:
        ant_dir = os.path.join(data_dir, sub_dir, ses, "func")

        if not os.path.isdir(ant_dir):
            continue

        for file in os.listdir(ant_dir):
            if file.endswith(".tsv") and "ANT" in file:
                file_path = os.path.join(ant_dir, file)

                df = pd.read_csv(file_path, sep="\t")
                df = df[ant_col]
                df = df.groupby(df.index // 2).first().reset_index(drop=True)
                # Replace NaN in cues with 'spatial' (there was only one NaN in cues, and it should be a spatial cue)
                df['cue'] = df['cue'].fillna('spatial')
                # Replace RT < 0 with NaN
                df['response_time'] = df['response_time'].apply(lambda x: np.nan if x < 0 else x)
                # If RT is NaN, add an indication column (NaN = 1, not NaN = 0)
                df['rt_nan'] = df['response_time'].isna().astype(int)
                # Now transform original NaN RT to be the maximal possible RT by design (2.1s)
                df['response_time'] = df['response_time'].fillna(2.10)
                ant_data.append(df)

                found_ant = True
                break

        if found_ant:
            break

    if not found_ant:
        print(f"Participant {sub_dir} does not have ANT file in ses-2 or ses-1.")

ant_data = pd.concat(ant_data, ignore_index=True)
ant_data.to_csv('./Data/hypernetwork_data/ant_data.csv', index=False)
print(f'Total ANT data shape: {ant_data.shape}; number of unique participants: {ant_data["worker_id"].nunique()}')
print(f'ANT data contains NaN: {ant_data.isnull().values.any()}')

# ======================================================================================================================
# Delayed Discounting (N = 99)
# ======================================================================================================================
dd_data = []
dd_col = ['choice', 'large_amount', 'later_delay', 'small_amount', 'response_time', 'rt_nan', 'worker_id']

for sub_dir in os.listdir(data_dir):
    if not sub_dir.startswith("sub-"):
        continue

    found_dd = False

    for ses in possible_sessions:
        dd_dir = os.path.join(data_dir, sub_dir, ses, "func")

        if not os.path.isdir(dd_dir):
            continue

        for file in os.listdir(dd_dir):
            if file.endswith(".tsv") and "discountFix" in file:
                file_path = os.path.join(dd_dir, file)
                df = pd.read_csv(file_path, sep='\t')
                # Replace RT < 0 with NaN
                df['response_time'] = df['response_time'].apply(lambda x: np.nan if x < 0 else x)
                # If RT is NaN, add an indication column (NaN = 1, not NaN = 0)
                df['rt_nan'] = df['response_time'].isna().astype(int)
                # Calculate the allowed RT window by subtracting the onset time for the next trial by the onset time for the current trial
                df['allowed_rt'] = df['onset'].shift(-1) - df['onset']
                df['allowed_rt'] = df['allowed_rt'].fillna(5.00)  # For the last trial, we can set the allowed RT to be the expected window (5 sec)
                # Now transform original NaN RT to be allowed RT
                df['response_time'] = df['response_time'].fillna(df['allowed_rt'])
                df = df[dd_col]
                dd_data.append(df)

                found_dd = True
                break

        if found_dd:
            break

    if not found_dd:
        print(f"Participant {sub_dir} does not have discountFix file in ses-2 or ses-1.")

dd_data = pd.concat(dd_data, ignore_index=True)
dd_data.to_csv('./Data/hypernetwork_data/dd_data.csv', index=False)
print(f'Total Delayed Discounting data shape: {dd_data.shape}; number of unique participants: {dd_data["worker_id"].nunique()}')
print(f'Delayed Discounting data contains NaN: {dd_data.isnull().values.any()}')