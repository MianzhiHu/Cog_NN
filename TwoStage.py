import pandas as pd
import numpy as np
import pyreadstat

# Load the data
exp1_summary, _ = pyreadstat.read_sav('./Data/TwoStage_1/twostagedeval100trial_Exp1.sav')
exp1 = pd.read_csv('./Data/TwoStage_1/TwoStage100_IDVar.txt', sep='\t')

exp_2_summary, _ = pyreadstat.read_sav('./Data/TwoStage_2/twostage200DATAExp2.sav')
exp2 = pd.read_csv('./Data/TwoStage_2/TwoStage200_FinalData.txt', sep='\t')

# =============================================================================
# The data preparation process starts here
# =============================================================================
for data in [exp1, exp2]:
    # recode the reward to be missing if the trial type is 1
    data.loc[data['TrialType'] == 1, 'Reward'] = None
    # recode the response to be 3 and 4 if the state is 1
    data.loc[data['State'] == 1, 'Response'] = data.loc[data['State'] == 1, 'Response'] + 2



