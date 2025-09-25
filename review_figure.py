import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm

palette = sns.color_palette("deep")
font_path = './Abhaya_Libre/AbhayaLibre-SemiBold.ttf'
prop = fm.FontProperties(fname=font_path)
# ======================================================================================================================
# Simulate the delta learning rule
a = 0.5 # Learning rate
EVs = [10, 10]
delta = [10, 10]
decay = [10, 10]
colors = [palette[3], palette[0]]

# Plot initial expected values
plt.figure(figsize=(8, 6))
bars = plt.bar(['Option A', 'Option B'], EVs, color=colors)
plt.ylim(0, 20)
plt.ylabel(fontproperties=prop, ylabel='Expected Value', fontsize=30)
plt.xticks(fontproperties=prop, fontsize=25)
plt.yticks(fontproperties=prop, fontsize=22, ticks=[0, 5, 10, 15, 20])
sns.despine()
plt.tight_layout()
plt.savefig('./Figures/Initial_Expected_Values.png', dpi=600)
plt.show()

# After 5 trials
for _ in range(5):
    delta[0] = delta[0] + a * (10 - delta[0])  # Reward of 10
    decay = [i * (1-a) for i in decay]
    decay[0] = decay[0] + 10  # Reward of 10

# Plot after 5 trials (Delta)
plt.figure(figsize=(8, 6))
bars = plt.bar(['Option A', 'Option B'], delta, color=colors)
plt.ylim(0, 20)
plt.ylabel(fontproperties=prop, ylabel='Expected Value', fontsize=30)
plt.xticks(fontproperties=prop, fontsize=25)
plt.yticks(fontproperties=prop, fontsize=22, ticks=[0, 5, 10, 15, 20])
sns.despine()
plt.tight_layout()
plt.savefig('./Figures/Delta_Expected_Values_After_5_Trials.png', dpi=600)
plt.show()

# Plot after 5 trials (Decay)
plt.figure(figsize=(8, 6))
bars = plt.bar(['Option A', 'Option B'], decay, color=colors)
plt.ylim(0, 20)
plt.ylabel(fontproperties=prop, ylabel='Expected Value', fontsize=30)
plt.xticks(fontproperties=prop, fontsize=25)
plt.yticks(fontproperties=prop, fontsize=22, ticks=[0, 5, 10, 15, 20])
sns.despine()
plt.tight_layout()
plt.savefig('./Figures/Decay_Expected_Values_After_5_Trials.png', dpi=600)
plt.show()

# Simulate delta model with different learning rates
a_list = [0.1, 0.3, 0.5, 0.7, 0.9]
trials = 8
EVs_history = {a: [0] for a in a_list}
legend_prop = fm.FontProperties(fname=prop.get_file(), size=25)
legend_text_prop = fm.FontProperties(fname=prop.get_file(), size=20)

for a in a_list:
    for _ in range(trials):
        EVs_history[a].append(EVs_history[a][-1] + a * (10 - EVs_history[a][-1]))  # Reward of 10
plt.figure(figsize=(10, 6))
for i, a in enumerate(a_list):
    sns.lineplot(x=list(range(trials+1)), y=EVs_history[a], linewidth=4, label='a = '+str(a), color=palette[i])
plt.ylabel(fontproperties=prop, ylabel='Expected Value', fontsize=30)
plt.xticks(fontproperties=prop, fontsize=25)
plt.yticks(fontproperties=prop, fontsize=25)
plt.legend(title="Learning Rate\n        (alpha)", title_fontproperties=legend_prop, prop=legend_text_prop, loc="best")
sns.despine()
plt.tight_layout()
plt.savefig('./Figures/Delta_Model_Learning_Rates_Single.png', dpi=600)
plt.close()

# Simulate delta model with different learning rates
a_pos = 0.3
a_neg = 0.7
a_sym = 0.5
trials = 8
EVs_pos = [0]
EVs_neg = [0]
EV_sym_gains = [0]
EV_sym_losses = [0]
ev_history = {0: [], 1: [], 2: [], 3: []}
ev_history[0].append(EVs_pos[0])
ev_history[1].append(EVs_neg[0])
ev_history[2].append(EV_sym_gains[0])
ev_history[3].append(EV_sym_losses[0])

plt.figure(figsize=(10, 6))
for _ in range(trials):
    EVs_pos[0] = EVs_pos[0] + a_pos * (10 - EVs_pos[0])  # Reward of 10
    EVs_neg[0] = EVs_neg[0] + a_neg * (-10 - EVs_neg[0])  # Loss of -10
    EV_sym_gains[0] = EV_sym_gains[0] + a_sym * (10 - EV_sym_gains[0])  # Reward of 10
    EV_sym_losses[0] = EV_sym_losses[0] + a_sym * (-10 - EV_sym_losses[0])  # Loss of -10
    ev_history[0].append(EVs_pos[0])
    ev_history[1].append(EVs_neg[0])
    ev_history[2].append(EV_sym_gains[0])
    ev_history[3].append(EV_sym_losses[0])
sns.lineplot(x=list(range(trials+1)), y=ev_history[0], linewidth=4, label=f'a = {a_pos} (Gains Asymmetric)', color=palette[1])
sns.lineplot(x=list(range(trials+1)), y=ev_history[1], linewidth=4, label=f'a = {a_neg} (Losses Asymmetric)', color=palette[3])
sns.lineplot(x=list(range(trials+1)), y=ev_history[2], linewidth=4, label=f'a = {a_sym} (Gains Symmetric)', color=palette[2], linestyle='--')
sns.lineplot(x=list(range(trials+1)), y=ev_history[3], linewidth=4, label=f'a = {a_sym} (Losses Symmetric)', color=palette[2], linestyle='--')
plt.xlabel(fontproperties=prop, xlabel='Trial', fontsize=30)
plt.ylabel(fontproperties=prop, ylabel='Expected Value', fontsize=30)
plt.xticks(fontproperties=prop, fontsize=25)
plt.yticks(fontproperties=prop, fontsize=25)
plt.legend(title="Learning Rate\n        (alpha)", title_fontproperties=legend_prop, prop=legend_text_prop, loc="best")
sns.despine()
plt.tight_layout()
plt.savefig('./Figures/Delta_Model_Learning_Rates.png', dpi=600)
plt.show()