import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from HyperNetwork import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16

# ANT
ant_data = pd.read_csv('./Data/hypernetwork_data/ant_data.csv')

# Inspect the data
for col in ant_data.columns:
    print(col)
    print(ant_data[col].value_counts())

baseline_acc = ant_data["correct"].mean()
print(baseline_acc)
baseline_nll = -(ant_data["correct"] * np.log(baseline_acc) + (1 - ant_data["correct"]) * np.log(1 - baseline_acc)).mean()
print(baseline_nll)

# Reassign participant IDs to be consecutive integers starting from 0
ant_data["participant_id"] = ant_data["worker_id"].astype("category").cat.codes
n_participants = ant_data["participant_id"].nunique()
print(n_participants)

# Add trial index
ant_data["trial"] = ant_data.groupby("participant_id").cumcount() + 1
ant_data = ant_data.sort_values(["participant_id", "trial"]).reset_index(drop=True)

# Encode categorical variables
# (this dataset does not have more than 2 categories for each variable, so we can simply use binary encoding)
ant_cat_col = ['cue', 'flanker_location', 'flanker_middle_direction', 'flanker_type']
y_col = 'correct'
for col in ant_cat_col:
    ant_data[col] = ant_data[col].astype("category").cat.codes.astype("float32")

# Make sure y is integer 0/1
ant_data["correct"] = ant_data["correct"].astype(int)

# Standardizer RT
scaler = StandardScaler()
ant_data['response_time'] = scaler.fit_transform(ant_data[['response_time']])

# Split train, validation, and test data
train_df, val_df, test_df = split_by_participant_trials(ant_data)

# Transform into sequences
train_seq = BehavioralDataset(train_df, x_var=ant_cat_col, y_var=y_col)
val_seq = BehavioralDataset(val_df, x_var=ant_cat_col, y_var=y_col)
test_seq = BehavioralDataset(test_df, x_var=ant_cat_col, y_var=y_col)

train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True, collate_fn=behavioral_collate_fn)
val_loader = DataLoader(val_seq, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)
test_loader = DataLoader(test_seq, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)

# Define model-general hyperparameters
n_epochs = 100
lr = 1e-3
patience = 20

# ======================================================================================================================
# Model Training (Just reuse the code and change the model initialization to fit basic RNN and LSTM)
# ======================================================================================================================
participant_emb_dims = [2, 4, 8, 16, 32]
hyper_hidden_dims = [2, 4, 8, 16, 32, 64]
hidden_dims = [2, 4, 8, 16, 32, 64]

results = []

best_val_loss = np.inf
best_config = None
best_model = None
best_history = None

save_dir = './Results/NN_Results/hyper_rnn'
os.makedirs(save_dir, exist_ok=True)

for participant_emb_dim, hyper_hidden_dim, hidden_dim in itertools.product(participant_emb_dims, hyper_hidden_dims, hidden_dims):
    print("=" * 80)
    print(f"Training HyperRNN | participant_emb_dim={participant_emb_dim}, hyper_hidden_dim={hyper_hidden_dim}, hidden_dim={hidden_dim}")

    hyper_rnn = HyperRNN(
        n_participants=n_participants,
        input_dim=len(ant_cat_col),
        hidden_dim=hidden_dim,
        output_dim=2,
        participant_emb_dim=participant_emb_dim,
        hyper_hidden_dim=hyper_hidden_dim,
    )

    save_path = (f'{save_dir}/ant_hyper_rnn_{participant_emb_dim}pdim_{hyper_hidden_dim}hyperh_{hidden_dim}wh_{n_epochs}ep.pt')

    model, history, best_val = fit_model(
        model=hyper_rnn,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        device=device,
        save_path=save_path,
        model_type="hyper",
        lr=lr,
        patience=patience,
    )

    # Test model
    test_metrics = test_model(model=model, test_loader=test_loader, device=device)

    # Collect predictions
    y_true, y_pred, p_correct = collect_predictions(
        model,
        test_loader,
        device,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p_correct = np.asarray(p_correct)

    test_acc = (y_true == y_pred).mean()
    test_bal_acc = balanced_accuracy_score(y_true, y_pred)
    test_auc = roc_auc_score(y_true, p_correct)

    print("Best validation loss:", best_val)
    print("Test accuracy:", test_acc)
    print("Balanced accuracy:", test_bal_acc)
    print("AUC:", test_auc)
    print(classification_report(y_true, y_pred, digits=4))

    # Save history for this model
    history_df = pd.DataFrame(history)
    history_path = (
        f"{save_dir}/history_ant_hyper_rnn_"
        f"{participant_emb_dim}pdim_"
        f"{hyper_hidden_dim}hyperh_"
        f"{hidden_dim}wh_"
        f"{n_epochs}ep.csv"
    )
    history_df.to_csv(history_path, index=False)

    result = {
        "participant_emb_dim": participant_emb_dim,
        "hyper_hidden_dim": hyper_hidden_dim,
        "hidden_dim": hidden_dim,
        "n_epochs": n_epochs,
        "lr": lr,
        "patience": patience,
        "best_val_loss": best_val,
        "test_accuracy": test_acc,
        "test_balanced_accuracy": test_bal_acc,
        "test_auc": test_auc,
        "model_path": save_path,
        "history_path": history_path,
    }

    results.append(result)

    # Track best model by validation loss
    # This is the statistically cleaner choice.
    if best_val < best_val_loss:
        best_val_loss = best_val
        best_config = result
        best_model = model
        best_history = history_df

# Save all results
results_df = pd.DataFrame(results)

results_path = f"{save_dir}/ant_hyperparameter_results.csv"
results_df.to_csv(results_path, index=False)

# Sort for inspection
results_sorted = results_df.sort_values(
    by=["best_val_loss"],
    ascending=True,
)

print("\n" + "=" * 80)
print("Best model by validation loss:")
print(best_config)

print("\nTop 10 models by validation loss:")
print(results_sorted.head(10))

# Also save the best config separately
best_config_df = pd.DataFrame([best_config])
best_config_path = f"{save_dir}/ant_best_hyperparameter_config.csv"
best_config_df.to_csv(best_config_path, index=False)

# plt.hist(p_correct[y_true == 1], bins=50, alpha=0.5, label="correct")
# plt.hist(p_correct[y_true == 0], bins=50, alpha=0.5, label="error")
# plt.legend()
# plt.show()

# ======================================================================================================================
# Model Training (Just reuse the code and change the model initialization to fit basic RNN and LSTM)
# ======================================================================================================================
num_layers = [1, 2, 3]
hidden_dims = [2, 4, 8, 16, 32, 64]

results = []

best_val_loss = np.inf
best_config = None
best_model = None
best_history = None

save_dir = './Results/NN_Results/rnn'
os.makedirs(save_dir, exist_ok=True)

for num_layer, hidden_dim in itertools.product(num_layers, hidden_dims):
    print("=" * 80)
    print(f"Training HyperRNN | num_layer={num_layer}, hidden_dim={hidden_dim}")

    rnn = BaselineRNN(
        input_dim=len(ant_cat_col),
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        output_dim=2,
    )

    save_path = (f'{save_dir}/ant_rnn_{num_layer}layer_{hidden_dim}wh_{n_epochs}ep.pt')

    model, history, best_val = fit_model(
        model=rnn,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        device=device,
        save_path=save_path,
        model_type="baseline",
        lr=lr,
        patience=patience,
    )

    # Test model
    test_metrics = test_model(model=model, test_loader=test_loader, device=device)

    # Collect predictions
    y_true, y_pred, p_correct = collect_predictions(
        model,
        test_loader,
        device,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p_correct = np.asarray(p_correct)

    test_acc = (y_true == y_pred).mean()
    test_bal_acc = balanced_accuracy_score(y_true, y_pred)
    test_auc = roc_auc_score(y_true, p_correct)

    print("Best validation loss:", best_val)
    print("Test accuracy:", test_acc)
    print("Balanced accuracy:", test_bal_acc)
    print("AUC:", test_auc)
    print(classification_report(y_true, y_pred, digits=4))

    # Save history for this model
    history_df = pd.DataFrame(history)
    history_path = (
        f"{save_dir}/history_ant_rnn_"
        f"{participant_emb_dim}pdim_"
        f"{hyper_hidden_dim}hyperh_"
        f"{hidden_dim}wh_"
        f"{n_epochs}ep.csv"
    )
    history_df.to_csv(history_path, index=False)

    result = {
        "num_layer": num_layer,
        "hidden_dim": hidden_dim,
        "n_epochs": n_epochs,
        "lr": lr,
        "patience": patience,
        "best_val_loss": best_val,
        "test_accuracy": test_acc,
        "test_balanced_accuracy": test_bal_acc,
        "test_auc": test_auc,
        "model_path": save_path,
        "history_path": history_path,
    }

    results.append(result)

    # Track best model by validation loss
    # This is the statistically cleaner choice.
    if best_val < best_val_loss:
        best_val_loss = best_val
        best_config = result
        best_model = model
        best_history = history_df

# Save all results
results_df = pd.DataFrame(results)

results_path = f"{save_dir}/ant_hyperparameter_results.csv"
results_df.to_csv(results_path, index=False)

# Sort for inspection
results_sorted = results_df.sort_values(
    by=["best_val_loss"],
    ascending=True,
)

print("\n" + "=" * 80)
print("Best model by validation loss:")
print(best_config)

print("\nTop 10 models by validation loss:")
print(results_sorted.head(10))

# Also save the best config separately
best_config_df = pd.DataFrame([best_config])
best_config_path = f"{save_dir}/ant_best_hyperparameter_config.csv"
best_config_df.to_csv(best_config_path, index=False)
