import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from HyperNetwork import *

device = torch.device("cpu")
batch_size = 16
n_fold = 5

# CCT
cct_data = pd.read_csv('./Data/hypernetwork_data/cct_data.csv')

# # Inspect the data
# for col in cct_data.columns:
#     print(col)
#     print(cct_data[col].value_counts())

# Reassign participant IDs to be consecutive integers starting from 0
cct_data["participant_id"] = cct_data["worker_id"].astype("category").cat.codes
n_participants = cct_data["participant_id"].nunique()
print(n_participants)

# Add trial index
cct_data["trial"] = cct_data.groupby("participant_id").cumcount() + 1
cct_data = cct_data.sort_values(["participant_id", "trial"]).reset_index(drop=True)

# Encode
cct_norm_col = ['gain_amount', 'loss_amount', 'num_cards', 'num_click_in_round', 'num_loss_cards']
x_col = cct_norm_col + ['gain_probability', 'loss_probability']
y_col = 'action'

# Scale numeric variables
scaler = MinMaxScaler()
cct_data[cct_norm_col] = scaler.fit_transform(cct_data[cct_norm_col])

# Make sure y is integer 0/1
cct_data[y_col] = cct_data[y_col].astype('category').cat.codes

# All choices are included
cct_data["choice_mask"] = 1

baseline_acc = cct_data[y_col].mean()
print(baseline_acc)
baseline_nll = -(cct_data[y_col] * np.log(baseline_acc) + (1 - cct_data[y_col]) * np.log(1 - baseline_acc)).mean()
print(baseline_nll)

# Interspersed cross-validation setup
cct_data = add_interspersed_folds(cct_data, n_folds=n_fold, seed=42)

# Define model-general hyperparameters
n_epochs = 400
lr = 5e-3
patience = 50

if __name__ == "__main__":
    # # ======================================================================================================================
    # # Model Training (Just reuse the code and change the model initialization to fit basic RNN and LSTM)
    # # ======================================================================================================================
    # participant_emb_dims = [2, 4, 8, 16, 32]
    # hyper_hidden_dims = [2, 4, 8, 16, 32, 64]
    # hidden_dims = [2, 4, 8, 16, 32, 64]
    #
    # all_results = []
    # results = []
    #
    # save_dir = './Results/NN_Results/hyper_rnn/cct'
    # os.makedirs(save_dir, exist_ok=True)
    #
    # for participant_emb_dim, hyper_hidden_dim, hidden_dim in itertools.product(participant_emb_dims, hyper_hidden_dims, hidden_dims):
    #     fold_results = []
    #
    #     for test_fold in range(n_fold):
    #         val_fold = (test_fold + 1) % n_fold
    #
    #         # Define train, val, test splits based on folds
    #         train_df = make_full_sequence_split_df(cct_data, test_fold=test_fold, val_fold=val_fold, split="train")
    #         val_df = make_full_sequence_split_df(cct_data, test_fold=test_fold, val_fold=val_fold, split="val")
    #         test_df = make_full_sequence_split_df(cct_data, test_fold=test_fold, val_fold=val_fold, split="test")
    #
    #         mask_total = (train_df["loss_mask"].sum() + val_df["loss_mask"].sum() + test_df["loss_mask"].sum())
    #         assert mask_total == cct_data["choice_mask"].sum()
    #
    #         # Transform into sequences
    #         train_seq = BehavioralDataset(train_df, x_var=x_col, y_var=y_col)
    #         val_seq = BehavioralDataset(val_df, x_var=x_col, y_var=y_col)
    #         test_seq = BehavioralDataset(test_df, x_var=x_col, y_var=y_col)
    #
    #         # Dataloaders
    #         train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True, collate_fn=behavioral_collate_fn)
    #         val_loader = DataLoader(val_seq, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)
    #         test_loader = DataLoader(test_seq, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)
    #
    #         print("=" * 80)
    #         print(f"Training HyperRNN | fold={test_fold}, participant_emb_dim={participant_emb_dim}, "
    #               f"hyper_hidden_dim={hyper_hidden_dim}, hidden_dim={hidden_dim}")
    #
    #         model = HyperRNN(
    #             n_participants=n_participants,
    #             input_dim=len(x_col),
    #             hidden_dim=hidden_dim,
    #             output_dim=2,
    #             participant_emb_dim=participant_emb_dim,
    #             hyper_hidden_dim=hyper_hidden_dim,
    #         )
    #
    #         save_path = (f"{save_dir}/cct_hyper_rnn_fold{test_fold}_{participant_emb_dim}pdim_{hyper_hidden_dim}hyperh_{hidden_dim}hidden.pt")
    #
    #         model, history, best_val = fit_model(
    #             model=model,
    #             train_loader=train_loader,
    #             val_loader=val_loader,
    #             n_epochs=n_epochs,
    #             device=device,
    #             save_path=save_path,
    #             model_type="hyper",
    #             lr=lr,
    #             patience=patience,
    #         )
    #
    #         y_true, y_pred, p_correct = collect_predictions(model, test_loader, device)
    #
    #         y_true = np.asarray(y_true)
    #         y_pred = np.asarray(y_pred)
    #         p_correct = np.asarray(p_correct)
    #
    #         test_acc = (y_true == y_pred).mean()
    #         test_bal_acc = balanced_accuracy_score(y_true, y_pred)
    #         test_auc = roc_auc_score(y_true, p_correct)
    #
    #         print("Best val loss:", best_val)
    #         print("Test accuracy:", test_acc)
    #         print("Balanced accuracy:", test_bal_acc)
    #         print("AUC:", test_auc)
    #         print("Predicted counts:", np.unique(y_pred, return_counts=True))
    #         print(classification_report(y_true, y_pred, zero_division=0))
    #
    #         result = {
    #             "model_name": "HyperRNN",
    #             "test_fold": test_fold,
    #             "val_fold": val_fold,
    #             "participant_emb_dim": participant_emb_dim,
    #             "hyper_hidden_dim": hyper_hidden_dim,
    #             "hidden_dim": hidden_dim,
    #             "n_epochs": n_epochs,
    #             "lr": lr,
    #             "patience": patience,
    #             "best_val_loss": best_val,
    #             "test_accuracy": test_acc,
    #             "test_balanced_accuracy": test_bal_acc,
    #             "test_auc": test_auc,
    #             "n_pred_0": int(np.sum(y_pred == 0)),
    #             "n_pred_1": int(np.sum(y_pred == 1)),
    #             "model_path": save_path,
    #         }
    #
    #         fold_results.append(result)
    #         all_results.append(result)
    #
    #     fold_results_df = pd.DataFrame(fold_results)
    #
    #     print("\nMean CV result for this config:")
    #     print(fold_results_df[[
    #         "best_val_loss",
    #         "test_accuracy",
    #         "test_balanced_accuracy",
    #         "test_auc",
    #     ]].mean())
    #
    # # Save the results
    # all_results_df = pd.DataFrame(all_results)
    # all_results_df.to_csv(f"{save_dir}/cct_hyper_rnn_interspersed_cv_all_results.csv", index=False)
    #
    # summary_df = (all_results_df.groupby(["participant_emb_dim", "hyper_hidden_dim", "hidden_dim", "lr", "patience"]).agg(
    #     mean_best_val_loss=("best_val_loss", "mean"),
    #     mean_test_accuracy=("test_accuracy", "mean"),
    #     mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
    #     mean_test_auc=("test_auc", "mean"),
    #     sd_test_accuracy=("test_accuracy", "std"),
    #     sd_test_balanced_accuracy=("test_balanced_accuracy", "std"),
    #     sd_test_auc=("test_auc", "std"),
    #     ).reset_index())
    #
    # summary_df.to_csv(f"{save_dir}/cct_hyper_rnn_interspersed_cv_summary.csv", index=False)
    # best_overall = summary_df.sort_values(by="mean_best_val_loss", ascending=True).iloc[0]
    #
    # print("=" * 80)
    # print("Best overall model by mean validation loss:")
    # print(best_overall)
    #
    # pd.DataFrame([best_overall]).to_csv(f"{save_dir}/cct_hyper_rnn_best_interspersed_cv_model.csv", index=False)

    # # ======================================================================================================================
    # # Model Training (Just reuse the code and change the model initialization to fit basic RNN and LSTM)
    # # ======================================================================================================================
    # num_layers = [1, 2, 3]
    # hidden_dims = [2, 4, 8, 16, 32, 64]
    #
    # all_results = []
    # results = []
    #
    # save_dir = './Results/NN_Results/rnn/cct'
    # os.makedirs(save_dir, exist_ok=True)
    #
    # for num_layer, hidden_dim in itertools.product(num_layers, hidden_dims):
    #     fold_results = []
    #
    #     for test_fold in range(n_fold):
    #         val_fold = (test_fold + 1) % n_fold
    #
    #         # Define train, val, test splits based on folds
    #         train_df = make_full_sequence_split_df(cct_data, test_fold=test_fold, val_fold=val_fold, split="train")
    #         val_df = make_full_sequence_split_df(cct_data, test_fold=test_fold, val_fold=val_fold, split="val")
    #         test_df = make_full_sequence_split_df(cct_data, test_fold=test_fold, val_fold=val_fold, split="test")
    #
    #         mask_total = (train_df["loss_mask"].sum() + val_df["loss_mask"].sum() + test_df["loss_mask"].sum())
    #         assert mask_total == cct_data["choice_mask"].sum()
    #
    #         # Transform into sequences
    #         train_seq = BehavioralDataset(train_df, x_var=x_col, y_var=y_col)
    #         val_seq = BehavioralDataset(val_df, x_var=x_col, y_var=y_col)
    #         test_seq = BehavioralDataset(test_df, x_var=x_col, y_var=y_col)
    #
    #         # Dataloaders
    #         train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True, collate_fn=behavioral_collate_fn)
    #         val_loader = DataLoader(val_seq, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)
    #         test_loader = DataLoader(test_seq, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)
    #
    #         print("=" * 80)
    #         print(f"Training BaselineRNN | fold={test_fold}, num_layer={num_layer}, hidden_dim={hidden_dim}")
    #
    #         model = BaselineRNN(
    #             input_dim=len(x_col),
    #             hidden_dim=hidden_dim,
    #             num_layer=num_layer,
    #             output_dim=2,
    #         )
    #
    #         save_path = (f"{save_dir}/cct_rnn_fold{test_fold}_{num_layer}layer_{hidden_dim}hidden.pt")
    #
    #         model, history, best_val = fit_model(
    #             model=model,
    #             train_loader=train_loader,
    #             val_loader=val_loader,
    #             n_epochs=n_epochs,
    #             device=device,
    #             save_path=save_path,
    #             model_type="baseline",
    #             lr=lr,
    #             patience=patience,
    #         )
    #
    #         y_true, y_pred, p_correct = collect_predictions(model, test_loader, device)
    #
    #         y_true = np.asarray(y_true)
    #         y_pred = np.asarray(y_pred)
    #         p_correct = np.asarray(p_correct)
    #
    #         test_acc = (y_true == y_pred).mean()
    #         test_bal_acc = balanced_accuracy_score(y_true, y_pred)
    #         test_auc = roc_auc_score(y_true, p_correct)
    #
    #         print("Best val loss:", best_val)
    #         print("Test accuracy:", test_acc)
    #         print("Balanced accuracy:", test_bal_acc)
    #         print("AUC:", test_auc)
    #         print("Predicted counts:", np.unique(y_pred, return_counts=True))
    #         print(classification_report(y_true, y_pred, zero_division=0))
    #
    #         result = {
    #             "model_name": "BaselineRNN",
    #             "test_fold": test_fold,
    #             "val_fold": val_fold,
    #             "num_layer": num_layer,
    #             "hidden_dim": hidden_dim,
    #             "n_epochs": n_epochs,
    #             "lr": lr,
    #             "patience": patience,
    #             "best_val_loss": best_val,
    #             "test_accuracy": test_acc,
    #             "test_balanced_accuracy": test_bal_acc,
    #             "test_auc": test_auc,
    #             "n_pred_0": int(np.sum(y_pred == 0)),
    #             "n_pred_1": int(np.sum(y_pred == 1)),
    #             "model_path": save_path,
    #         }
    #
    #         fold_results.append(result)
    #         all_results.append(result)
    #
    #     fold_results_df = pd.DataFrame(fold_results)
    #
    #     print("\nMean CV result for this config:")
    #     print(fold_results_df[[
    #         "best_val_loss",
    #         "test_accuracy",
    #         "test_balanced_accuracy",
    #         "test_auc",
    #     ]].mean())
    #
    # # Save the results
    # all_results_df = pd.DataFrame(all_results)
    # all_results_df.to_csv(f"{save_dir}/cct_rnn_interspersed_cv_all_results.csv", index=False)
    #
    # summary_df = (all_results_df.groupby(["num_layer", "hidden_dim", "lr", "patience"]).agg(
    #     mean_best_val_loss=("best_val_loss", "mean"),
    #     mean_test_accuracy=("test_accuracy", "mean"),
    #     mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
    #     mean_test_auc=("test_auc", "mean"),
    #     sd_test_accuracy=("test_accuracy", "std"),
    #     sd_test_balanced_accuracy=("test_balanced_accuracy", "std"),
    #     sd_test_auc=("test_auc", "std"),
    #     ).reset_index())
    #
    # summary_df.to_csv(f"{save_dir}/cct_rnn_interspersed_cv_summary.csv", index=False)
    # best_overall = summary_df.sort_values(by="mean_best_val_loss", ascending=True).iloc[0]
    #
    # print("=" * 80)
    # print("Best overall model by mean validation loss:")
    # print(best_overall)
    #
    # pd.DataFrame([best_overall]).to_csv(f"{save_dir}/cct_rnn_best_interspersed_cv_model.csv", index=False)
    #
    # ==================================================================================================================
    # Train the final model after comparing across all the CV results
    # ==================================================================================================================
    final_participant_emb_dim = 2
    final_hyper_hidden_dim = 16
    final_hidden_dim = 8

    save_dir = './Results/Final_Models/'
    os.makedirs(save_dir, exist_ok=True)

    # Randomly select one fold as validation data
    val_fold = np.random.randint(0, 5)  # 2

    # Define train, val, test splits based on folds
    train_df = cct_data.copy()
    train_df["loss_mask"] = train_df["choice_mask"]
    val_df = make_full_sequence_split_df(cct_data, test_fold=None, val_fold=val_fold, split="val")

    # Transform into sequences
    train_seq = BehavioralDataset(train_df, x_var=x_col, y_var=y_col)
    val_seq = BehavioralDataset(val_df, x_var=x_col, y_var=y_col)

    # Dataloaders
    train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True, collate_fn=behavioral_collate_fn)
    val_loader = DataLoader(val_seq, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)
    print("=" * 80)
    print(f"Training final HyperRNN | participant_emb_dim={final_participant_emb_dim}, "
          f"hyper_hidden_dim={final_hyper_hidden_dim}, hidden_dim={final_hidden_dim}")

    model = HyperRNN(n_participants=n_participants, input_dim=len(x_col), hidden_dim=final_hidden_dim,
                     output_dim=2, participant_emb_dim=final_participant_emb_dim, hyper_hidden_dim=final_hyper_hidden_dim)

    save_path = f"./Results/Final_Models/cct_hyper_rnn_final_model.pt"

    model, history, best_val = fit_model(model=model, train_loader=train_loader, val_loader=val_loader,
                                         n_epochs=200, device=device, save_path=save_path, model_type="hyper",
                                         lr=lr, patience=200)

    y_true, y_pred, p_correct = collect_predictions(model, val_loader, device)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p_correct = np.asarray(p_correct)
    test_acc = (y_true == y_pred).mean()
    test_bal_acc = balanced_accuracy_score(y_true, y_pred)
    test_auc = roc_auc_score(y_true, p_correct)
    print("Best val loss:", best_val)
    print("Test accuracy:", test_acc)
    print("Balanced accuracy:", test_bal_acc)
    print("AUC:", test_auc)
    print("Predicted counts:", np.unique(y_pred, return_counts=True))
    print(classification_report(y_true, y_pred, zero_division=0))

    # Save the participant embeddings
    participant_embeddings = model.participant_embedding.weight.data.cpu().numpy()
    np.save(f"./Results/Final_Models/cct_hyper_rnn_final_participant_embeddings.npy", participant_embeddings)

    # fold_embeddings = []
    # fold_metrics = []
    #
    # for val_fold in range(5):
    #     print("=" * 80)
    #     print(f"Training HyperRNN fold {val_fold} | participant_emb_dim={final_participant_emb_dim}, "
    #           f"hyper_hidden_dim={final_hyper_hidden_dim}, hidden_dim={final_hidden_dim}")
    #
    #     # Define train/val splits
    #     train_df = make_full_sequence_split_df(cct_data, test_fold=None, val_fold=val_fold, split="train")
    #     val_df = make_full_sequence_split_df(cct_data, test_fold=None, val_fold=val_fold, split="val")
    #
    #     # Sanity check: train + val loss masks should cover all valid choice trials
    #     mask_total = train_df["loss_mask"].sum() + val_df["loss_mask"].sum()
    #     assert mask_total == cct_data["choice_mask"].sum(), (
    #         f"Fold {val_fold}: mask mismatch. "
    #         f"train + val = {mask_total}, total = {cct_data['choice_mask'].sum()}"
    #     )
    #
    #     # Transform into sequence datasets
    #     train_seq = BehavioralDataset(train_df, x_var=x_col, y_var=y_col)
    #     val_seq = BehavioralDataset(val_df, x_var=x_col, y_var=y_col)
    #
    #     # Dataloaders
    #     train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True, collate_fn=behavioral_collate_fn)
    #     val_loader = DataLoader(val_seq, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)
    #
    #     # Initialize a fresh model for this fold
    #     model = HyperRNN(
    #         n_participants=n_participants,
    #         input_dim=len(x_col),
    #         hidden_dim=final_hidden_dim,
    #         output_dim=2,
    #         participant_emb_dim=final_participant_emb_dim,
    #         hyper_hidden_dim=final_hyper_hidden_dim,
    #     )
    #
    #     # Save one checkpoint per fold
    #     save_path = os.path.join(save_dir, f"cct_hyper_rnn_final_model_fold{val_fold}.pt")
    #
    #     model, history, best_val = fit_model(
    #         model=model,
    #         train_loader=train_loader,
    #         val_loader=val_loader,
    #         n_epochs=n_epochs,
    #         device=device,
    #         save_path=save_path,
    #         model_type="hyper",
    #         lr=lr,
    #         patience=patience,
    #         extra_config={
    #             "task": "ant",
    #             "val_fold": val_fold,
    #             "n_participants": n_participants,
    #             "input_dim": len(x_col),
    #             "hidden_dim": final_hidden_dim,
    #             "output_dim": 2,
    #             "participant_emb_dim": final_participant_emb_dim,
    #             "hyper_hidden_dim": final_hyper_hidden_dim,
    #         }
    #     )
    #
    #     # Evaluate on validation fold
    #     y_true, y_pred, p_correct = collect_predictions(model, val_loader, device)
    #
    #     y_true = np.asarray(y_true)
    #     y_pred = np.asarray(y_pred)
    #     p_correct = np.asarray(p_correct)
    #
    #     val_acc = (y_true == y_pred).mean()
    #     val_bal_acc = balanced_accuracy_score(y_true, y_pred)
    #     val_auc = roc_auc_score(y_true, p_correct)
    #
    #     print("Best val loss:", best_val)
    #     print("Validation accuracy:", val_acc)
    #     print("Validation balanced accuracy:", val_bal_acc)
    #     print("Validation AUC:", val_auc)
    #     print("Predicted counts:", np.unique(y_pred, return_counts=True))
    #     print(classification_report(y_true, y_pred, zero_division=0))
    #
    #     fold_metrics.append({
    #         "fold": val_fold,
    #         "best_val_loss": best_val,
    #         "val_acc": val_acc,
    #         "val_bal_acc": val_bal_acc,
    #         "val_auc": val_auc,
    #     })
    #
    #     # Extract participant embeddings for this fold
    #     participant_embeddings = (model.participant_embedding.weight.detach().cpu().numpy())
    #     fold_embeddings.append(participant_embeddings)
    #
    #     # Save fold-specific embeddings
    #     np.save(os.path.join(save_dir, f"cct_hyper_rnn_participant_embeddings_fold{val_fold}.npy"), participant_embeddings)
    #
    # # Calculate average participant embeddings across folds
    # fold_embeddings = np.stack(fold_embeddings, axis=0)
    #
    # # Use fold 0 as the anchor/reference
    # reference_embedding = fold_embeddings[0].copy()
    # aligned_fold_embeddings = []
    #
    # for fold_idx in range(fold_embeddings.shape[0]):
    #     current_embedding = fold_embeddings[fold_idx].copy()
    #
    #     # For fold 0, keep unchanged
    #     if fold_idx == 0:
    #         aligned_fold_embeddings.append(current_embedding)
    #         continue
    #
    #     # Align each embedding dimension to the same dimension in fold 0
    #     for dim_idx in range(current_embedding.shape[1]):
    #         ref_dim = reference_embedding[:, dim_idx]
    #         cur_dim = current_embedding[:, dim_idx]
    #
    #         corr = np.corrcoef(ref_dim, cur_dim)[0, 1]
    #
    #         # If this dimension is flipped relative to fold 0, flip its sign
    #         if corr < 0:
    #             current_embedding[:, dim_idx] *= -1
    #
    #         print(
    #             f"Fold {fold_idx}, dim {dim_idx}: "
    #             f"corr with fold 0 = {corr:.4f}, "
    #             f"{'flipped' if corr < 0 else 'kept'}"
    #         )
    #
    #     aligned_fold_embeddings.append(current_embedding)
    #
    # aligned_fold_embeddings = np.stack(aligned_fold_embeddings, axis=0)
    #
    # # Average after sign alignment
    # avg_participant_embeddings = aligned_fold_embeddings.mean(axis=0)
    #
    # # Save aligned fold embeddings
    # np.save(os.path.join(save_dir, "cct_hyper_rnn_aligned_fold_embeddings.npy"), aligned_fold_embeddings)
    #
    # # Save aligned average embeddings
    # np.save(os.path.join(save_dir, "cct_hyper_rnn_avg_participant_embeddings_sign_aligned.npy"), avg_participant_embeddings)