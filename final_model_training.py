import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ant_training import ant_data
from cct_training import cct_data
from dd_training import dd_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import eigh
from sklearn.cross_decomposition import CCA
from functools import reduce
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import pairwise_distances

# Read the hypernetwork model results
ant_hyper_results = pd.read_csv('./Results/NN_Results/hyper_rnn/ant/ant_hyper_rnn_interspersed_cv_summary.csv')
cct_hyper_results = pd.read_csv('./Results/NN_Results/hyper_rnn/cct/cct_hyper_rnn_interspersed_cv_summary.csv')
dd_hyper_results = pd.read_csv('./Results/NN_Results/hyper_rnn/dd/dd_hyper_rnn_interspersed_cv_summary.csv')
stroop_hyper_results = pd.read_csv('./Results/NN_Results/hyper_rnn/stroop/stroop_hyper_rnn_interspersed_cv_summary.csv')

# Read the regular RNN results
ant_rnn_results = pd.read_csv('./Results/NN_Results/rnn/ant/ant_rnn_interspersed_cv_summary.csv')
cct_rnn_results = pd.read_csv('./Results/NN_Results/rnn/cct/cct_rnn_interspersed_cv_summary.csv')
dd_rnn_results = pd.read_csv('./Results/NN_Results/rnn/dd/dd_rnn_interspersed_cv_summary.csv')
stroop_rnn_results = pd.read_csv('./Results/NN_Results/rnn/stroop/stroop_rnn_interspersed_cv_summary.csv')

# Sort by evaluation metrics
metric = ['mean_test_auc'] # mean_best_val_loss
task_names = ['ANT', 'CCT', 'DD', 'Stroop']
i = 0
for df in [ant_hyper_results, cct_hyper_results, dd_hyper_results, stroop_hyper_results]:
    print(f'{task_names[i]}:')
    print(df.groupby('participant_emb_dim')[metric].mean())
    i += 1

# Hyperparameter columns shared by both summary files
config_cols = [
    "participant_emb_dim",
    "hyper_hidden_dim",
    "hidden_dim"
]

def build_joint_config_results(results_list, config_cols, task_names=None, rank_metrics=None, raw_mean_metrics=None):
    if task_names is None:
        task_names = [f"task{i + 1}" for i in range(len(results_list))]

    if len(results_list) != len(task_names):
        raise ValueError("results_list and task_names must have the same length.")

    if rank_metrics is None:
        rank_metrics = {
            "mean_best_val_loss": True,
            "mean_test_auc": False,
            "mean_test_balanced_accuracy": False,
        }

    if raw_mean_metrics is None:
        raw_mean_metrics = [
            "mean_best_val_loss",
            "mean_test_accuracy",
            "mean_test_balanced_accuracy",
            "mean_test_auc",
        ]

    processed = []

    for df, task_name in zip(results_list, task_names):
        df = df.copy()

        # Check required columns
        required_cols = set(config_cols) | set(rank_metrics.keys()) | set(raw_mean_metrics)
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            raise ValueError(
                f"Task '{task_name}' is missing columns: {sorted(missing_cols)}"
            )

        # Rank columns are created WITHOUT task suffix first
        for metric, ascending in rank_metrics.items():
            df[f"rank_{metric}"] = df[metric].rank(
                ascending=ascending,
                method="min"
            )

        # Now add task suffix to all non-config columns
        rename_map = {col: f"{col}_{task_name}" for col in df.columns if col not in config_cols}

        df = df.rename(columns=rename_map)
        processed.append(df)

    # Merge all dfs by shared hyperparameter configuration
    joint = reduce(lambda left, right: left.merge(right, on=config_cols), processed)

    # Average ranks across tasks
    for metric in rank_metrics:
        rank_cols = [f"rank_{metric}_{task_name}" for task_name in task_names]
        joint[f"avg_rank_{metric}"] = joint[rank_cols].mean(axis=1)

    # Average raw metrics across tasks
    for metric in raw_mean_metrics:
        metric_cols = [f"{metric}_{task_name}" for task_name in task_names]
        joint[f"mean_{metric}_across_tasks"] = joint[metric_cols].mean(axis=1)

    return joint

joint_results = build_joint_config_results([ant_hyper_results, dd_hyper_results, cct_hyper_results,
                                            stroop_hyper_results], config_cols, task_names)

# Load participant embeddings
ant_embeddings = pd.DataFrame(np.load('./Results/Final_Models/ant_hyper_rnn_final_participant_embeddings.npy'))
dd_embeddings = pd.DataFrame(np.load('./Results/Final_Models/dd_hyper_rnn_final_participant_embeddings.npy'))
cct_embeddings = pd.DataFrame(np.load('./Results/Final_Models/cct_hyper_rnn_final_participant_embeddings.npy'))
stroop_embeddings = pd.DataFrame(np.load('./Results/Final_Models/stroop_hyper_rnn_final_participant_embeddings.npy'))
motor_embeddings = pd.DataFrame(np.load('./Results/Final_Models/motor_hyper_rnn_final_participant_embeddings.npy'))

# Add participant IDs to the embeddings
ant_embeddings['participant_id'] = ant_embeddings.index
dd_embeddings['participant_id'] = dd_embeddings.index
cct_embeddings['participant_id'] = cct_embeddings.index

# Revert participant ID to worker id
ant_embeddings = pd.merge(ant_embeddings, ant_data[['participant_id', 'worker_id']].drop_duplicates(), on='participant_id', how='left').drop(columns=['participant_id'])
dd_embeddings = pd.merge(dd_embeddings, dd_data[['participant_id', 'worker_id']].drop_duplicates(), on='participant_id', how='left').drop(columns=['participant_id'])
cct_embeddings = pd.merge(cct_embeddings, cct_data[['participant_id', 'worker_id']].drop_duplicates(), on='participant_id', how='left').drop(columns=['participant_id'])

def rename_embedding_cols(df, task_name):
    emb_cols = [c for c in df.columns if c != "worker_id"]
    rename_dict = {c: f"{task_name}_{c}" for c in emb_cols}
    return df[["worker_id"] + emb_cols].rename(columns=rename_dict)

ant_embeddings = rename_embedding_cols(ant_embeddings, "ant")
dd_embeddings = rename_embedding_cols(dd_embeddings, "dd")
cct_embeddings = rename_embedding_cols(cct_embeddings, "cct")

all_embeddings = ant_embeddings.merge(dd_embeddings, on="worker_id", how="inner").merge(cct_embeddings, on="worker_id", how="inner")
print(f"Number of participants with embeddings in all three tasks: {all_embeddings.shape[0]}")

# Extract each task embeddings matrix
ant_cols = [c for c in all_embeddings.columns if c.startswith("ant_")]
dd_cols = [c for c in all_embeddings.columns if c.startswith("dd_")]
cct_cols = [c for c in all_embeddings.columns if c.startswith("cct_")]

E_ant = all_embeddings[ant_cols].to_numpy()
E_dd = all_embeddings[dd_cols].to_numpy()
E_cct = all_embeddings[cct_cols].to_numpy()



def standardize(E):
    return StandardScaler().fit_transform(E)

def align_to_reference(E_ref, E_target):
    """
    Align E_target to E_ref using orthogonal Procrustes.

    Both matrices should be:
        n_participants × embedding_dim

    Participants must be in the same order.
    """
    E_ref_z = standardize(E_ref)
    E_target_z = standardize(E_target)

    R, scale = orthogonal_procrustes(E_target_z, E_ref_z)

    E_target_aligned = E_target_z @ R

    return E_target_aligned, R, scale


E_ant_ref = standardize(E_ant)

E_dd_aligned, R_dd, scale_dd = align_to_reference(E_dd, E_dd)
E_cct_aligned, R_cct, scale_cct = align_to_reference(E_ant, E_cct)

for dim in range(E_ant_ref.shape[1]):
    corr = np.corrcoef(E_dd_aligned[:, dim], E_cct_aligned[:, dim])[0, 1]
    print(f"Correlation for dimension {dim}: {corr:.4f}")


# E_ant_ref: n_participants × emb_dim
# E_dd_aligned: n_participants × emb_dim
# rows must be same participants in same order

cross_dist = cdist(E_ant_ref, E_dd_aligned, metric="euclidean")

# diagonal = same participant across ANT and DD
same_participant_dist = np.diag(cross_dist)

# off-diagonal = different participants
off_diag_mask = ~np.eye(cross_dist.shape[0], dtype=bool)
different_participant_dist = cross_dist[off_diag_mask]

print("Mean same-participant distance:", same_participant_dist.mean())
print("Mean different-participant distance:", different_participant_dist.mean())

# Standardize the embeddings
scaler = StandardScaler()
E_ant_z = scaler.fit_transform(E_ant)
E_dd_z = scaler.fit_transform(E_dd)
E_cct_z = scaler.fit_transform(E_cct)

metric = 'cosine'
D_ant = pairwise_distances(E_ant_ref, metric=metric)
D_dd = pairwise_distances(E_dd_aligned, metric=metric)
D_cct = pairwise_distances(E_cct_aligned, metric=metric)

def upper_tri_vec(M):
    idx = np.triu_indices_from(M, k=1)
    return M[idx]

r_ant_dd, p_ant_dd = spearmanr(upper_tri_vec(D_ant), upper_tri_vec(D_dd))
r_ant_cct, p_ant_cct = spearmanr(upper_tri_vec(D_ant), upper_tri_vec(D_cct))
r_dd_cct, p_dd_cct = spearmanr(upper_tri_vec(D_dd), upper_tri_vec(D_cct))

print("ANT-DD:", r_ant_dd, p_ant_dd)
print("ANT-CCT:", r_ant_cct, p_ant_cct)
print("DD-CCT:", r_dd_cct, p_dd_cct)

def cross_embedding_corr_matrix(E1, E2, method="spearman", standardize=True, task_names=None):
    if standardize:
        E1 = StandardScaler().fit_transform(E1)
        E2 = StandardScaler().fit_transform(E2)

    if task_names is None:
        task_names = ['task1', 'task2']

    d1 = E1.shape[1]
    d2 = E2.shape[1]

    corr_mat = np.zeros((d1, d2))
    pval_mat = np.zeros((d1, d2))

    for i in range(d1):
        for j in range(d2):
            if method == "pearson":
                r, p = pearsonr(E1[:, i], E2[:, j])
            elif method == "spearman":
                r, p = spearmanr(E1[:, i], E2[:, j])
            else:
                raise ValueError("method must be 'pearson' or 'spearman'")

            corr_mat[i, j] = r
            pval_mat[i, j] = p

    corr_df = pd.DataFrame(
        corr_mat,
        index=[f"{task_names[0]}_{i}" for i in range(d1)],
        columns=[f"{task_names[1]}_{j}" for j in range(d2)]
    )

    pval_df = pd.DataFrame(pval_mat, index=[f"{task_names[0]}_{i}" for i in range(d1)],
                           columns=[f"{task_names[1]}_{j}" for j in range(d2)])

    # Apply Bonferroni correction for multiple comparisons
    pvals_flat = pval_df.values.flatten()
    pvals_corrected = multipletests(pvals_flat, method='fdr_bh')[1]
    pval_df_corrected = pd.DataFrame(pvals_corrected.reshape(pval_df.shape))

    return corr_df, pval_df, pval_df_corrected

corr_ant_dd, p_ant_dd, p_ant_dd_corrected = cross_embedding_corr_matrix(
    E_ant_ref,
    E_cct_aligned,
    method="pearson",
    standardize=False,
    task_names=["ANT", "CCT"]
)




D1 = pairwise_distances(E_ant_ref, metric="cosine")
D2 = pairwise_distances(E_cct_aligned, metric="cosine")

idx = np.triu_indices_from(D1, k=1)
r, p = spearmanr(D1[idx], D2[idx])

print(r, p)

D1 = squareform(pdist(E_ant_z, metric="cosine"))
D2 = pairwise_distances(E_ant_z, metric="cosine")

np.allclose(D1, D2)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_ant_dd,
    annot=True,          # show correlation values
    cmap="coolwarm",     # color map
    center=0,            # make 0 the midpoint
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig('./Figures/Correlation_Matrix.png', dpi=300)


def get_cca_correlations(X, Y, n_components=1):
    """
    X: participants × features
    Y: participants × features

    Returns canonical correlations.
    """
    Xz = StandardScaler().fit_transform(X)
    Yz = StandardScaler().fit_transform(Y)

    cca = CCA(n_components=n_components, max_iter=10000)
    U, V = cca.fit_transform(Xz, Yz)

    rs = []
    for k in range(n_components):
        r, _ = pearsonr(U[:, k], V[:, k])
        rs.append(abs(r))  # sign is arbitrary in CCA

    return np.array(rs)


def permutation_test_cca(X, Y, n_components=1, n_perm=5000, seed=123):
    """
    Permutation test for CCA.

    Tests whether the observed first canonical correlation is larger than chance.
    """
    rng = np.random.default_rng(seed)

    observed_rs = get_cca_correlations(X, Y, n_components=n_components)
    observed_stat = observed_rs[0]

    perm_stats = []

    for _ in range(n_perm):
        perm_idx = rng.permutation(Y.shape[0])
        Y_perm = Y[perm_idx, :]

        perm_rs = get_cca_correlations(X, Y_perm, n_components=n_components)
        perm_stats.append(perm_rs[0])

    perm_stats = np.array(perm_stats)

    p_value = (np.sum(perm_stats >= observed_stat) + 1) / (n_perm + 1)

    return {
        "observed_r": observed_stat,
        "p_value": p_value,
        "perm_distribution": perm_stats,
    }
x = E_ant_ref[:, 6].reshape(-1, 1)
result = permutation_test_cca(E_ant_ref, E_dd_aligned, n_components=1, n_perm=1000)

print("Observed canonical r:", result["observed_r"])
print("Permutation p-value:", result["p_value"])


def generalized_cca_scores(embeddings, n_components=3, ridge=1e-3):
    """
    embeddings: list of arrays
        Example: [E_ant_z, E_dd_z, E_task3_z]
        each array is participants × embedding_dim

    Returns
    -------
    G : participants × n_components
        Shared participant factors.
    eigvals : array
        Strength of each shared component.
    """

    Xs = [StandardScaler().fit_transform(E) for E in embeddings]

    n = Xs[0].shape[0]
    M = np.zeros((n, n))

    for X in Xs:
        d = X.shape[1]

        XtX = X.T @ X
        P = X @ np.linalg.solve(XtX + ridge * np.eye(d), X.T)

        M += P

    eigvals, eigvecs = eigh(M)
    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx[:n_components]]
    G = eigvecs[:, idx[:n_components]]

    G = StandardScaler().fit_transform(G)

    return G, eigvals


# G, eigvals = generalized_cca_scores(
#     [E_ant_z, E_dd_z],
#     n_components=2
# )
#
# print(G.shape)
# print(eigvals)


def standardize(E):
    return StandardScaler().fit_transform(E)

def procrustes_align(E_ref, E_target):
    E_ref_z = standardize(E_ref)
    E_target_z = standardize(E_target)

    R, scale = orthogonal_procrustes(E_target_z, E_ref_z)
    E_target_aligned = E_target_z @ R

    return E_ref_z, E_target_aligned

def same_vs_different_distance_test(E_ref, E_target, n_perm=5000, seed=123):
    rng = np.random.default_rng(seed)

    # Observed alignment
    E_ref_z, E_target_aligned = procrustes_align(E_ref, E_target)

    cross_dist = cdist(E_ref_z, E_target_aligned, metric="euclidean")

    same_dist = np.diag(cross_dist)
    off_diag_mask = ~np.eye(cross_dist.shape[0], dtype=bool)
    diff_dist = cross_dist[off_diag_mask]

    observed_gap = diff_dist.mean() - same_dist.mean()

    perm_gaps = []

    for _ in range(n_perm):
        perm_idx = rng.permutation(E_target.shape[0])
        E_target_perm = E_target[perm_idx]

        E_ref_z_perm, E_target_perm_aligned = procrustes_align(E_ref, E_target_perm)

        cross_dist_perm = cdist(
            E_ref_z_perm,
            E_target_perm_aligned,
            metric="euclidean"
        )

        same_perm = np.diag(cross_dist_perm)
        diff_perm = cross_dist_perm[~np.eye(cross_dist_perm.shape[0], dtype=bool)]

        perm_gap = diff_perm.mean() - same_perm.mean()
        perm_gaps.append(perm_gap)

    perm_gaps = np.array(perm_gaps)

    # Larger gap means same-participant pairs are more uniquely close
    p_value = (np.sum(perm_gaps >= observed_gap) + 1) / (n_perm + 1)

    return {
        "mean_same_distance": same_dist.mean(),
        "mean_different_distance": diff_dist.mean(),
        "observed_gap": observed_gap,
        "p_value": p_value,
        "perm_gaps": perm_gaps,
    }


res = same_vs_different_distance_test(E_ant_ref, E_dd_aligned, n_perm=5000)

print("Mean same-participant distance:", res["mean_same_distance"])
print("Mean different-participant distance:", res["mean_different_distance"])
print("Observed gap:", res["observed_gap"])
print("Permutation p-value:", res["p_value"])


E_ant_ref, E_dd_aligned = procrustes_align(E_ant, E_dd)

cross_dist = cdist(E_ant_ref, E_dd_aligned, metric="euclidean")

nearest_dd = np.argmin(cross_dist, axis=1)
retrieval_acc = np.mean(nearest_dd == np.arange(cross_dist.shape[0]))

print("Retrieval accuracy:", retrieval_acc)
print("Chance level:", 1 / cross_dist.shape[0])