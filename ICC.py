import numpy as np
import pandas as pd
from functools import reduce
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from ant_training import ant_data
from cct_training import cct_data
from dd_training import dd_data
from stroop_training import stroop_data
from motor_training import motor_data
from scipy.linalg import orthogonal_procrustes

# ------------------------------------------------------------
# Load embeddings
# ------------------------------------------------------------
embedding_paths = {
    "ant": "./Results/Final_Models/ant_hyper_rnn_final_participant_embeddings.npy",
    "dd": "./Results/Final_Models/dd_hyper_rnn_final_participant_embeddings.npy",
    "cct": "./Results/Final_Models/cct_hyper_rnn_final_participant_embeddings.npy",
    "stroop": "./Results/Final_Models/stroop_hyper_rnn_final_participant_embeddings.npy",
    "motor": "./Results/Final_Models/motor_hyper_rnn_final_participant_embeddings.npy",
}

task_data_dict = {
    "ant": ant_data,
    "dd": dd_data,
    "cct": cct_data,
    "stroop": stroop_data,
    "motor": motor_data,
}

task_names = ["ant", "dd", "cct", "stroop", "motor"]


def load_and_prepare_embeddings(path, task_data, task_name):
    """
    Load participant embeddings, map participant_id back to worker_id,
    and rename embedding columns by task.
    """
    df = pd.DataFrame(np.load(path))

    # Important assumption:
    # embedding row index == participant_id used during training
    df["participant_id"] = df.index

    id_map = task_data[["participant_id", "worker_id"]].drop_duplicates()

    df = (
        df.merge(id_map, on="participant_id", how="left")
          .drop(columns=["participant_id"])
    )

    emb_cols = [c for c in df.columns if c != "worker_id"]
    rename_dict = {c: f"{task_name}_{c}" for c in emb_cols}

    df = df[["worker_id"] + emb_cols].rename(columns=rename_dict)

    return df


embedding_dfs = []

for task_name, path in embedding_paths.items():
    df_task = load_and_prepare_embeddings(
        path=path,
        task_data=task_data_dict[task_name],
        task_name=task_name,
    )
    embedding_dfs.append(df_task)


# Inner merge keeps only participants who have embeddings in all 5 tasks
all_embeddings = reduce(
    lambda left, right: left.merge(right, on="worker_id", how="inner"),
    embedding_dfs
)

print(f"Number of participants with embeddings in all five tasks: {all_embeddings.shape[0]}")


# ------------------------------------------------------------
# Extract raw task matrices
# ------------------------------------------------------------

raw_task_embeddings = {}

for task in task_names:
    task_cols = [c for c in all_embeddings.columns if c.startswith(f"{task}_")]

    # Sort columns by dimension number:
    # ant_0, ant_1, ..., ant_7
    task_cols = sorted(task_cols, key=lambda x: int(x.split("_")[-1]))

    raw_task_embeddings[task] = all_embeddings[task_cols].to_numpy()

emb_dim = raw_task_embeddings["ant"].shape[1]
print("Embedding dimension:", emb_dim)

for task in task_names:
    print(task, raw_task_embeddings[task].shape)


# ------------------------------------------------------------
# Align all tasks to ANT using orthogonal Procrustes
# ------------------------------------------------------------

def standardize(E):
    return StandardScaler().fit_transform(E)


def align_to_reference(E_ref, E_target):
    """
    Align E_target to E_ref using orthogonal Procrustes.

    E_ref and E_target:
        n_participants × embedding_dim

    Participants must already be in the same order.
    """
    E_ref_z = standardize(E_ref)
    E_target_z = standardize(E_target)

    R, scale = orthogonal_procrustes(E_target_z, E_ref_z)

    E_target_aligned = E_target_z @ R

    return E_target_aligned, R, scale


reference_task = "cct"

aligned_task_embeddings = {}
rotation_matrices = {}
alignment_scales = {}

E_ant_ref = standardize(raw_task_embeddings[reference_task])
aligned_task_embeddings[reference_task] = E_ant_ref

for task in task_names:
    if task == reference_task:
        continue

    E_aligned, R, scale = align_to_reference(
        E_ref=raw_task_embeddings[reference_task],
        E_target=raw_task_embeddings[task]
    )

    aligned_task_embeddings[task] = E_aligned
    rotation_matrices[task] = R
    alignment_scales[task] = scale

    print(f"Aligned {task} to {reference_task}; Procrustes scale = {scale:.4f}")


# Shape:
# n_participants × n_tasks × n_dims
E = np.stack(
    [aligned_task_embeddings[task] for task in task_names],
    axis=1
)

print("Aligned E shape:", E.shape)

task_names = ["ant", "dd", "cct", "stroop", "motor"]

n, n_tasks, n_dims = E.shape

# Reorder as: dim0 all tasks, dim1 all tasks, ...
X_dim_first = []

col_names_dim_first = []

for d in range(n_dims):
    for t, task in enumerate(task_names):
        X_dim_first.append(E[:, t, d])
        col_names_dim_first.append(f"dim{d}_{task}")

X_dim_first = np.column_stack(X_dim_first)

corr_dim_first = pd.DataFrame(
    X_dim_first,
    columns=col_names_dim_first
).corr(method="pearson")

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_dim_first,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.3
)

plt.title("Cross-Task Embedding Correlation Matrix, Grouped by Dimension")
plt.tight_layout()
plt.savefig("./Figures/Cross_Task_Correlation_Matrix_Grouped_By_Dimension.png", dpi=300)
plt.show()


def standardize_by_task_dim(E):
    """
    Standardize each task-dimension column across participants.

    E shape:
        n_participants × n_tasks × n_dims
    """
    E_z = E.copy().astype(float)

    mean = E_z.mean(axis=0, keepdims=True)
    std = E_z.std(axis=0, keepdims=True, ddof=1)

    E_z = (E_z - mean) / std

    return E_z


def icc_3_1(Y):
    """
    ICC(3,1): two-way mixed, consistency, single measurement.

    Y shape:
        n_participants × n_tasks

    In your case:
        rows = participants
        columns = tasks
    """
    Y = np.asarray(Y, dtype=float)

    n, k = Y.shape

    grand_mean = Y.mean()
    row_means = Y.mean(axis=1, keepdims=True)
    col_means = Y.mean(axis=0, keepdims=True)

    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((Y - grand_mean) ** 2)

    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)

    return icc


def omnibus_icc_stat(E):
    """
    Compute ICC for each embedding dimension,
    then average across dimensions.

    E shape:
        n_participants × n_tasks × n_dims
    """
    iccs = []

    for d in range(E.shape[2]):
        Y_d = E[:, :, d]  # n_participants × n_tasks
        iccs.append(icc_3_1(Y_d))

    iccs = np.array(iccs)

    return iccs.mean(), iccs


E_z = standardize_by_task_dim(E)

obs_mean_icc, obs_dim_iccs = omnibus_icc_stat(E_z)

print("Observed mean ICC:", obs_mean_icc)
print("Dimension-wise ICCs:", obs_dim_iccs)

def permutation_test_omnibus_icc(E, n_perm=5000, seed=123):
    """
    Permutation test for cross-task embedding consistency.

    Null:
        participant alignment across tasks is arbitrary.

    Alternative:
        the same participants show stable embedding values across tasks.
    """
    rng = np.random.default_rng(seed)

    E_z = standardize_by_task_dim(E)

    obs_mean_icc, obs_dim_iccs = omnibus_icc_stat(E_z)

    perm_stats = []

    for _ in range(n_perm):
        E_perm = E_z.copy()

        # Keep ANT fixed.
        # Permute participant labels in DD, CCT, Stroop, Motor.
        for t in range(1, E_perm.shape[1]):
            perm_idx = rng.permutation(E_perm.shape[0])
            E_perm[:, t, :] = E_perm[perm_idx, t, :]

        perm_mean_icc, _ = omnibus_icc_stat(E_perm)
        perm_stats.append(perm_mean_icc)

    perm_stats = np.array(perm_stats)

    p_value = (1 + np.sum(perm_stats >= obs_mean_icc)) / (1 + n_perm)

    return {
        "observed_mean_icc": obs_mean_icc,
        "dimension_iccs": obs_dim_iccs,
        "perm_stats": perm_stats,
        "p_value": p_value,
    }


icc_results = permutation_test_omnibus_icc(E, n_perm=5000, seed=123)

print("Observed mean ICC:", icc_results["observed_mean_icc"])
print("Dimension-wise ICCs:", icc_results["dimension_iccs"])
print("Permutation p-value:", icc_results["p_value"])


plt.figure(figsize=(8, 6))

plt.hist(
    icc_results["perm_stats"],
    bins=50,
    alpha=0.8
)

plt.axvline(
    icc_results["observed_mean_icc"],
    linestyle="--",
    linewidth=2,
    label=f"Observed mean ICC = {icc_results['observed_mean_icc']:.3f}"
)

plt.xlabel("Mean ICC under permutation")
plt.ylabel("Frequency")
plt.title("Permutation Test for Cross-Task Embedding Consistency")
plt.legend()
plt.tight_layout()
plt.savefig("./Figures/Omnibus_ICC_Permutation_Test.png", dpi=300)
plt.show()


# ======================================================================================================================
def safe_corr(x, y, method="pearson"):
    """
    Correlation between two vectors.
    """
    if method == "pearson":
        r, _ = pearsonr(x, y)
    elif method == "spearman":
        r, _ = spearmanr(x, y)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    return r


def fisher_mean_r(r_values):
    """
    Average correlations using Fisher z transform.
    """
    r_values = np.asarray(r_values, dtype=float)

    # Avoid infinite arctanh values
    r_values = np.clip(r_values, -0.999999, 0.999999)

    mean_z = np.nanmean(np.arctanh(r_values))
    mean_r = np.tanh(mean_z)

    return mean_r, mean_z


def task_pair_stat(E1, E2, method="pearson"):
    """
    Compute dimension-wise same-dimension correlations
    between two task embedding matrices.

    E1, E2 shape:
        n_participants × n_dims
    """
    n_dims = E1.shape[1]

    dim_rs = []

    for d in range(n_dims):
        r = safe_corr(E1[:, d], E2[:, d], method=method)
        dim_rs.append(r)

    dim_rs = np.array(dim_rs)

    mean_r, mean_z = fisher_mean_r(dim_rs)

    return mean_r, mean_z, dim_rs


def task_pair_permutation_test(
    E,
    task_names,
    method="pearson",
    n_perm=5000,
    alternative="greater",
    seed=123
):
    """
    Task-pairwise permutation test.

    E shape:
        n_participants × n_tasks × n_dims

    For each task pair:
        1. compute same-dimension correlations
        2. summarize using Fisher-z mean
        3. permute participant rows in one task
        4. compute p-value

    alternative:
        "greater" = test whether observed correlation is more positive than chance
        "two-sided" = test whether observed absolute association is stronger than chance
    """
    rng = np.random.default_rng(seed)

    results = []

    for i, j in combinations(range(E.shape[1]), 2):
        task1 = task_names[i]
        task2 = task_names[j]

        E1 = E[:, i, :]
        E2 = E[:, j, :]

        obs_mean_r, obs_mean_z, obs_dim_rs = task_pair_stat(
            E1, E2, method=method
        )

        perm_stats_z = []

        for _ in range(n_perm):
            perm_idx = rng.permutation(E.shape[0])

            # Permute participant rows in task2 only
            E2_perm = E2[perm_idx, :]

            _, perm_mean_z, _ = task_pair_stat(
                E1, E2_perm, method=method
            )

            perm_stats_z.append(perm_mean_z)

        perm_stats_z = np.array(perm_stats_z)

        if alternative == "greater":
            p_value = (1 + np.sum(perm_stats_z >= obs_mean_z)) / (1 + n_perm)
        elif alternative == "two-sided":
            p_value = (1 + np.sum(np.abs(perm_stats_z) >= np.abs(obs_mean_z))) / (1 + n_perm)
        else:
            raise ValueError("alternative must be 'greater' or 'two-sided'")

        row = {
            "task1": task1,
            "task2": task2,
            "mean_r": obs_mean_r,
            "mean_z": obs_mean_z,
            "p_value": p_value,
        }

        # Save dimension-wise correlations
        for d, r in enumerate(obs_dim_rs):
            row[f"dim{d}_r"] = r

        results.append(row)

    results_df = pd.DataFrame(results)

    # FDR correction across the 10 task-pair tests
    results_df["p_fdr"] = multipletests(
        results_df["p_value"],
        method="fdr_bh"
    )[1]

    return results_df


task_names = ["ant", "dd", "cct", "stroop", "motor"]

pairwise_results = task_pair_permutation_test(
    E=E,
    task_names=task_names,
    method="pearson",
    n_perm=5000,
    alternative="greater",
    seed=123
)

pairwise_results = pairwise_results.sort_values("mean_r", ascending=False)

print(pairwise_results)