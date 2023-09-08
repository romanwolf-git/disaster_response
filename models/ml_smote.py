import numpy as np
import pandas as pd
import random
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors


def get_tail_label(df: pd.DataFrame, ql=[0.05, 1.]) -> list:
    """
    Find the underrepresented targets.
    Underrepresented targets are those observed less than the median frequency.
    Targets above a quantile limit are filtered.
    """
    irlbl = df.sum(axis=0)
    irlbl = irlbl[(irlbl > irlbl.quantile(ql[0])) & ((irlbl < irlbl.quantile(ql[1])))]  # Filtering
    irlbl = irlbl.max() / irlbl
    threshold_irlbl = irlbl.median()
    tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()
    return tail_label


def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame, ql=[0.05, 1.]):
    """
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    tail_labels = get_tail_label(y, ql=ql)
    index = y[y[tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()

    X_sub = X[X.index.isin(index)].reset_index(drop=True)
    y_sub = y[y.index.isin(index)].reset_index(drop=True)
    return X_sub, y_sub


def nearest_neighbour(X: pd.DataFrame, neigh) -> list:
    """
    Give index of 10 nearest neighbor of all the instance

    args
    X: np.array, array whose nearest neighbor has to find

    return
    indices: list of list, index of 5 NN of each element in X
    """

    nn_alg = 'kd_tree' if issparse(X) else 'brute'  # kd_tree can not be applied to a sparse matrix

    nbs = NearestNeighbors(
        n_neighbors=neigh,
        metric='euclidean',
        algorithm=nn_alg
    ).fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices


def ml_smote(X, y, n_sample=100, neigh=5):
    """
    Augmented data using multi-label SMOTE (MLSMOTE)

    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample

    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X, neigh=5)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n - 1)
        neighbor = random.choice(list(indices2[reference, 1:]))
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis=0, skipna=True)
        target[i] = np.array([1 if val > 0 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference, :] - X.loc[neighbor, :]
        new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    return new_X, target


def resample(X, y, n_sample):
    """
    Resample input data using multi-lable SMOTE
    :param X: features
    :param y: labels
    :param n_sample: number of samples to create
    :return: resampled features and labels as pandas data frames
    """

    if isinstance(X, csr_matrix):
        X = pd.DataFrame.sparse.from_spmatrix(X)
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X_sub, y_sub = get_minority_samples(X, y)                 # Get minority samples of dataframe
    X_res, y_res = ml_smote(X_sub, y_sub, n_sample=n_sample)  # Apply MLSMOTE to augment data
    X_con = pd.concat([X, X_res], ignore_index=True)
    y_con = pd.concat([y, y_res], ignore_index=True)

    return X_con, y_con