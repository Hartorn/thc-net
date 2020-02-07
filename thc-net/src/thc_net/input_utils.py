import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from thc_net.safe_label_encoder import SafeLabelEncoder


def detect_cat(X, ratio=0.005):
    """
    Detect categories base on type and on a ratio between n_unique and sample_size
    
    Parameters
    ----------
    X : DataFrame
        the dataframe with the data
    ratio : float, optional
        ratio between n_unique and size of df, by default 0.005
    
    Returns
    -------
    list of index for categories, list of number of modalities for categories (number +1 for unknown)
        list with index and dims
    """
    n_unique = X.nunique()
    ratios = (n_unique / X.shape[0]) < ratio
    cat_idxs = np.argwhere(X.columns.isin(X.columns[ratios | (X.dtypes == 'object')])).ravel()
    cat_dims = n_unique[cat_idxs].values + X.isnull().sum()[cat_idxs].values + 1
    return cat_idxs, cat_dims


def prepare_input_data(x, cat_idx, encoders=None, fit=True):
    """
    Prepare data for training : encode categoricals columns, scale the rest.
    Fit the encoders if fit is True.
    
    Parameters
    ----------
    x : numpy matrice
        data to prepare
    cat_idx : list of idx
        list of index
    encoders : list, optional
        list of encoders to apply (cat, then one other), by default None
    fit : bool, optional
        if encoders should be fitted, by default True
    
    Returns
    -------
    tuple of numpy matrice, encoders list
        transformed data and encoders
    """
    results = np.zeros(shape=x.shape)
    if encoders is None:
        encoders = []

    if fit:
        for idx in cat_idx:
            lb_enc = SafeLabelEncoder()
            lb_enc.fit(x[:, idx])
            encoders.append(lb_enc)

    non_cat_idx = list(set(range(x.shape[1])) - set(cat_idx))
    if len(non_cat_idx) > 0:
        if fit:
            scaler = Pipeline(
                steps=[("fillna", SimpleImputer()), ("scaler", StandardScaler())]
            )
            scaler.fit(x[:, non_cat_idx])
            encoders.append(scaler)

    for i, idx in enumerate(cat_idx):
        results[:, idx] = encoders[i].transform(x[:, idx])

    if len(non_cat_idx) > 0:
        results[:, non_cat_idx] = encoders[-1].transform(x[:, non_cat_idx])

    return results, encoders

