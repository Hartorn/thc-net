import numpy as np
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor as PoolExecutor

from thc_net.safe_label_encoder import SafeLabelEncoder
from sklearn.impute import SimpleImputer


def word_to_np_array(word, cut_length):
    result = np.zeros(cut_length, dtype="uint8")
    for i, letter in enumerate(word[:cut_length]):
        result[i] = ord(letter)
    return result


def line_to_2darray(line, cut_length):
    result = np.zeros((line.shape[0], cut_length), dtype="uint8")
    for i in range(line.shape[0]):
        result[i] = word_to_np_array(line[i], cut_length)
    return result


def do_parallel_numpy(map_func, iter_params, constant_params=None):
    repeated_params = (
        [] if constant_params is None else list(map(repeat, constant_params))
    )
    results = None
    with PoolExecutor() as executor:
        results = np.stack(
            list(executor.map(map_func, *iter_params, *repeated_params)), axis=0
        )
    return results


def format_number(nb):
    if not np.isfinite(nb):
        return str(nb)
    return np.format_float_scientific(
        nb, precision=9, unique=False, pad_left=None, exp_digits=2, sign=True
    )


def preproc_dataset(train_df, target=None, ids=None, params=None):
    if params is None:
        n_unique = train_df.nunique()

    params = params if params is not None else {}
    to_ignore = []

    if target is not None:
        to_ignore.append(target)

    if ids is not None:
        to_ignore.extend(ids)

    if "constant_cols" not in params:
        constant_cols = train_df.columns[n_unique <= 1]
        constant_cols = list(set(constant_cols.tolist()) - set(to_ignore))
        params["constant_cols"] = constant_cols

    if "bool_cols" not in params:
        bool_cols = train_df.columns[n_unique == 2]
        bool_cols = list(set(bool_cols.tolist()) - set(to_ignore))
        params["bool_cols"] = bool_cols
    if "num_cols" not in params:
        num_cols = list(
            set(
                train_df.columns[
                    (n_unique > 2) & (train_df.dtypes != "object")
                ].tolist()
            )
            - set(to_ignore)
        )
        params["num_cols"] = num_cols

    if "cat_cols" not in params:
        cat_cols = list(
            set(train_df.columns.tolist())
            - set(num_cols)
            - set(bool_cols)
            - set(constant_cols)
            - set(to_ignore)
        )
        params["cat_cols"] = cat_cols

    # Let's handle numeric columns
    X_num_values = train_df[params["num_cols"]].values

    if "num_encoder" not in params:
        #  Let's calculate fillna for num columns
        fillna_values = (
            train_df[params["num_cols"]].min() - train_df[params["num_cols"]].std() / 10
        )
        params["num_encoder"] = []
        for i in range(len(params["num_cols"])):
            enc = SimpleImputer(strategy="constant", fill_value=fillna_values[i])
            enc.fit(X_num_values[:, i].reshape(-1, 1))
            params["num_encoder"].append(enc)

    for i, enc in enumerate(params["num_encoder"]):
        X_num_values[:, i] = (
            enc.transform(X_num_values[:, i].reshape(-1, 1)).reshape(-1).astype("float")
        )

    # Let's handle boolean columns
    X_bool_values = train_df[params["bool_cols"]].values

    if "bool_encoder" not in params:
        params["bool_encoder"] = []
        for i in range(len(params["bool_cols"])):
            enc = SafeLabelEncoder()
            enc.fit(X_bool_values[:, i].reshape(-1))
            params["bool_encoder"].append(enc)

    for i, enc in enumerate(params["bool_encoder"]):
        X_bool_values[:, i] = (
            enc.transform(X_bool_values[:, i].reshape(-1)).reshape(-1).astype("uint")
        )

    #  For cat cols, let's strip spaces
    X_cat_values = np.char.strip(train_df[params["cat_cols"]].values.astype("str"))
    # Now, let's calculate the number of "channels" needed (max string length)
    if "nb_channels" not in params and len(cat_cols) > 0:
        params["nb_channels"] = np.vectorize(len)(X_cat_values).max()
    elif "nb_channels" not in params:
        params["nb_channels"] = 0
    # Finally, let's transform it into 1d array
    X_cat_values = do_parallel_numpy(
        line_to_2darray, [X_cat_values], [params["nb_channels"]]
    )

    X_bool_values = X_bool_values.astype("uint8")

    to_return = []
    if len(params["bool_cols"]) > 0:
        to_return.append(X_bool_values)
    if len(params["num_cols"]) > 0:
        to_return.append(X_num_values)
    if len(params["cat_cols"]) > 0:
        to_return.append(X_cat_values)

    return to_return, params
