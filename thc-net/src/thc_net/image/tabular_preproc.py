import gzip
import io
import os
import json
import shutil
from pathlib import Path

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from thc_net.utils import do_parallel_numpy

MAX_MEMORY_USE = 1  # in Go

# https://he-arc.github.io/livre-python/pillow/index.html#methodes-de-dessin
# https://stackoverflow.com/questions/26649716/how-to-show-pil-image-in-ipython-notebook
# https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array
# line = np.array(pic, dtypes="uint8")
# from https://arxiv.org/pdf/1902.02160.pdf page 2


def save_numpy_as_image_gz(arr, path, one_channel=False, optimize=True):
    mode = "L" if one_channel else "RGB"

    im = Image.fromarray(arr, mode=mode)
    output = io.BytesIO()
    im.save(output, "jpeg", optimize=optimize)
    with gzip.open(path, "wb") as f:
        f.write(output.getvalue())
    return True


def format_number(nb):
    return np.format_float_scientific(
        nb, precision=9, unique=False, pad_left=None, exp_digits=2, sign=True
    )


def word_to_square_image(text, size, cut_length=None, one_channel=False):

    if not isinstance(text, str) and np.isfinite(text):
        text = format_number(text)
    truncated = text[:cut_length] if cut_length is not None else text
    max_x = np.ceil(np.sqrt(len(truncated))).astype("int")
    character_size = np.floor(size / max_x).astype("int")
    padding = np.floor((size - (max_x * character_size)) / 2).astype("int")
    # Do we need pt to px conversion ? Seems like not
    # font_size =  int(np.floor(character_size*0.75))
    # font_url = "https://ff.static.1001fonts.net/r/o/roboto-condensed.regular.ttf"

    font_size = character_size
    out_font = Path(os.getcwd()) / f"RobotoCondensed-Regular.ttf"

    fnt = ImageFont.truetype(out_font.as_posix(), font_size)

    # 1 (1-bit pixels, black and white, stored with one pixel per byte)
    # L (8-bit pixels, black and white)
    # RGB (3x8-bit pixels, true color)
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    mode = "L" if one_channel else "RGB"
    WHITE = 255 if one_channel else (255, 255, 255)
    BLACK = 0 if one_channel else (0, 0, 0)

    image = Image.new(mode, (size, size), BLACK)
    # Obtention du contexte graphique
    draw = ImageDraw.Draw(image)
    x = 0
    y = 0
    for letter in truncated:
        draw.text(
            (padding + x * character_size, padding + y * character_size),
            letter,
            font=fnt,
            fill=WHITE,
        )
        if x + 1 < max_x:
            x += 1
        else:
            y += 1
            x = 0
    return np.array(image)


def features_to_square_image(
    features, image_size=224, cut_length=None, one_channel=False
):
    nb_channel = 1 if one_channel else 3
    square_nb = np.ceil(np.sqrt(len(features))).astype("int")
    word_size = np.floor(image_size / square_nb).astype("int")
    max_features = len(features)
    padding = np.floor((image_size - square_nb * word_size) / 2).astype("int")
    if one_channel:
        result_image = np.zeros((image_size, image_size), dtype="uint8")
    else:
        result_image = np.zeros((image_size, image_size, nb_channel), dtype="uint8")
    i_feature = 0
    features_str = features.astype("str")
    for x in range(0, square_nb):
        if i_feature is None:
            break
        for y in range(0, square_nb):
            i_feature = x * (square_nb) + y
            if i_feature >= max_features:
                i_feature = None
                break
            x_pos = x * word_size + padding
            y_pos = y * word_size + padding
            result_image[
                x_pos : x_pos + word_size, y_pos : y_pos + word_size
            ] = word_to_square_image(
                features_str[i_feature],
                size=word_size,
                cut_length=cut_length,
                one_channel=one_channel,
            )
    return result_image


def features_to_square_image_params(values, params):
    return features_to_square_image(
        values,
        image_size=params["image_size"],
        cut_length=params["cut_length"],
        one_channel=params["one_channel"],
    )


def compute_sizes_(used_columns, feature_size, image_size):
    square_side_nb_feature = np.ceil(np.sqrt(len(used_columns))).astype("int")
    image_size = (
        square_side_nb_feature * feature_size if image_size is None else image_size
    )
    feature_size = np.floor(image_size / square_side_nb_feature).astype("int")
    return feature_size, image_size


def compute_chunksize_(used_columns, feature_size, nb_channel):
    square_side_nb_feature = np.ceil(np.sqrt(len(used_columns))).astype("int")
    memory_image_size = (
        square_side_nb_feature ** 2 * feature_size ** 2 * nb_channel
    )  # in bytes
    chunk_size = np.floor((MAX_MEMORY_USE * 1024 ** 3) / memory_image_size).astype(
        "int"
    )
    return chunk_size


def get_classes_nb_lines(dataset_path, target, panda_kwargs={}):
    target_values = pd.read_csv(
        dataset_path, **panda_kwargs, usecols=[target]
    ).values.reshape(-1)
    return np.unique(target_values).astype("str"), target_values.shape[0]


def csv_to_image_file(
    dataset_path,
    train_indices,
    valid_indices,
    out_folder,
    used_columns,
    target,
    feature_size=32,
    image_size=None,
    nb_channel=1,
    panda_kwargs={},
):

    if feature_size is not None and image_size is not None:
        raise ValueError("Only feature_size or image_size can be defined at once")

    feature_size, image_size = compute_sizes_(used_columns, feature_size, image_size)
    chunk_size = compute_chunksize_(used_columns, feature_size, nb_channel)

    classnames, nb_lines = get_classes_nb_lines(dataset_path, target, panda_kwargs)

    params = {
        "image_size": image_size,
        "cut_length": None,
        "one_channel": nb_channel == 1,
    }

    file_list = {
        "train": [],
        "valid": [],
        "test": [],
    }

    for set_label in ["train", "valid", "test"]:
        prep_data_folder = out_folder / f"prep_data/{image_size}/{set_label}/"
        if prep_data_folder.exists():
            shutil.rmtree(prep_data_folder)
        prep_data_folder.mkdir(parents=True, exist_ok=True)
        for classname in classnames:
            (prep_data_folder / classname).mkdir(parents=True, exist_ok=True)

    for i, chunk in tqdm(
        enumerate(
            pd.read_csv(
                dataset_path,
                **panda_kwargs,
                chunksize=chunk_size,
                usecols=used_columns + [target],
            )
        ),
        total=(nb_lines // chunk_size) + (1 if nb_lines % chunk_size > 0 else 0),
    ):
        X = chunk[used_columns].values
        Y = chunk[target].values.reshape(-1)
        image_X = do_parallel_numpy(features_to_square_image_params, [X], [params])

        chunk_list = []
        for j, label in enumerate(Y):
            idx = i * chunk_size + j
            set_label = (
                "train"
                if idx in train_indices
                else "valid"
                if idx in valid_indices
                else "test"
            )
            full_path = (
                out_folder
                / f"prep_data/{image_size}"
                / set_label
                / str(label)
                / (str(j + i * chunk_size) + ".jpeg.gz")
            ).as_posix()
            chunk_list.append(full_path)
            file_list[set_label].append(full_path)

        assert all(
            do_parallel_numpy(
                save_numpy_as_image_gz, [image_X, chunk_list], [nb_channel == 1]
            ).reshape(-1)
        )

    json_file = out_folder / f"prep_data/{image_size}/file_list.json"

    with json_file.open(mode="w") as fp:
        json.dump(file_list, fp)
    json_file

    classnames_file = out_folder / f"prep_data/{image_size}/classnames.json"

    with classnames_file.open(mode="w") as fp:
        json.dump(classnames.tolist(), fp)
    classnames_file


# Range for ord is 0 to 1,114,111
# So we need 3 value to write it in base 255
# Let's use this 3 values as 3 channels
# Each letter will be triplet [0,0,0]

import unicodedata


def normalize(word):
    return unicodedata.normalize("NFKC", word)


def word_to_unicode(word, ascii_only=False):
    nb_channel = [0] if ascii_only else [2, 1, 0]
    res = []
    for letter in word:  # NFKD ?
        int_letter = ord(letter)
        if ascii_only:
            if int_letter > 255:
                raise ValueError("Should be only ascii:" + str(int_letter))
            res.append([int_letter])
        else:
            res_letter = []
            for pow_i in nb_channel:
                res_letter.append(int_letter // (255 ** pow_i))
                int_letter = int_letter % (255 ** pow_i)
            res.append(res_letter)
    return res


def number_to_img(nb, ascii_only=False):
    nb_channel = 1 if ascii_only else 3
    word = format_number(nb)
    word = normalize(format_number(nb).rjust(16, " "))
    return np.array(
        word_to_unicode(word, ascii_only=ascii_only), dtype="uint8"
    ).reshape(4, 4, nb_channel)


def word_to_img(word, ascii_only=False):
    nb_channel = 1 if ascii_only else 3
    word = np.array(
        word_to_unicode(normalize(str(word))[:64], ascii_only=ascii_only), dtype="uint8"
    )
    res = np.zeros(shape=(64, nb_channel), dtype="uint8")
    res[: word.shape[0]] = word
    return res.reshape(8, 8, nb_channel)


def feature_to_square_image(feature, is_nb, ascii_only=False):
    if is_nb:
        return number_to_img(feature, ascii_only=ascii_only)
    return word_to_img(feature, ascii_only=ascii_only)


OTHER_FEATURE_SIZE = 8
NUMERIC_FEATURE_SIZE = 4


def compute_img_optimized_stats(is_numeric):
    # Number of numeric features
    nb_feat_num = np.sum(is_numeric)
    # Number of others features
    nb_feat_other = len(is_numeric) - nb_feat_num
    # Size of a side a square to place all other features (unit = feature)
    nb_big_square_side = np.ceil(np.sqrt(nb_feat_other)).astype("int")
    # Size of a side of a square to place all other features (unit = pixel)
    square_size = nb_big_square_side * OTHER_FEATURE_SIZE
    # Nb of empty square to use
    empty_squares = nb_big_square_side ** 2 - nb_feat_other

    # Define how much more space we need for numeric features
    # In one square for text, we can put 4 num features (OTHER_FEATURE_SIZE // NUMERIC_FEATURE_SIZE) ** 2

    left_num_feat = max(
        nb_feat_num - empty_squares * (OTHER_FEATURE_SIZE // NUMERIC_FEATURE_SIZE) ** 2,
        0,
    )

    # Since we want the image square, adding one row and one column for num feature allow us
    # to add 4*(2*nb_big_square_side-1) features
    # Adding n_row and columns, is equivalent to consuming "n_row*4*nb_big_square_side + n_row**2" features
    # => n_row*4*nb_big_square_side + n_row**2 -left_num_feat =0
    if left_num_feat > 0:
        n_row = 1
        while (
            n_row ** 2
            + 2
            * (OTHER_FEATURE_SIZE // NUMERIC_FEATURE_SIZE)
            * nb_big_square_side
            * n_row
            < left_num_feat
        ):
            n_row += 1
        #     n_row = np.ceil(max(np.roots([-left_num_feat, 4 * nb_big_square_side, 1]))).astype(
        #         "int"
        #     )
        square_size += n_row * 4
    else:
        n_row = 0
    return {
        "image_size": square_size,
        "nb_text_features": nb_feat_other,
        "nb_empty_text_squares": empty_squares,
        "text_square_side_size": nb_big_square_side,
        "n_row": n_row,
    }


def build_image(features, is_numeric, ascii_only=False):
    nb_channel = 1 if ascii_only else 3
    # Compute stats
    img_stats = compute_img_optimized_stats(is_numeric)
    # Build empty image
    result_image = np.zeros(
        shape=(img_stats["image_size"], img_stats["image_size"], nb_channel), dtype="uint8"
    )
    # Sort the index, so we get the text features first
    sorted_idx = np.argsort(is_numeric, kind="stable")

    x = 0
    y = 0

    # Let's handle text features
    # We place them as a square
    for idx in sorted_idx[: img_stats["nb_text_features"]]:
        result_image[
            y : y + OTHER_FEATURE_SIZE, x : x + OTHER_FEATURE_SIZE, :
        ] = feature_to_square_image(features[idx], is_numeric[idx], ascii_only=ascii_only)
        if (
            x + OTHER_FEATURE_SIZE
            >= img_stats["text_square_side_size"] * OTHER_FEATURE_SIZE
        ):
            x = 0
            y += OTHER_FEATURE_SIZE
        else:
            x += OTHER_FEATURE_SIZE
    # Let's remember last X and Y
    min_y = y
    min_x = x
    # Now, we need to fill the left squares from text with numeric features
    for idx in sorted_idx[
        img_stats["nb_text_features"] : img_stats["nb_text_features"]
        + NUMERIC_FEATURE_SIZE * img_stats["nb_empty_text_squares"]
    ]:
        result_image[
            y : y + NUMERIC_FEATURE_SIZE, x : x + NUMERIC_FEATURE_SIZE, :
        ] = feature_to_square_image(features[idx], is_numeric[idx], ascii_only=ascii_only)
        if (
            x + NUMERIC_FEATURE_SIZE
            >= img_stats["text_square_side_size"] * OTHER_FEATURE_SIZE
        ):
            x = 0 if y >= (min_y + NUMERIC_FEATURE_SIZE) else min_x
            y += NUMERIC_FEATURE_SIZE
        else:
            x += NUMERIC_FEATURE_SIZE
    del x, y
    min_rg = img_stats["text_square_side_size"] * 2 * NUMERIC_FEATURE_SIZE

    # Finally, if we still have some features, we need to add it as new rows, and new columns
    previous = (
        img_stats["nb_text_features"]
        + NUMERIC_FEATURE_SIZE * img_stats["nb_empty_text_squares"]
    )
    step = img_stats["image_size"] // NUMERIC_FEATURE_SIZE
    for i_row in range(0, img_stats["n_row"]):
        line_list = []
        for i in range(previous, min(previous + step, len(features))):
            idx = sorted_idx[i]
            line_list.append(feature_to_square_image(features[idx], is_numeric[idx], ascii_only=ascii_only))

        if len(line_list) > 0:
            line = np.hstack(line_list)
            result_image[
                min_rg
                + i_row * NUMERIC_FEATURE_SIZE : min_rg
                + (i_row + 1) * NUMERIC_FEATURE_SIZE,
                : line.shape[1],
            ] = line

        previous += step

        row = []
        for i in range(
            previous, min(previous + step - img_stats["n_row"], len(features))
        ):
            idx = sorted_idx[i]
            row.append(feature_to_square_image(features[idx], is_numeric[idx], ascii_only=ascii_only))
        if len(row) > 0:
            row = np.vstack(row)
            result_image[
                : row.shape[0],
                min_rg
                + i_row * NUMERIC_FEATURE_SIZE : min_rg
                + (i_row + 1) * NUMERIC_FEATURE_SIZE,
            ] = row

        previous += step - img_stats["n_row"]

    return result_image

from sklearn.preprocessing import StandardScaler

def csv_to_pixel(df, used_columns, target, ascii_only=False):


    extract_df = df[used_columns]
    nb_idx = ((extract_df.dtypes == "int64") | (extract_df.dtypes == "float64")) & ~((extract_df.nunique() / extract_df.shape[0]) < 0.005)
    nb_idx = extract_df.columns[nb_idx]
    is_numeric = np.isin(extract_df.columns.values, nb_idx.values)

    cat_cols = extract_df.columns[~is_numeric]
    num_cols = extract_df.columns[is_numeric]
    #  == True
    del extract_df, nb_idx
    # is_numeric
    # is_numeric = [False for col in range(X.shape[1])]
    # build_image(features, is_numeric)

    X = df[cat_cols].values
    Y = df[target].values.reshape(-1)

    image_X = do_parallel_numpy(build_image, [X], [[False for col in cat_cols], ascii_only])
    # image_X.shape

    return image_X, StandardScaler().fit_transform(df[num_cols].values), Y
