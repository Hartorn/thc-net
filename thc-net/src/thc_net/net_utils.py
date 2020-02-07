from tensorflow.keras.layers import (
    AlphaDropout,
    Dropout,
    Dense,
    BatchNormalization,
    Embedding,
    Reshape,
    Input,
    SpatialDropout1D,
    GaussianNoise,
    Concatenate,
)
from tensorflow.keras import Sequential
from tensorflow.keras.activations import selu

from tensorflow_addons.optimizers import RectifiedAdam, Lookahead
import numpy as np

SEED = 42


def create_snn_embeddings(
    in_dim, cat_emb_dims, dropout, max_emb, noise=None,
):
    return create_embeddings(
        in_dim,
        cat_emb_dims,
        dropout,
        max_emb,
        noise=noise,
        embeddings_initializer="lecun_normal",
        dropout_layer=alpha_dropout,
    )


def create_embeddings(
    in_dim,
    cat_emb_dims,
    dropout,
    max_emb,
    noise=None,
    embeddings_initializer="he_normal",
    dropout_layer=SpatialDropout1D,
):
    inputs = []
    embeddings = []
    total_size = 0
    for dim in cat_emb_dims:
        input = Input(shape=(1,))
        embedding_size = int(min(np.ceil((dim) / 2), max_emb))
        total_size += embedding_size
        embedding = Embedding(
            dim,
            embedding_size,
            input_length=1,
            embeddings_initializer=embeddings_initializer,
        )(input)
        embeddings.append(embedding)
        inputs.append(input)

    left_features = in_dim - len(cat_emb_dims)
    if left_features > 0:
        input = Input(left_features)
        inputs.append(input)
        if noise is not None:
            input = GaussianNoise(noise)(input)

    if len(cat_emb_dims) > 0:
        emb_layer = Concatenate()(embeddings)
        if dropout_layer == SpatialDropout1D and dropout is not None:
            emb_layer = dropout_layer(dropout)(emb_layer)
        emb_layer = Reshape(target_shape=(total_size,))(emb_layer)
        if (
            dropout_layer != SpatialDropout1D
            and dropout_layer != None
            and dropout is not None
        ):
            emb_layer = dropout_layer(dropout)(emb_layer)

        if left_features > 0:
            emb_layer = Concatenate()([emb_layer, input])
    else:
        emb_layer = input
    return inputs, emb_layer, total_size + in_dim - len(cat_emb_dims)


def alpha_dropout(rate):
    return AlphaDropout(rate=rate, seed=SEED)


def add_dense_block(
    x,
    layer_size,
    activation,
    dropout,
    dropout_layer=Dropout,
    normalize=False,
    dense_options={"kernel_initializer": "he_normal"},
):
    x = Dense(layer_size, activation=activation, **dense_options)(x)
    if normalize:
        x = BatchNormalization()(x)
    if dropout is not None:
        x = dropout_layer(dropout)(x)
    return x


def add_snn_block(x, layer_size, dropout, dense_options={}):
    return add_dense_block(
        x=x,
        layer_size=layer_size,
        activation=selu,
        dropout=dropout,
        dropout_layer=alpha_dropout,
        normalize=False,
        dense_options={"kernel_initializer": "lecun_normal", **dense_options},
    )


def build_ranger_optimizer(
    radam_options={}, lookahead_options={"sync_period": 6, "slow_step_size": 0.5}
):
    radam = RectifiedAdam(**radam_options)
    ranger = Lookahead(radam, **lookahead_options)
    return ranger
