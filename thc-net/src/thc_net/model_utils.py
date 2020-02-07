from tensorflow.keras.layers import (
    Dense,
    Concatenate,
    Input,
    Dropout,
    BatchNormalization,
    SpatialDropout1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow_addons.activations import mish
from thc_net.net_utils import (
    add_dense_block,
    add_snn_block,
    build_ranger_optimizer,
    create_embeddings,
    create_snn_embeddings,
)

SEED = 42


# config
# layers = [size, activation, dropout, dropout_layer, batch_norm]
# output = [size, activation]
# model = [ optimizer loss metrics]

def build_keras_snn(
    n_layer,
    in_dim,
    out_dim,
    mul_input,
    loss,
    activation,
    metrics=[],
    cat_emb_dims=[],
    dropout=0.5,
    max_emb=50,
    noise=0.1,
):

    inputs, x, new_in_dim = create_snn_embeddings(
        in_dim, cat_emb_dims, dropout, max_emb, noise
    )

    first_layer_size = int(new_in_dim * mul_input)

    for i_layer in range(n_layer, 0, -1):
        layer_size = int(((first_layer_size - out_dim) / n_layer) * i_layer + out_dim)
        x = add_snn_block(x=x, layer_size=layer_size, dropout=dropout)

    output = Dense(out_dim, activation=activation)(x)

    optimizer = build_ranger_optimizer()
    model = Model(inputs, output)
    model.compile(optimizer=optimizer, loss=[loss], metrics=metrics)

    return model


def build_keras_mlp(
    n_layer,
    in_dim,
    out_dim,
    mul_input,
    loss,
    activation,
    metrics=[],
    cat_emb_dims=[],
    dropout=0.5,
    normalize=False,
    max_emb=50,
    noise=0.1,
    hidden_activation=mish
):

    inputs, x, new_in_dim = create_embeddings(
        in_dim, cat_emb_dims, dropout, max_emb, noise
    )

    if normalize:
        x = BatchNormalization()(x)

    first_layer_size = int(new_in_dim * mul_input)

    for i_layer in range(n_layer, 0, -1):
        layer_size = int(((first_layer_size - out_dim) / n_layer) * i_layer + out_dim)
        x = add_dense_block(
            x=x,
            layer_size=layer_size,
            dropout=dropout,
            activation=hidden_activation,
            normalize=normalize,
        )

    output = Dense(out_dim, activation=activation)(x)

    optimizer = build_ranger_optimizer()
    model = Model(inputs, output)
    model.compile(optimizer=optimizer, loss=[loss], metrics=metrics)

    return model
