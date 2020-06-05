import copy

import numpy as np

from tensorflow.keras import Model, Input

from tensorflow.keras.layers import (
    LocallyConnected1D,
    BatchNormalization,
    Activation,
    Reshape,
    Concatenate,
    Dense,
)
from tensorflow_addons.activations import mish
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead


DEFAULT_CONV_OPTS = {
    "padding": "valid",
    "kernel_size": 1,
    "strides": 1,
    "use_bias": False,
    "activation": None,
}


def add_local_conv_block(
    in_layer,
    lconv_dim,
    prefix,
    activation,
    use_bn=True,
    activation_first_layer=None,
    options=None,
):
    if options is None:
        options = {}

    x_num_layer = in_layer
    for i, lconv_layer in enumerate(lconv_dim):
        name = f"{prefix}_{i}_"
        layer_opts = copy.deepcopy(DEFAULT_CONV_OPTS)
        layer_opts.update(options)

        x_num_layer = LocallyConnected1D(
            filters=lconv_layer, name=name + "conv", **layer_opts
        )(x_num_layer)
        if use_bn:
            x_num_layer = BatchNormalization(name=name + "nb")(x_num_layer)
        temp_activation = (
            activation_first_layer
            if i == 0 and activation_first_layer is not None
            else activation
        )
        x_num_layer = Activation(temp_activation, name=name + "activation")(x_num_layer)

    return x_num_layer


def build_optimizer():
    return Lookahead(RectifiedAdam(1e-3), sync_period=6, slow_step_size=0.5)


def build_model(
    params, lconv_dim=[], lconv_num_dim=[], activation=None, optimizer=None
):
    if optimizer is None:
        optimizer = build_optimizer()
    if activation is None:
        activation = mish

    # Here, we get info necessary to build the model
    input_cat_dim = len(params["cat_cols"])
    input_bool_dim = len(params["bool_cols"])
    input_num_dim = len(params["num_cols"])
    nb_channels = params["nb_channels"]

    # Inputs of the model
    inputs = []
    #  Layers to concat before output
    concats = []

    # Handling booleans
    if input_bool_dim > 0:
        input_bool_layer = Input(shape=(input_bool_dim,), name="input_bool")
        inputs.append(input_bool_layer)
        concats.append(input_bool_layer)

    # Handling numeric
    if input_num_dim > 0:
        input_num_layer = Input(shape=(input_num_dim,), name="input_num")
        inputs.append(input_num_layer)
        x_num_layer = input_num_layer

        if len(lconv_num_dim) != 0 and input_num_dim > 0:
            x_num_layer = Reshape((input_num_dim, 1), name="reshape_num_input")(
                x_num_layer
            )

        x_num_layer = add_local_conv_block(
            x_num_layer,
            lconv_num_dim,
            "block_num",
            activation,
            use_bn=False,
            activation_first_layer="tanh",
            options=None,
        )

        nb_filters = lconv_num_dim[-1] if len(lconv_num_dim) > 0 else 1
        x_num_layer = Reshape((input_num_dim * nb_filters,), name="reshape_num_output")(
            x_num_layer
        )
        concats.append(x_num_layer)

    # Handling cat
    if input_cat_dim > 0:

        input_cat_layer = Input(shape=(input_cat_dim, nb_channels), name="input_cat")
        inputs.append(input_cat_layer)

        x_layer = input_cat_layer
        x_layer = add_local_conv_block(
            x_layer,
            lconv_dim,
            "block_cat",
            activation,
            use_bn=True,
            activation_first_layer=None,
            options=None,
        )

        nb_filters = lconv_dim[-1] if len(lconv_dim) > 0 else 1
        x_layer = Reshape((input_cat_dim * nb_filters,), name="reshape_cat_output")(
            x_layer
        )

        concats.append(x_layer)

    if len(concats) > 1:
        concat = Concatenate()(concats)
    else:
        concat = concats[0]

    # For now, output is only for binary classification
    output = Dense(1, activation="sigmoid", name="output")(concat)

    model = Model(inputs=inputs, outputs=[output], name="explainable_model",)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    return model


def predict(model, input_model):
    # preds, expl_boo, expl_num, expl_others
    # model.get_layer("input_bool")
    # model.get_layer("reshape_num_input")
    # model.get_layer("reshape_cat_output")
    # input_cat_dim = len(params["cat_cols"])
    # input_bool_dim = len(params["bool_cols"])
    # input_num_dim = len(params["num_cols"])

    log_reg_weights = model.get_layer("output").get_weights()[0]
    log_reg_bias = model.get_layer("output").get_weights()[1]

    outputs = []
    shapes = []
    weights = []

    layers_names = [layer.name for layer in model.layers]

    consumed = 0

    for name in ["input_bool", "reshape_num_output", "reshape_cat_output"]:
        if name not in layers_names:
            continue
        layer = model.get_layer(name)
        outputs.append(layer.output)
        input_shape = layer.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        print(input_shape)

        nb_channel = (
            input_shape[-1]
            if len(input_shape) > 2
            else 1
        )
        nb_features = (
            input_shape[-2]
            if len(input_shape) > 2
            else input_shape[-1]
        )
        nb_weights = nb_channel * nb_features
        weights.append(log_reg_weights[consumed : consumed + nb_weights].reshape(nb_features, nb_channel))
        shapes.append((nb_features, nb_channel))
        consumed += nb_weights

    # if "input_bool" in layers_names:
    #     layer = model.get_layer("input_bool")
    #     outputs.append(layer.output)
    #     shapes.append(layer.output_shape)

    #     consumed += layer.output_shape[-1]
    #     weights.append(log_reg_weights[:consumed])
    # if "reshape_num_input" in layers_names:
    #     layer = model.get_layer("reshape_num_input")
    #     outputs.append(layer.output)
    #     shapes.append(layer.output_shape)

    #     nb_weights = layer.output_shape[-2] * layer.output_shape[-1]
    #     weights.append(log_reg_weights[consumed : consumed + nb_weights])
    #     consumed += nb_weights

    # if "reshape_cat_output" in layers_names:

    #     layer = model.get_layer("reshape_cat_output")
    #     outputs.append(layer.output)
    #     shapes.append(layer.output_shape)

    #     nb_weights = layer.output_shape[-2] * layer.output_shape[-1]
    #     weights.append(log_reg_weights[consumed : consumed + nb_weights])
    #     consumed += nb_weights

    explainable_model = Model(inputs=[model.input], outputs=[model.output, *outputs],)
    # features_explain = np.hstack(
    #     [
    #         (expl_boo * bool_weight).sum(axis=-1).reshape(-1, 1),
    #         (expl_num * num_weight).sum(-1),
    #         (expl_others * others_weight).sum(axis=-1),
    #     ]
    # )
    # features_explain.shape
    # (expl_boo * bool_weight).sum(axis=-1).reshape(-1, 1),
    # (expl_num * num_weight).sum(-1),
    # (expl_others * others_weight).sum(axis=-1),

    predictions = explainable_model.predict(input_model)
    probas = predictions[0]
    aggregated_explanation = []
    print()
    for weight_slice, shape_feat, raw_explanation in zip(weights, shapes, predictions[1:]):
        reshaped_expl = raw_explanation.reshape(-1, shape_feat[0], shape_feat[1])
        reshaped_weights = weight_slice.reshape(1, *weight_slice.shape)
        feature_explaination = (reshaped_expl * reshaped_weights).sum(axis=-1).reshape(-1, shape_feat[0])
        aggregated_explanation.append(feature_explaination)

    return probas, np.hstack(aggregated_explanation)
    # bool_weight = model.get_weights()[-2][:1]
    # bool_weight.shape
    # num_weight = model.get_weights()[-2][1 : 9 * 16 + 1].reshape(-1, 16)
    # num_weight.shape
    # others_weight = model.get_weights()[-2][9 * 16 + 1 :].reshape(-1, 32)
    # others_weight.shape
    # features_explain = np.hstack(
    # [
    #     (expl_boo * bool_weight).sum(axis=-1).reshape(-1, 1),
    #     (expl_num * num_weight).sum(-1),
    #     (expl_others * others_weight).sum(axis=-1),
    # ]
    # )

    # features_explain.shape
    # model.layers[-1].get_weights()[0]
    # new_model = Model(
    #     inputs=[model.inputs],
    #     outputs=[model.output, model.layers[-2].output, model.layers[-3].output],
    # )
    # return
