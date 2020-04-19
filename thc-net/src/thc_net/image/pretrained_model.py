import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input,
    AveragePooling2D,
    ZeroPadding2D,
    ZeroPadding2D,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
)
from tensorflow.keras.models import Model


AUTOTUNE = tf.data.experimental.AUTOTUNE


def build_input(image_size, nb_channel, input_size):
    pooling = int(np.ceil(image_size / input_size))
    padding = np.floor(input_size - np.floor(image_size / pooling))
    padding_asym = int(padding % 2)
    padding = int(np.floor(padding / 2))

    inputs = Input(shape=(image_size, image_size, nb_channel))
    input_layer = inputs
    if image_size > input_size:
        input_layer = AveragePooling2D(pool_size=(pooling, pooling))(input_layer)
    if padding > 0 or padding_asym > 0:
        input_layer = ZeroPadding2D(
            padding=(
                (padding, padding + padding_asym),
                (padding, padding + padding_asym),
            )
        )(input_layer)
    return input_layer, inputs


def build_process_path(classnames, nb_channel):
    def process_path(file_path):
        label = tf.strings.split(file_path, os.path.sep)[-2]
        label = label == classnames
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = tf.io.decode_compressed(img, "GZIP")
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=nb_channel)
        return img, label

    return process_path


def build_dataset(file_list, process_path, *, repeat, batch_size, prefetch):
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    if prefetch is not None:
        dataset = dataset.prefetch(prefetch)

    return dataset


def build_model(
    model_class,
    image_size,
    nb_channel,
    input_size,
    output_dim,
    weights=None,
    patience=5,
):

    input_layer, inputs = build_input(image_size, nb_channel, input_size)
    base_model = model_class(
        input_tensor=input_layer,
        input_shape=(input_size, input_size, nb_channel),
        weights=weights,
        include_top=False,
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu", kernel_initializer="he_normal")(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = Dropout(0.3)(x)

    predictions = Dense(
        output_dim, activation="softmax" if output_dim > 2 else "sigmoid"
    )(x)
    model = Model(inputs=inputs, outputs=predictions)

    es = EarlyStopping(
        monitor="val_loss",
        verbose=1,
        mode="min",
        patience=patience,
        restore_best_weights=True,
    )
    # compile the model (should be done *after* setting layers to non-trainable)

    # We need to recompile the model for these modifications to take effect
    es.set_model(model)
    loss = "binary_crossentropy" if output_dim == 2 else "categorical_crossentropy"
    # optimizer = Lookahead(RectifiedAdam(), sync_period=6, slow_step_size=0.5)

    model.compile(optimizer="rmsprop", loss=loss)
    return model, [es]

