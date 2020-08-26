import tensorflow as tf


def _build_encoding_sub_model(shape, batch_size, layer_sizes=(32, 32), name=None):
    assert len(layer_sizes) > 0, "You need at least one layer in an encoding submodel."

    inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(shape))

    x = inputs
    for i in range(len(layer_sizes)):
        x = tf.keras.layers.Dense(layer_sizes[i])(x)
        x = tf.keras.layers.Activation("relu")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)
