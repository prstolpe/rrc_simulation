import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate


def Critic_gen(state_size, action_size, hidden_layers):
    input_x = Input(shape=state_size)
    input_a = Input(shape=action_size)
    x = input_x
    for i, j in enumerate(hidden_layers[:-1]):
        if i == 1:
            x = concatenate([x, input_a], axis=-1)
        x = tf.keras.layers.LayerNormalization()(x)
        x = Dense(j, activation='relu')(x)
    x = Dense(hidden_layers[-1],  kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003))(x)

    return tf.keras.Model([input_x, input_a], x)


def Actor_gen(state_size, action_size, hidden_layers, action_mult=1):
    input_x = Input(shape=state_size)
    x = input_x
    for i in hidden_layers:
        x = tf.keras.layers.LayerNormalization()(x)
        x = Dense(i, activation='relu')(x)
    x = Dense(action_size, activation='tanh', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003))(x)
    x = tf.math.multiply(x, action_mult)
    return tf.keras.Model(input_x, x)