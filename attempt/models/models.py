import tensorflow as tf
import gym
from attempt.utilities.utils import env_extract_dims
from attempt.models.components import _build_encoding_sub_model


from typing import Tuple, Union


def build_ffn_models(env: gym.Env, shared: bool = False,
                     layer_sizes: Tuple = (64, 64)):
    """Build simple two-layer model."""

    # preparation
    state_dimensionality, n_actions = env_extract_dims(env)

    # input preprocessing
    inputs_value = tf.keras.Input(shape=(state_dimensionality + n_actions,))
    inputs_policy = tf.keras.Input(shape=(state_dimensionality,))
    # policy network
    latent = _build_encoding_sub_model(inputs_policy.shape[1:], None, layer_sizes=layer_sizes,
                                       name="policy_encoder")(inputs_policy)
    out_policy = tf.keras.layers.Dense(n_actions, kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(latent)

    policy = tf.keras.Model(inputs=inputs_policy, outputs=out_policy, name="policy")

    # value network
    if not shared:
        value_latent = _build_encoding_sub_model(inputs_value.shape[1:], None, layer_sizes=layer_sizes,
                                                 name="value_encoder")(inputs_value)
        value_out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(value_latent)
    else:
        value_out = tf.keras.layers.Dense(1, input_dim=layer_sizes[-1], kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(latent)

    value = tf.keras.Model(inputs=inputs_value, outputs=value_out, name="value")
    policy_target = tf.keras.Model(inputs=inputs_policy, outputs=out_policy, name="policy_target")
    value_target = tf.keras.Model(inputs=inputs_value, outputs=value_out, name="value_target")

    return policy, value, policy_target, value_target