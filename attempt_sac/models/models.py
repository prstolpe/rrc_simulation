from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate


def actor(state_shape, action_shape, units=(400, 300, 100)):
    state = Input(shape=state_shape)
    x = Dense(units[0], name="L0", activation="relu")(state)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation="relu")(x)

    actions_mean = Dense(action_shape[0], name="Out_mean")(x)
    actions_std = Dense(action_shape[0], name="Out_std")(x)

    model = Model(inputs=state, outputs=[actions_mean, actions_std])

    return model


def critic(state_shape, action_shape, units=(400, 200, 100)):
    inputs = [Input(shape=state_shape), Input(shape=action_shape)]
    concat = Concatenate(axis=-1)(inputs)
    x = Dense(units[0], name="Hidden0", activation="relu")(concat)
    for index in range(1, len(units)):
        x = Dense(units[index], name="Hidden{}".format(index), activation="relu")(x)

    output = Dense(1, name="Out_QVal")(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def update_target_weights(model, target_model, tau=0.005):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)