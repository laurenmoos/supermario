import collections
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam


@staticmethod
def one_hot_encode_action(action_shape, action):
    """
    Utility function for transforming categorical action action_space
    into one-hot encoded numerical values.
    """
    action_encoded = np.zeros(action_shape, np.float32)
    action_encoded[action] = 1
    return action_encoded

@staticmethod
def inverse_net(f_state_t, f_state_t1, output_dim=3):
    inverse_net = concatenate([f_state_t, f_state_t1])
    inverse_net = Dense(24, activation='relu')(inverse_net)
    inverse_net = Dense(output_dim, activation='sigmoid')(inverse_net)
    return inverse_net


@staticmethod
def forward_model(f_state_t, action_t, output_dim=2):
    forward_net = concatenate([f_state_t, action_t])
    forward_net = Dense(24, activation='relu')(forward_net)
    forward_net = Dense(output_dim, activation='linear')(forward_net)
    return forward_net


@staticmethod
def create_feature_vector(input_shape):
    """
    Encode the feature vector using the first three layers of the model
    """

    model = Sequential()
    model.add(Dense(24, input_shape=input_shape, activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(2, activation='linear', name='feature'))
    return model


class HyperParameters:
    DEFAULT_BETA = 0.01
    DEFAULT_LMBD = 0.99
    BATCH_S IZE = 10

    DEFAULT_GAMES = 1000
    DEFAULT_GAME_STEPS = 100

    def __init__(self, beta=DEFAULT_BETA, lmbd=DEFAULT_LMBD, batch_size=BATCH_SIZE, games=DEFAULT_GAMES,
                 steps=DEFAULT_GAME_STEPS):
        """
        Initialize the top level hyper-parameters for the model, defaulting if not-provided.
        """
        self.beta = beta
        self.lmbd = lmbd
        self.batch_size = batch_size

        Iterations = collections.namedtuple('Iterations', 'training_games game_steps')
        self.iterations = Iterations(training_games=games, game_steps=steps)

        self.optimizer = Adam()
        self.loss = 'mse'
