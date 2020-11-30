import numpy as np
from keras import backend as K
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Reshape, Lambda
from tensorflow.python.keras.optimizers import Adam
from model import inverse_net, forward_model, create_feature_vector, one_hot_encode_action

from baselines import logger


class IntrinsicRewardModel:
    """
    This approach uses intrinsic reward which minimizes predictability of the next
    state from the current one, maximizing surprise and exploring unknown
    trajectories.

    r_{t}{i} = theta/2 ||phi'(s_{t+1} - phi(s_{t+1}))L2

    intrinsic reward is the
    difference between predicted and real feature vector of the next state
    with a scaling factor theta / 2

    to predict this ICM has two sub-modules, the inverse and forward model

    (predicted state)
    forward model: predicts feature representation of the next state, given the
    feature vector of the current state and the action

    (predicted action)
    inverse model: tries to predict the next action by learning the current
    state and the next state


    ICM agent covers four elements
    1. the inverse loss: cross-entropy between predicted action and true action
    2. forward loss: difference between predicted state and next state
    3. policy gradient loss
    4. intrinsic reward

    with hyperparameters

    beta: inverse loss against the reward generated from the forward model
    lambda: importance of policy gradient loss against intrinsic reward

    """

    def __init__(self, env, params):
        self.env = env  # import env

        self.state_shape = env.observation_space.shape  # the state space
        self.action_shape = env.action_space.n  # the action space

        self.lmd = params.lmbd  # ratio of the external loss against the intrinsic reward
        self.beta = params.beta  # ratio of the inverse loss against the forward reward

        self.training_games = params.iterations.training_games  # N training games
        self.goal_steps = params.iterations.game_steps  # N training steps
        self.batch_size = params.batch_size  # batch size for training the model

        # Build and Compile Model

        self.model = self._build_icm_model()  # build ICM
        self.model.compile(optimizer=Adam(), loss="mse")  # Complies ICM

        # Instrumentation

    def _build_icm_model(self, state_shape=(2,), action_shape=(3,)):
        """
        Ths function:

        1. featurizes current and future state
        2. runs a forward pass with inverse model and forward model
        3. Uses the inverse model and the forward model to calculate intrinsic reward (surprise)
        4. Computes loss with intrinsic and forward reward, beta used to weigh relevance
        5. Computes loss with intrinsic reward and policy gradient, lambda is used to weigh relevance

        @return Model with updated loss
        """
        # Main ICM network

        # create state, next state, and action input
        s_t = Input(shape=state_shape, name="state_t")
        s_t1 = Input(shape=state_shape, name="state_t1")
        a_t = Input(shape=action_shape, name="action")

        reshape = Reshape(target_shape=(2,))

        # encode state and next state
        feature_vector_map = create_feature_vector((2,))
        fv_t = feature_vector_map(reshape(s_t))
        fv_t1 = feature_vector_map(reshape(s_t1))

        # predict action and feed-forward state
        a_t_hat = inverse_net(fv_t, fv_t1)
        s_t1_hat = forward_model(fv_t, a_t)

        # the intrinsic reward reflects difference between predicted and actual next state
        int_reward = Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1),
                            output_shape=(1,),
                            name="reward_intrinsic")([fv_t1, s_t1_hat])

        # inverse model loss
        inv_loss = Lambda(lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), axis=-1),
                          output_shape=(1,))([a_t, a_t_hat])

        # combined model loss

        # beta weighs the inverse loss against the rwd (generate from the forward model)
        loss = Lambda(lambda x: self.beta * x[0] + (1.0 - self.beta) * x[1], output_shape=(1,))([int_reward, inv_loss])

        # combined model loss

        # lmd is lambda, the param the weights the importance of the policy gradient loss against the intrinsic reward
        rwd = Input(shape=(1,))
        loss = Lambda(lambda x: (-self.lmd * x[0] + x[1]), output_shape=(1,))([rwd, loss])

        return Model([s_t, s_t1, a_t, rwd], loss)

    def act(self, current_state):
        """
        This method samples the action space, makes a move, and using the ICM minimizes the combined loss of the
        extrinsic and intrinsic reward.
        """
        losses = []
        # TODO: replace this hard-coded value
        for action_option in range(3):
            copy_env = np.copy.deepcopy(self.env)
            new_state, reward, _, _ = copy_env.step(action_option)
            action_option = one_hot_encode_action(self.action_shape, action_option)

            current_state = np.array(current_state).reshape(-1, len(current_state))
            new_state = np.array(new_state).reshape(-1, len(new_state))
            action_option = np.array(action_option).reshape(-1, len(action_option))
            reward = np.array(reward).reshape(-1, 1)

            loss = self.model.predict([current_state, new_state, action_option, reward])
            losses.append(loss)
        # TODO: doesn't this maximize the losses?
        chosen_action = np.argmax(losses)
        return chosen_action

    def learn(self, prev_states, states, actions, rewards):
        """
        Batch trains the network and logs loss.
        """
        # batch train the network
        s_t = prev_states
        s_t1 = states

        actions = np.array(actions)

        icm_loss = self.model.train_on_batch([s_t, s_t1, np.array(actions), np.array(rewards).reshape((-1, 1))],
                                             np.zeros((self.batch_size,)))
        logger.logkv('InternalCuriosityModelLoss', icm_loss)

    def get_intrinsic_reward(self, x):
        # x -> [prev_state, state, action]
        state_t = self.model.get_layer("state_t").input
        state_t1 = self.model.get_layer("state_t1").input
        action = self.model.get_layer("action").input

        reward_intrinsic = self.model.get_layer("reward_intrinsic").output

        return K.function([state_t, state_t1, action], [reward_intrinsic])(x)[0]
