import numpy as np
import matplotlib as plt
import gym
import IntrinsicRewardModel

from environment import Environment
from model import HyperParameters, one_hot_encode_action

BENCHMARK_ENVIRONMENT = 'CartPole-v0'


def batch_train():
    # track results
    states = []
    actions = []
    ext_rewards = []

    game_best_position = -.4
    current_position = np.ndarray([0, 2])

    params = HyperParameters()

    env = Environment(gym.make(BENCHMARK_ENVIRONMENT))

    model = IntrinsicRewardModel(env, params)

    batch_size = params.batch_size

    positions = np.zeros((params.iterations.training_games, 2))  # record actions
    rewards = np.zeros(params.iterations.training_games)  # record rewards

    for game_index in range(params.iterations.training_games):
        state = env.reset()  # each game is a new env
        game_reward = 0
        running_reward = 0
        for step_index in range(params.iterations.game_steps):

            # act
            if step_index > batch_size:
                action = model.act(state)
            else:
                action = model.env.action_space.sample()

            step = env.step(action)

            action = one_hot_encode_action(action, env.action_shape())

            # reward
            int_r_state = np.reshape(step.state, (1, 2))
            int_r_next_state = np.reshape(step.next_state, (1, 2))
            int_r_action = np.reshape(action, (1, 3))

            intrinsic_reward = model.get_intrinsic_reward([np.array(int_r_state), np.array(int_r_next_state),
                                                           np.array(int_r_action)])

            # calc total reward
            reward = intrinsic_reward + step.ext_reward
            game_reward += reward

            if step.state[0] > game_best_position:
                game_best_position = step.state[0]
                positions = np.append(current_position, [[game_index, game_best_position]], axis=0)
                running_reward += 10
            else:
                running_reward += reward

            # move state
            state = step.next_state

            # batch train

            states.append(step.next_state)
            ext_rewards.append(step.ext_reward)
            actions.append(step.action)

            if step_index % batch_size == 0 and step_index >= batch_size:
                all_states = states[-(batch_size + 1):]

            model.learn(prev_states=all_states[:batch_size],
                        states=all_states[batch_size:],
                        actions=actions[batch_size:],
                        rewards=ext_rewards[batch_size:])

            if step.done:
                break

            rewards[game_index] = game_reward
            positions[-1][0] = game_index
            positions[game_index] = positions[-1]

    _show_training_data(positions, rewards)


@staticmethod
def _show_training_data(positions, rewards):
    """
    plot the training data
    """
    positions[0] = [0, -0.4]
    plt.figure(1, figsize=[10, 5])
    plt.subplot(211)
    plt.plot(positions[:, 0], positions[:, 1])
    plt.xlabel('Episode')
    plt.ylabel('Furthest Position')
    plt.subplot(212)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
