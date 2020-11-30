import collections

class Environment:
    """
    Simple wrapper for an environment.
    """
    def __init__(self, env):
        self.env = env

    def action_space(self):
        return self.env.action_space

    def observation_space(self):
        return self.env.observation_space

    def action_shape(self):
        return self.action_space().n

    def state_shape(self):
        return self.observation_space().shape

    def random_action(self):
        return self.env.action_space().sample()

    def reset(self):
        self.env.reset()

    def step(self, action):
        Step = collections.namedtuple('Step', 'state next_state ext_reward done info')
        next_state, ext_reward, done, info = self.env.step(action)
        return Step(state='state', next_state=next_state, ext_reward=ext_reward, done=done, info=info)