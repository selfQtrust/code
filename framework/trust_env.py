import numpy as np


class TrustEnv:
    class ActionSpace(object):
        def __init__(self, trust_n_levels):
            self.n = trust_n_levels

        """
        n = self.n
        """

        n = 21

    def __init__(self, env, max_trust_level=10):
        self.env = env
        self.size = self.env.size
        self.max_trust_level = max_trust_level
        self.action_space = self.ActionSpace(
            trust_n_levels=self.max_trust_level * 2 + 1
        )
        # WARNING : self.possible_trust_values MUST be a list of int
        self.possible_trust_values = np.arange(
            -self.max_trust_level,
            self.max_trust_level + self.max_trust_level / 10,
            self.max_trust_level / 10,
            dtype=int,
        ).tolist()

    def step(self, trustor, trustee, trust_greedy_level, action):
        state, reward_env, done, _ = self.env.step(trustor.name, action)

        reward_trust = (
            trust_greedy_level / self.max_trust_level
        ) * trustor.global_trust_reward[trustee.name]

        info = {}

        return state, reward_env, reward_trust, done, info

