import numpy as np
import logging
import collections

from framework.trust_env import TrustEnv
from framework.qlearning import QTrust


class Agent:
    def __init__(
        self,
        name,
        learning_algo,
        pipeline,
        agent_step_lock,
        with_trust=False,
        visu=True,
        store_results=False,
    ):
        self.name = name
        self.learning_algo = learning_algo
        self.state = None

        # lock for multi-agent purpose
        self.__agent_step_lock = agent_step_lock

        self.pipeline = pipeline

        # trust params
        self.__with_trust = with_trust
        self.trust_begin = False

        # results storage
        self.__store_results = store_results
        self.__scores = []

        # visu (enable to send datas in the pipeline if you want to look at them in real time)
        self.__visu = visu
        if self.__visu:
            self.pipeline.put(
                (
                    self.name,
                    "draw_macro_params",
                    {
                        "n_episodes_uc": self.learning_algo.episodes,
                        "learning_rate_uc": self.learning_algo.learning_rate,
                        "discount_uc": self.learning_algo.discount,
                    },
                )
            )

    def init_trust(self, agents, env, episodes, learning_rate, discount):
        # init learning algo
        self.__env = env
        self.__trust_env = TrustEnv(env)
        self.learning_algo_trust = QTrust(
            env,
            self.__trust_env,
            self,
            agents,
            episodes=episodes,
            learning_rate=learning_rate,
            discount=discount,
        )

        if self.__visu:
            self.pipeline.put(
                (
                    self.name,
                    "draw_trust_learning_params",
                    {
                        "n_episodes_trust": episodes,
                        "learning_rate_trust": learning_rate,
                        "discount_trust": discount,
                    },
                )
            )

        # init global_trust_reward
        self.global_trust_reward = dict()
        for agent in agents:
            self.global_trust_reward[agent.name] = 0

        # init learning trust_values maps
        trust_values = dict()

        for agent in agents:
            trust_values_agent = np.zeros(env.size)
            for x in range(env.size[0]):
                for y in range(env.size[1]):
                    trust_values_agent[x, y] = self.trust(agent, (y, x))
            trust_values[agent.name] = trust_values_agent

        self.__trust_values = trust_values
        if self.__visu:
            self.pipeline.put(
                (self.name, "draw_trust_maps", {"trust_values": trust_values,},)
            )

        # init trust values dict of lists for recording trust values lists per trustee within an episode
        self.__trust_values_episode = dict()

        # init trustees deque
        self.trustees = collections.deque(self.learning_algo_trust.agents)
        self.trustees.remove(self)

        # init self.__compute_global_trust_reward boolean
        self.__compute_global_trust_reward = False

    def __set_reward_trust(self, trustee_name, history_n_episodes=15):
        """function to be called after an episode where agent always follows advice of trustee to reset global reward of trustee

        Args:
            trustee_name (int): name of trustee whose one want to reset global reward
            history_n_episodes(int): number of last espisodes to take into account to compare to the trustee's score
        """
        if len(self.__scores) <= 1:
            self.global_trust_reward[trustee_name] = 0
        else:
            self.global_trust_reward[trustee_name] = (
                -(self.__scores[-1] - max(self.__scores[-history_n_episodes:-1]))
                / self.__scores[-1]
            )
        if (len(self.__scores) > history_n_episodes) and (
            np.var(self.__scores[-history_n_episodes:-1]) == 0
        ):
            self.global_trust_reward[trustee_name] = 0

        logging.debug(
            f"agent {self.name} update global_trust_reward of trustee {trustee_name} to {self.global_trust_reward[trustee_name]}"
        )

    def __update_trust_values(self, trustee, state):
        new_value = self.trust(trustee, state)
        self.__trust_values[trustee.name][state] = new_value
        if self.__visu:
            self.pipeline.put(
                (
                    self.name,
                    "update_trust_maps",
                    {
                        "trustee_name": trustee.name,
                        "state": state,
                        "trust_level": new_value,
                    },
                )
            )
            qtrust_values = self.learning_algo_trust.q_table[trustee.name][state]
            self.pipeline.put(
                (
                    self.name,
                    "draw_trust_learning",
                    {
                        "trustee_name": trustee.name,
                        "state": state,
                        "qtrust_values": qtrust_values,
                    },
                )
            )

    def trust(self, *args):
        """Return trust of self in trustee at self.state or another state

        Args:
            trustee (Agent): the agent level of trust we want to know
            state (tuple) (optionnal): if mentionned, the state where we want to measure trust, else get current state of self

        Returns:
            float: trust level
        """
        if len(args) == 1:
            trustee = args[0]
            state = self.state
        elif len(args) == 2:
            trustee = args[0]
            state = args[1]

        # TODO: In case of multiple occurrences of the maximum values,
        # the indices corresponding to the first occurrence are returned.
        # Is it what we want ?
        index = int(np.argmax(self.learning_algo_trust.q_table[trustee.name][state]))
        return self.__trust_env.possible_trust_values[index]

    def run(self, env, trust_env, agents):
        agents_names = [agent.name for agent in agents]
        episode_number = 0

        for episode in range(self.learning_algo.episodes):

            # reinitialise steps number
            steps = 0

            # reinitialise trust values dict of lists for recording trust values lists per trustee within an episode
            for trustee_name in agents_names:
                self.__trust_values_episode[trustee_name] = []

            # explore trust : begining conditions
            self.__compute_global_trust_reward = False
            if self.__with_trust:
                # rotate trustee deque if not empty
                if self.trustees:
                    trustee = self.trustees.pop()
                    self.trustees.appendleft(trustee)

                # choose to compute global trust reward of a trustee or not
                # only if self.trustees is not empty
                # TODO experiment: a trustee is able to provide advices if its overall trust reward is sufficiently stable.
                n = np.random.random()
                if n <= 0.1 and self.trustees:
                    self.__compute_global_trust_reward = True

            # reinitialise state and done
            self.state = env.reset(self.name)
            done = False

            while not done:
                with self.__agent_step_lock:
                    # env.learn_on enables step by step control
                    if env.learn_on:
                        if self.__compute_global_trust_reward:
                            done = self.__make_step_following_trustee(
                                env, trust_env, self.trustees[0]
                            )
                        else:
                            done = self.__make_step(env, trust_env, agents)

                        # Update steps
                        steps += 1

                        # Communication to visu
                        if self.__visu:
                            self.pipeline.put(
                                (
                                    self.name,
                                    "draw_episode",
                                    {
                                        "steps": steps,
                                        "episode": episode_number,
                                        "done": done,
                                    },
                                )
                            )

                        # if control is in stepbystep mode, set env.learn_on to False
                        if env.stepbystep_on:
                            env.learn_on = False

                    else:
                        done = False

            # at the end of the episode do 9 steps ...
            # 1. draw trust map
            if self.__visu:
                self.pipeline.put(
                    (
                        self.name,
                        "draw_trust_maps",
                        {"trust_values": self.__trust_values,},
                    )
                )

            # 2. add new score in the memory of the agent
            self.__scores.append(-steps)

            # 3. compute global trust reward : ending conditions : reset global trust reward of self.trustees[0]
            if self.__compute_global_trust_reward:
                self.__set_reward_trust(self.trustees[0].name)

            # 4. compute global trust reward for self : ending conditions : reset global trust reward of self
            self.__set_reward_trust(self.name)

            # 5.1 draw trust vs score
            if self.__visu:
                self.pipeline.put(
                    (
                        self.name,
                        "draw_trust_vs_score",
                        {"trust_values": self.__trust_values, "score": -steps},
                    )
                )
            if self.__store_results:
                self.pipeline.put(
                    (
                        self.name,
                        "store_trust_vs_score",
                        {
                            "episode": episode,
                            "trust_values": self.__trust_values.copy(),
                            "score": -steps,
                            "global_trust_reward": self.global_trust_reward.copy(),
                        },
                    )
                )

            # 5.2 at the end of the episode, send a message to the pipeline to store self.__trust_values_episode dict()
            if self.__store_results:
                self.pipeline.put(
                    (
                        self.name,
                        "store_mean_trust",
                        {
                            "episode": episode,
                            "trust_values_episode": self.__trust_values_episode.copy(),
                            "score": -steps,
                            "global_trust_reward": self.global_trust_reward,
                        },
                    )
                )

            # 6. draw score vs episode
            if self.__visu:
                self.pipeline.put((self.name, "draw_score_vs_episode", {},))

            # 7. Log
            logging.debug(f"agent {self.name} episode {episode_number} steps {steps}")

            # 8. Decay epsilon (call self.learning_algo.afterlearn)
            self.learning_algo.afterlearn(self, episode_number)
            if self.__with_trust:
                # TODO: think about it!
                # Should we only decrease the epsilon of the trust for agents who advices are taken in account?
                self.learning_algo_trust.afterlearn(episode_number)

            # 9. iterate episode number
            episode_number += 1

        # at the end, send a message to the pipeline to save results
        if self.__store_results:
            self.pipeline.put((self.name, "save_datas", {},))

    def __make_step(self, env, trust_env, agents):

        # Trust Bloc
        if self.__with_trust:

            # Get the list of trusted advices
            option = "every time with every agent"
            action, advice = self.learning_algo_trust.choose_action()

            # Get trustee and current_trust_level from advice
            if advice[0]["choose_action"]["trust"]:
                current_trust_level = advice[0]["trust"]["greedy_level"]
                trustee_name = advice[0]["choose_action"]["trustee_name"]
            else:
                current_trust_level = trust_env.max_trust_level
                trustee_name = advice[0]["advisor"]

            for adv in advice:
                if adv["choose_action"]["trust"]:
                    if (
                        adv["choose_action"]["action"] == action
                        and adv["trust"]["greedy_level"] > current_trust_level
                    ):
                        trustee_name = adv["choose_action"]["trustee_name"]
                        current_trust_level = adv["trust"]["greedy_level"]

            trustee = [x for x in agents if x.name == trustee_name][0]

            # Make step in trust_env with chosen action
            state_, reward_env, reward_trust, done, _ = trust_env.step(
                self, trustee, current_trust_level, action
            )

            # Learn trust
            for adv in advice:
                if adv["trust"]:
                    trustee = [x for x in agents if x.name == adv["trust"]["trustee"]][
                        0
                    ]
                    trust_action_value = int(adv["trust"]["greedy_level"])

                    # TODO: run an experiment on this design of the trust_action_value which can be improved

                    trust_action = self.__trust_env.possible_trust_values.index(
                        trust_action_value
                    )
                    adv["learn"], action_trust_v = self.learning_algo_trust.learn(
                        trustee, trust_action, self.state, state_, reward_trust, done,
                    )
                    self.__update_trust_values(trustee, self.state)
                # TODO: else

            if self.__visu:
                self.pipeline.put(
                    (
                        self.name,
                        "draw_trust_compromise",
                        {"action_trust_v": action_trust_v, "advice": advice,},
                    )
                )

            # at the end of the step, store in self.__trust_values_episode the trust value computed by the model
            self.__trust_values_episode[trustee.name].append(
                self.trust(trustee, self.state)
            )

        else:
            # Set reward_trust to None
            reward_trust = None

            # Choose an action using epsilon-greedy
            action, infos = self.learning_algo.choose_action(self.state, True)
            if self.__visu:
                self.pipeline.put(
                    (
                        self.name,
                        "draw_trust_compromise",
                        {
                            "action_trust_v": None,
                            "advice": [
                                {
                                    "action_probs": None,
                                    "advisor": None,
                                    "choose_action": infos,
                                    "trust": None,
                                    "learn": None,
                                }
                            ],
                        },
                    )
                )

            # Make step with chosen action
            state_, reward_env, done, _ = env.step(self.name, action)

        # Learn
        self.learning_algo.learn(action, self.state, state_, reward_env, done)

        # Update state
        self.state = state_

        return done

    def __make_step_following_trustee(self, env, trust_env, trustee):

        # Choose action following advice of trustee
        # TODO: this advice is greedy. Should it be not greedy (i.e. only ask model of the agent) ?
        action, advice = self.learning_algo_trust.choose_action_following_trustee(
            trustee
        )

        # Make step in trust_env with chosen action
        state_, reward_env, reward_trust, done, _ = trust_env.step(
            self, trustee, int(advice[0]["trust"]["greedy_level"]), action
        )

        # Learn trust
        trust_action_value = int(advice[0]["trust"]["greedy_level"])
        trust_action = self.__trust_env.possible_trust_values.index(trust_action_value)
        advice[0]["learn"], action_trust_v = self.learning_algo_trust.learn(
            trustee, trust_action, self.state, state_, reward_trust, done,
        )
        self.__update_trust_values(trustee, self.state)

        # Display
        if self.__visu:
            self.pipeline.put(
                (
                    self.name,
                    "draw_trust_compromise",
                    {"action_trust_v": action_trust_v, "advice": advice,},
                )
            )

        # Learn
        self.learning_algo.learn(action, self.state, state_, reward_env, done)

        # Update state
        self.state = state_

        # at the end of the step, store in self.__trust_values_episode the trust value computed by the model
        self.__trust_values_episode[trustee.name].append(
            self.trust(trustee, self.state)
        )

        return done
