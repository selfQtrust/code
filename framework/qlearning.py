import numpy as np
from scipy import signal
import copy

# If you are a reviewer from ICLR2022,
# Please note that the QTrust class is an early-stage implementation
# of the multi-agent case of our trust learning model.
# SelfQ-Trust explained in the paper is running exactly this code,
# where agents param is a list containing only the trustor param.
# In this case, parts of the code are not useful,
# but we prefer to publish this code because the experiments were done with.


class QLearning:
    # TODO: change this class by one in the following RL frameworks
    # coax (for basic algos ; Q-learning, SARSA and its RL-oriented mindeset)
    # MushroomRL (for basic algos ; simplicity)
    # Dopamine (for the Distributional Reinforcement Learning)

    def __init__(
        self,
        env,
        episodes=500,
        learning_rate=0.1,
        discount=0.95,
        epsilon=1,
        start_episode_decaying=1,
    ):
        self.env = env

        self.episodes = episodes
        self.learning_rate = learning_rate
        self.discount = discount

        self.epsilon = epsilon
        self.start_episode_decaying = start_episode_decaying
        self.end_episode_decaying = self.episodes // 2
        self.epsilon_decay_value = self.epsilon / (
            self.episodes // 2 - self.start_episode_decaying
        )

        self.q_table = np.zeros(
            shape=([env.size[0], env.size[1]] + [env.action_space.n])
        )

    def afterlearn(self, agent, episode):
        # Decaying is being done every episode if episode number is within decaying range
        if self.end_episode_decaying >= episode >= self.start_episode_decaying:
            self.epsilon -= self.epsilon_decay_value

        # Start trust episode
        if self.epsilon <= 0.97:
            agent.trust_begin = True

    def choose_action(self, state, greedy=True):
        explore = None
        if greedy:
            n = np.random.random()
            explore = True
            if n <= self.epsilon:
                # Explore : Get random action
                action = np.random.randint(0, self.env.action_space.n)
            else:
                # Exploit : Get action from Q table
                explore = False
                action = int(np.argmax(self.q_table[state]))
        else:
            action = int(np.argmax(self.q_table[state]))
            n = None
            explore = None

        # TODO: fill-in trust infos?
        infos = {
            "epsilon": self.epsilon,
            "n": n,
            "action": action,
            "action_values": copy.deepcopy(self.q_table[state]),
            "explore": explore,
            "trust": None,
            "trustee_name": None,
        }

        return (action, infos)

    def learn(self, action, state, state_, reward, done):
        current_q = self.q_table[state + (action,)]
        if not done:
            # If simulation did not end yet after last step - update Q table
            max_future_q = np.max(self.q_table[state_])
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                reward + self.discount * max_future_q
            )
        else:
            # If goal position is achived - update Q value with reward directly
            self.q_table[state + (action,)] = reward
            max_future_q = 0
            new_q = reward
        self.q_table[state + (action,)] = new_q

        # log
        infos = {
            "current_q": current_q,
            "reward": reward,
            "max_future_q": max_future_q,
            "new_q": new_q,
        }

        return infos


class QTrust(QLearning):
    def __init__(
        self,
        env,
        trust_env,
        trustor,
        agents,
        episodes=300,
        learning_rate=0.1,
        discount=0.9,
        epsilon=1,
        start_episode_decaying=1,
    ):
        super().__init__(
            env, episodes, learning_rate, discount, epsilon, start_episode_decaying,
        )
        self.epsilon_decay_value = self.epsilon_decay_value
        self.trustor = trustor
        self.agents = agents
        self.trust_env = trust_env
        self.q_table = dict()
        # self.q_table[agent.name][state] is initialized to a gaussian
        # so that max Q-value corresponds to trust = 0
        # the agent is neither optimist nor pessimist
        # TODO: run an experiment changing this distribution a priori
        # See if it can modulate the default behaviour of the agent?
        for agent in agents:
            self.q_table[agent.name] = np.zeros(
                shape=([env.size[0], env.size[1]] + [trust_env.action_space.n])
            )
            for x in range(env.size[0]):
                for y in range(env.size[1]):
                    self.q_table[agent.name][x, y, :] = (
                        signal.windows.gaussian(
                            trust_env.action_space.n, std=trust_env.action_space.n / 4
                        )
                        - 1
                    )

    def choose_action(self, greedy=True):
        # here we play trust-greedy algorithm
        # design basic explanations :
        # 1. in the begining, agent learn by herself without use of other agent advices
        # 2. then, other agents began to be ready to give advices : agent learn with the help of advices of others (only one trustee per episode)
        # 3. at the end of the learning process, agent has learned her own good compromise on others : she exploit the trust model
        # TODO: check n and m (see if design is relevant)
        # TODO: recode how other agents communicate they are ready to give advices
        n = np.random.random()
        if n <= self.epsilon:
            # 1. Explore : agent learn by herself without use of other agent advices
            action, infos = self.trustor.learning_algo.choose_action(
                self.trustor.state, greedy
            )
            action = np.random.randint(0, self.env.action_space.n)
            advice = []
            action_probs = np.zeros(self.env.action_space.n)
            trust_action_value = self.trust_env.max_trust_level
            base_trust_action_value = int(self.trustor.trust(self.trustor))
            action_probs[action] = trust_action_value
            advice.append(
                {
                    "action_probs": action_probs,
                    "advisor": self.trustor.name,
                    "choose_action": infos,
                    "trust": {
                        "trustee": self.trustor.name,
                        "greedy_level": trust_action_value,
                        "base_level": base_trust_action_value,
                    },
                    "learn": None,
                }
            )

        else:
            # Exploit or Trust Exploring
            m = np.random.random()
            if m <= self.epsilon and self.trustor.trustees:
                # 2. Trust Exploring (if self.trustor.trustees is not empty)
                trustee = self.trustor.trustees[0]
                action, advice = self.__choose_action_explore(trustee, greedy)
            else:
                # 3. Trust model exploitation
                action, advice = self.__choose_action_exploit(greedy=greedy)

        return (action, advice)

    def __choose_action_exploit(
        self, weigth_trustor=0.5, option="every time with every agent", greedy=True
    ):
        # TODO: run an experiment to test other compromise designs.
        # The design of the compromise may even be different from one agent to another to add much diversity.

        advice = []

        # step 1 : compute trustees list
        trustees = self.__meeting(option)

        # weigth_trustor is the weight of the opinion of self.trustor in the decision
        # weigth_trustees is the weight of the opinion of one trustee in the decision
        # weigth_trustor + len(trustees)*weigth_trustees = 1
        if trustees:
            weigth_trustees = (1 - weigth_trustor) / len(trustees)
        else:
            weigth_trustor = 1

        # step 2 : take into account the opinion of the trustor
        action, infos = self.trustor.learning_algo.choose_action(
            self.trustor.state, greedy
        )
        action_probs = np.zeros(self.env.action_space.n)
        # TODO: greedy : run an experiment on this choice of design
        base_trust_action_value = int(self.trustor.trust(self.trustor))
        trust_action_value = self.__get_greedy_trust_action_value(
            base_trust_action_value
        )

        action_probs[action] = weigth_trustor * trust_action_value
        advice.append(
            {
                "action_probs": action_probs,
                "advisor": self.trustor.name,
                "choose_action": infos,
                "trust": {
                    "trustee": self.trustor.name,
                    "greedy_level": trust_action_value,
                    "base_level": base_trust_action_value,
                },
                "learn": None,
            }
        )

        # step 3 : take into account the opinion of the trustees
        # weighted by self.trustor.trust(trustee)*weigth_trustees
        # note that self.trustor.trust(trustee) can be >= 0 (trust case)
        # note that self.trustor.trust(trustee) can be < 0 (mistrust case)
        for agent in trustees:
            action, infos = agent.learning_algo.choose_action(
                self.trustor.state, greedy
            )
            infos["trust"] = True
            infos["trustee_name"] = agent.name
            action_probs_agent = np.zeros(self.env.action_space.n)
            # TODO: greedy : run an experiment on this choice of design
            base_trust_action_value = int(self.trustor.trust(agent))
            trust_action_value = self.__get_greedy_trust_action_value(
                base_trust_action_value
            )
            action_probs_agent[action] = weigth_trustees * trust_action_value
            if (action_probs_agent == np.zeros(self.env.action_space.n)).all():
                # in this case, the advisor does not give any advice
                # do not add advice to advice list !
                pass
            else:
                advice.append(
                    {
                        "action_probs": action_probs_agent,
                        "advisor": agent.name,
                        "choose_action": infos,
                        "trust": {
                            "trustee": agent.name,
                            "greedy_level": trust_action_value,
                            "base_level": base_trust_action_value,
                        },
                        "learn": None,
                    }
                )

        # step 4 : build action_probs vector compromise computation
        # simply add opinions together
        # mistrust opinions should decay trust opinions,
        # and an opinion is a positive weight on an action of the action space
        action_probs = np.zeros(self.env.action_space.n)
        for adv in advice:
            action_probs += adv["action_probs"]
            action_probs = np.array(
                [x if x > 0 else 0 for x in action_probs], dtype=float
            )

        # step 5 : get action from action_probs and infer trust_action
        if (action_probs == np.zeros(self.env.action_space.n)).all():
            # step 5.1 action_probs may be 0 for all
            # in this case, choose action of self.trustor herself (no trust)
            action = [x for x in advice if x["advisor"] == self.trustor.name][0][
                "choose_action"
            ]["action"]
            # and advice resume to the first one, i.e. trustor one (see step 1)
            advice = [advice[0]]

            # trust_action computation : in this case, computation of self-trust
            # if trustor followed its opinion, it is because it trusted itself
            # if trustor.trust(trustor) > 0 then trust_action corresponds to trustor.trust(trustor)
            # but if trustor.trust(trustor) < 0 ; trust_action corresponds to -trustor.trust(trustor)
            # TODO: run an experiment on this choice of design
            # TODO: greedy : run an experiment on this choice of design

        else:
            # Normalize action_probs
            action_probs = action_probs / action_probs.sum()
            # Get action from action_probs
            # TODO : test a greedy process for action selection
            action = np.random.choice(self.env.action_space.n, p=action_probs)

            # TODO: adv["action_probs"] should'nt be equal to [0,0,0,0].
            # We should understand why it is sometime the case?
            # remove this part after testing it is useless
            n = 0
            for adv in advice:
                if (adv["action_probs"] == np.zeros(self.env.action_space.n)).all():
                    del advice[n]
                n += 1

        return (action, advice)

    def __choose_action_explore(self, trustee, greedy=True):
        # TODO: experiment greedy param
        # TODO: factorise with choose_action_following_trustee()

        # init advice list
        advice = []

        # get the opinion of the trustee
        action, infos = trustee.learning_algo.choose_action(self.trustor.state, greedy)

        # forge advice with greedy trust level
        infos["trust"] = True
        infos["trustee_name"] = trustee.name
        base_trust_action_value = int(self.trustor.trust(trustee))
        trust_action_value = self.__get_greedy_trust_action_value(
            base_trust_action_value
        )
        action_probs = np.zeros(self.env.action_space.n)
        trust_action_value = self.trust_env.max_trust_level
        action_probs[action] = trust_action_value
        advice.append(
            {
                "action_probs": action_probs,
                "advisor": trustee.name,
                "choose_action": infos,
                "trust": {
                    "trustee": trustee.name,
                    "greedy_level": trust_action_value,
                    "base_level": base_trust_action_value,
                },
                "learn": None,
            }
        )

        return (action, advice)

    def choose_action_following_trustee(self, trustee, greedy=True):
        # TODO: experiment greedy param

        # init advice list
        advice = []

        # get the opinion of the trustee
        action, infos = trustee.learning_algo.choose_action(self.trustor.state, greedy)

        # forge advice with max level (self.trust_env.max_trust_level)
        infos["trust"] = True
        infos["trustee_name"] = trustee.name
        action_probs = np.zeros(self.env.action_space.n)
        trust_action_value = self.trust_env.max_trust_level
        base_trust_action_value = int(self.trustor.trust(self.trustor))
        action_probs[action] = trust_action_value
        advice.append(
            {
                "action_probs": action_probs,
                "advisor": trustee.name,
                "choose_action": infos,
                "trust": {
                    "trustee": trustee.name,
                    "greedy_level": trust_action_value,
                    "base_level": base_trust_action_value,
                },
                "learn": None,
            }
        )

        return (action, advice)

    def __meeting(self, option):
        # TODO: to be tested ...
        if option == "every time with every agent":
            met_agents = [
                x
                for x in self.agents
                if (x.name != self.trustor.name and x.trust_begin)
            ]
        elif option == "only with agents whose states are the same":
            met_agents = [
                x
                for x in self.agents
                if (
                    x.name != self.trustor.name
                    and x.trust_begin
                    and x.state == self.trustor.state
                )
            ]

        return met_agents

    def __get_greedy_trust_action_value(self, value):
        inter = (int(-10 * self.epsilon), int(10 * self.epsilon))

        if value + inter[0] < -10:
            low = -10
        else:
            low = value + inter[0]

        if value + inter[1] > 10:
            high = 10
        else:
            high = value + inter[1]

        rng = np.random.default_rng()
        if low == high:
            trust_action_value = low
        else:
            rints = rng.integers(low=low, high=high + 1, size=1,)
            trust_action_value = rints[0]

        return trust_action_value

    def learn(self, trustee, trust_action, state, state_, reward, done):
        current_q = self.q_table[trustee.name][state + (trust_action,)]
        if not done:
            max_future_q = np.max(self.q_table[trustee.name][state_])
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                reward + self.discount * max_future_q
            )
        else:
            max_future_q = 0
            new_q = reward

        # Add gaussian delta on vector self.q_table[trustee.name][state] centralized on index trust_action
        # TODO: better adding random.normal
        delta = new_q - current_q
        vector_delta_all = delta * signal.windows.gaussian(
            2 * self.trust_env.action_space.n - 1, std=self.trust_env.action_space.n / 2
        )
        id_begin = (self.trust_env.action_space.n - 1) - trust_action
        id_end = (2 * self.trust_env.action_space.n - 1) - trust_action
        vector_delta = vector_delta_all[id_begin:id_end]
        self.q_table[trustee.name][state] = (
            self.q_table[trustee.name][state] + vector_delta
        )

        infos = {
            "current_q": current_q,
            "reward": reward,
            "max_future_q": max_future_q,
            "new_q": new_q,
        }

        return infos, self.__set_action_trust_v(state)

    def __set_action_trust_v(self, state):
        action_trust_v = []
        for trustee in self.agents:
            action_trust_v.append(self.q_table[trustee.name][state])
        return action_trust_v

    def afterlearn(self, episode):
        # Decaying is being done every episode if episode number is within decaying range
        if self.end_episode_decaying >= episode >= self.start_episode_decaying:
            self.epsilon -= self.epsilon_decay_value
            if self.epsilon < 0:
                self.epsilon = 0
