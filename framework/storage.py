import numpy as np
import pandas as pd

import os


class Store:
    def __init__(self, pipeline, filename):
        self.__pipeline = pipeline
        self.__filename = filename

        # Init dataframes
        self.__df_trust_history = pd.DataFrame(
            columns=[
                "episode",
                "trustor",
                "trustee",
                "mean_trust",
                "score",
                "global_trust_reward",
            ]
        )
        self.__df_mean_trust_history = pd.DataFrame(
            columns=[
                "episode",
                "trustor",
                "trustee",
                "mean_trust",
                "score",
                "global_trust_reward",
            ]
        )

    def run(self):
        stop = False
        while not stop:
            message = self.__pipeline.get()
            agent_name = message[0]
            callback = message[1]
            params = message[2]

            if callback == "store_trust_vs_score":
                self.__store_trust_vs_score(agent_name, params)
            if callback == "store_mean_trust":
                self.__store_mean_trust(agent_name, params)
            if callback == "save_datas":
                self.__save_datas(self.__filename)
                stop = True

    def __store_trust_vs_score(self, agent_name, params):
        for trustee_name, trustee_trust_matrix in params["trust_values"].items():
            mean_trust = np.mean(trustee_trust_matrix)
            self.__df_trust_history = self.__df_trust_history.append(
                pd.DataFrame(
                    {
                        "episode": params["episode"],
                        "trustor": agent_name,
                        "trustee": trustee_name,
                        "mean_trust": mean_trust,
                        "score": params["score"],
                        "global_trust_reward": params["global_trust_reward"][
                            trustee_name
                        ],
                    },
                    index=[0],
                ),
                ignore_index=False,
            )

    def __store_mean_trust(self, agent_name, params):
        for trustee_name, trust_values_episode in params[
            "trust_values_episode"
        ].items():
            if not trust_values_episode:
                mean_trust = 0
                max_trust = 0
                min_trust = 0
            else:
                mean_trust = np.mean(trust_values_episode)
                max_trust = np.max(trust_values_episode)
                min_trust = np.min(trust_values_episode)
            self.__df_mean_trust_history = self.__df_mean_trust_history.append(
                pd.DataFrame(
                    {
                        "episode": params["episode"],
                        "trustor": agent_name,
                        "trustee": trustee_name,
                        "mean_trust": mean_trust,
                        "max_trust": max_trust,
                        "min_trust": min_trust,
                        "score": params["score"],
                        "global_trust_reward": params["global_trust_reward"][
                            trustee_name
                        ],
                    },
                    index=[0],
                ),
                ignore_index=False,
            )

    def __save_datas(self, filename):
        data_file = f"{filename}_trust_history.pkl"

        # load datas if available and concatenate with new acquisition
        if os.path.isfile(data_file):
            df = pd.concat([pd.read_pickle(data_file), self.__df_trust_history])
        else:
            df = self.__df_trust_history

        # write datas if available
        df.to_pickle(data_file)

        data_file = f"{filename}_meantrust_history.pkl"

        # load datas if available and concatenate with new acquisition
        if os.path.isfile(data_file):
            df = pd.concat([pd.read_pickle(data_file), self.__df_mean_trust_history])
        else:
            df = self.__df_mean_trust_history

        # write datas if available
        df.to_pickle(data_file)
