import sys

# If you run in a terminal, following line enables to find the framework package
sys.path.append("..")

import logging
import os
import errno
import numpy as np
import pandas as pd
import json

from threading import Thread, RLock, Condition
import queue

import datetime

from framework.env import MazeEnv
from framework.trust_env import TrustEnv
from framework.qlearning import QLearning
from framework.agent import Agent
from threading import Thread
from framework.storage import Store


class QTrustTesting:
    def __init__(
        self,
        maze_file,
        storage_filename,
        number_of_try,
        episodes_uc,
        learning_rate_uc,
        discount_uc,
        episodes_trust,
        learning_rate_trust,
        discount_trust,
        maze_size=None,
        from_scratch=False,
        enable_render=False,
        with_trust=True,
    ):
        self.maze_file = maze_file
        self.storage_filename = storage_filename
        self.number_of_try = number_of_try
        self.episodes_uc = episodes_uc
        self.learning_rate_uc = learning_rate_uc
        self.discount_uc = discount_uc
        self.episodes_trust = episodes_trust
        self.learning_rate_trust = learning_rate_trust
        self.discount_trust = discount_trust
        self.maze_size = maze_size
        self.from_scratch = from_scratch
        self.enable_render = enable_render
        self.with_trust = with_trust

    def play(self):
        format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

        # Create pipeline
        pipeline = queue.Queue(maxsize=10)  # aprés avoir créé l'env

        # Create locks
        agent_step_lock = RLock()

        # loop ...
        time_start = datetime.datetime.now()
        logging.info(f"Experiment starting time : {time_start}")
        time_elapsed_loop = []

        for i in range(self.number_of_try):
            time_start_loop = datetime.datetime.now()
            agent_loc_name = "Alice" + str(i)

            agent_dict = {
                agent_loc_name: (136, 170, 255, 255),
            }
            agent_params = {
                agent_loc_name: {
                    "episodes_uc": self.episodes_uc,
                    "learning_rate_uc": self.learning_rate_uc,
                    "discount_uc": self.discount_uc,
                    "episodes_trust": self.episodes_trust,
                    "learning_rate_trust": self.learning_rate_trust,
                    "discount_trust": self.discount_trust,
                },
            }

            # Create env
            if not os.path.exists(self.maze_file):
                self.from_scratch = True
            if self.from_scratch:
                env = MazeEnv(
                    agent_dict=agent_dict,
                    pipeline=pipeline,
                    maze_file=self.maze_file,
                    maze_size=self.maze_size,
                    save=True,
                )
            else:
                env = MazeEnv(
                    agent_dict=agent_dict, pipeline=pipeline, maze_file=self.maze_file,
                )

            # Create trust env
            trust_env = TrustEnv(env)

            threads = list()

            # Create store
            store = Store(pipeline=pipeline, filename=self.storage_filename)
            x = Thread(target=store.run, name="store")
            threads.append(x)

            # Create agents and learning algos
            agents = []
            for agent_name in agent_dict.keys():
                agents.append(
                    Agent(
                        agent_name,
                        QLearning(
                            env,
                            episodes=agent_params[agent_name]["episodes_uc"],
                            learning_rate=agent_params[agent_name]["learning_rate_uc"],
                            discount=agent_params[agent_name]["discount_uc"],
                        ),
                        pipeline,
                        agent_step_lock,
                        with_trust=self.with_trust,
                        visu=False,
                        store_results=True,
                    )
                )

            # Init trust computation
            if self.with_trust:
                for agent in agents:
                    agent.init_trust(
                        agents,
                        env,
                        episodes=agent_params[agent_name]["episodes_trust"],
                        learning_rate=agent_params[agent_name]["learning_rate_trust"],
                        discount=agent_params[agent_name]["discount_trust"],
                    )

            # Learn
            for agent in agents:
                x = Thread(
                    target=agent.run, name=agent.name, args=(env, trust_env, agents)
                )
                threads.append(x)

            # start
            for thread in threads:
                thread.start()
                logging.info(f"Main    : thread {thread.name} started")

            # join
            for thread in threads:
                logging.info(f"Main    : before joining thread {thread.name}")
                thread.join()
                logging.info(f"Main    : thread {thread.name} done")

            time_elapsed_loop.append(datetime.datetime.now() - time_start_loop)
            eval_time_to_finish = np.mean(time_elapsed_loop) * (
                self.number_of_try - (i + 1)
            )
            logging.info(
                f"Evaluated time remaining to complete the test : {eval_time_to_finish}"
            )

        time_elapsed = datetime.datetime.now() - time_start
        logging.info(f"computation time : {time_elapsed}")


class Util:
    def __init__(self, expe_path, experiments_filenames, figure_output_filename):
        self.expe_path = expe_path
        self.experiments_filenames = experiments_filenames
        self.figure_output_filename = figure_output_filename

    def save_datas(self, data_file, df, hyperparams):
        hyperparams_file = data_file.replace("pkl", "json")

        if os.path.isfile(data_file):
            input("Do you really want to overwrite the old data file? (Ctrl+C if no)")

        df.to_pickle(data_file)

        if os.path.isfile(hyperparams_file):
            input("Do you really want to overwrite the old data file? (Ctrl+C if no)")
        with open(hyperparams_file, "w") as f:
            json.dump(hyperparams, f)

    def load_datas(self, data_file):
        hyperparams_file = data_file.replace("pkl", "json")

        if os.path.isfile(data_file):
            df = pd.read_pickle(data_file)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_file)

        if os.path.isfile(hyperparams_file):
            with open(hyperparams_file) as f:
                hyperparams = json.load(f)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), hyperparams_file
            )

        return df, hyperparams

    def generate_dataframe(self, metric="10mean_trust2"):
        # metric is used for overconfidence computation for comparison with Sanchez and Dunning paper
        # allowed metric are:
        # "10mean_trust1": The mean trust over all the states of environnement at the end of the episode.
        # "10mean_trust2": The mean trust over all states the agent passed through during the episode.
        # "10max_trust": The max trust the agent had during the episode in the states it passed through

        hyperparams = {
            "expe_ID": [],
            "env_ID": [],
            "number_of_try": [],
            "episodes_uc": [],
            "learning_rate_uc": [],
            "discount_uc": [],
            "episodes_trust": [],
            "learning_rate_trust": [],
            "discount_trust": [],
        }

        df_out = pd.DataFrame(
            columns=[
                "expe_ID",
                "env_ID",
                "number_of_try",
                "episodes_uc",
                "learning_rate_uc",
                "discount_uc",
                "episodes_trust",
                "learning_rate_trust",
                "discount_trust",
                "score",
                "episode_plus1",
                "10max_trust",
                "accuracy",
                "overconfidence",
            ]
        )

        expe_ID = -1
        for data_file_name in self.experiments_filenames:
            # forge a unique ID for current test whose dataset is in data_file_name
            expe_ID += 1

            logging.info(f"***********************************")
            logging.info(f"*****expe_ID: {expe_ID}************")
            logging.info(f"extract datas from {data_file_name}")

            # extract hyperparameters from filename
            y = data_file_name.split("_")
            if y[0] != "try" and y[0] != ".pkl":
                # TODO: not filenotfound, but custom error indicating filename mismatch
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), data_file_name
                )

            # load datas
            data_file1 = os.path.join(self.expe_path, data_file_name)
            df = pd.read_pickle(data_file1)
            data_file2 = data_file1.replace("meantrust", "trust")
            df2 = pd.read_pickle(data_file2)

            # number of episodes
            N_episode = df[df["trustor"] == "Alice1"].shape[0]

            # add hyperparams values in df
            env_ID = float(y[1])
            number_of_try = float(y[2].replace("A", "."))
            episodes_uc = float(y[3].replace("A", "."))
            learning_rate_uc = float(y[4].replace("A", "."))
            discount_uc = float(y[5].replace("A", "."))
            episodes_trust = float(y[6].replace("A", "."))
            learning_rate_trust = float(y[7].replace("A", "."))
            discount_trust = float(y[8].replace("A", "."))

            hyperparams["expe_ID"].append(expe_ID)
            hyperparams["env_ID"].append(env_ID)
            hyperparams["number_of_try"].append(number_of_try)
            hyperparams["episodes_uc"].append(episodes_uc)
            hyperparams["learning_rate_uc"].append(learning_rate_uc)
            hyperparams["discount_uc"].append(discount_uc)
            hyperparams["episodes_trust"].append(episodes_trust)
            hyperparams["learning_rate_trust"].append(learning_rate_trust)
            hyperparams["discount_trust"].append(discount_trust)

            df["expe_ID"] = np.repeat(expe_ID, len(df.index),)
            df["env_ID"] = np.repeat(env_ID, len(df.index),)
            df["number_of_try"] = np.repeat(number_of_try, len(df.index),)
            df["episodes_uc"] = np.repeat(episodes_uc, len(df.index),)
            df["learning_rate_uc"] = np.repeat(learning_rate_uc, len(df.index),)
            df["discount_uc"] = np.repeat(discount_uc, len(df.index),)
            df["episodes_trust"] = np.repeat(episodes_trust, len(df.index),)
            df["learning_rate_trust"] = np.repeat(learning_rate_trust, len(df.index),)
            df["discount_trust"] = np.repeat(discount_trust, len(df.index),)

            # convert score to numeric
            df["score"] = pd.to_numeric(df["score"])
            df["episode_plus1"] = pd.to_numeric(df["episode"]) + 1

            # compute metric (i.e self-trust in percent), accuracy and overconfidence
            offset = 50
            df["10mean_trust1"] = 10 * pd.to_numeric(df2["mean_trust"])
            df["10mean_trust2"] = 10 * pd.to_numeric(df["mean_trust"])
            df["10max_trust"] = 10 * pd.to_numeric(df["max_trust"])
            df["accuracy"] = 100 * (
                1
                - pd.to_numeric(df["score"])
                / np.repeat(df.groupby("trustor")["score"].min().values, N_episode,)
            )
            df["overconfidence"] = df[metric] - df["accuracy"] + offset

            # concat df with df_out
            df_out = pd.concat([df_out, df])

        return df_out, hyperparams

    def set_plotting_conf(self, plt, sns, nrows, ncols, figsize, dpi):
        # set plt configuration
        latex_style_times = {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Times",
            "font.size": 8,
        }
        plt.rcParams.update(latex_style_times)
        plt.xticks(fontsize=6)

        # set sns configuration
        sns.set_palette(sns.color_palette("rocket"))

        # get figure and ax
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi,)
        # TODO: set [0%,20%,40%,60%,80%,100%] y axis ticks label

        logging.info(f"set plotting conf done!")

        return ax

    def save_fig(self, plt, figure_output_filename):
        plt.tight_layout(pad=0.2)
        plt.savefig(figure_output_filename)
        logging.info(f"figure file {figure_output_filename} done!")

