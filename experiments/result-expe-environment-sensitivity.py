# Showing results of sensitivity of SelfQTrust to complexity of environment

# SET PARAMS
params = {
    "results_storage_dir": [
        "datasets/sensitivity_to_environment_2",
        "datasets/sensitivity_to_environment_3",
    ],
    "figure_name": "sensitivity_to_environment",
    "tex_images_path": "figures",
    "maze_path": "experiments/mazes10x10",
    "generate_datas": True,
    "metric": "10mean_trust2",
    "palette": "rocket",
    "palette_r": "rocket_r",
}

import logging
import os
import errno
from qtrusttesting import Util

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("PDF")

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
logging.info(f"************************************")
logging.info(f"******post-treatment ...************")

# set filename of figure output (.pdf) and data file description (.pkl)
working_directory = os.getcwd()
if working_directory.split(os.path.sep)[-1] == "experiments":
    working_directory = os.path.dirname(working_directory)

# set figure output filename
name = params["figure_name"]
figure_output_filename = os.path.join(
    working_directory, params["tex_images_path"], f"{name}.pdf"
)

# set environment (maze) datas paths
maze_set_description = os.path.join(
    working_directory, params["maze_path"], "maze_set_description.pkl"
)
if os.path.isfile(maze_set_description):
    df_maze_set_description = pd.read_pickle(maze_set_description)
else:
    raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), maze_set_description
    )

dataset = []
for results_storage_dir in params["results_storage_dir"]:
    data_file = os.path.join(
        working_directory,
        params["tex_images_path"],
        f"{results_storage_dir.split(os.path.sep)[-1]}.pkl",
    )

    # retrieve experiments_filenames
    all_experiments_filenames = [
        f
        for f in os.listdir(results_storage_dir)
        if os.path.isfile(os.path.join(working_directory, results_storage_dir, f))
    ]
    experiments_filenames = [
        f for f in all_experiments_filenames if f.split("_")[9] == "meantrust"
    ]

    # instanciate Util class for post-treatment of experiments_data_files
    util = Util(
        os.path.join(working_directory, results_storage_dir),
        experiments_filenames,
        figure_output_filename,
    )

    # generate qtrust dataframes or load it if already generated
    if params["generate_datas"]:
        df, hyperparams = util.generate_dataframe()
        util.save_datas(data_file, df, hyperparams)
    else:
        df, hyperparams = util.load_datas(data_file)

    # Add a complexity column to df
    # Mapping df["env_ID"] to tuple (id,complexity) extracted from df_maze_set_description
    complexity_column = dict(
        zip(
            df_maze_set_description["id"].values,
            df_maze_set_description["complexity"].values,
        )
    )
    df["complexity"] = df["env_ID"].map(complexity_column)

    # display infos on dataset
    env_ID_set = set(hyperparams["env_ID"])
    number_of_try_set = set(hyperparams["number_of_try"])
    episodes_uc_set = set(hyperparams["episodes_uc"])
    episodes_trust_set = set(hyperparams["episodes_trust"])
    learning_rate_uc_set = set(hyperparams["learning_rate_uc"])
    discount_uc_set = set(hyperparams["discount_uc"])
    learning_rate_trust_set = set(hyperparams["learning_rate_trust"])
    discount_trust_set = set(hyperparams["discount_trust"])

    logging.info(f"***********************************")
    logging.info(f"*****  hyperparams  ***************")
    logging.info(f"******** environment **************")
    logging.info(f"number_of_try values: {number_of_try_set}")
    logging.info(f"episodes_uc values: {episodes_uc_set}")
    logging.info(f"episodes_trust values: {episodes_trust_set}")
    logging.info(f"******** learning ******************")
    logging.info(f"alpha values: {learning_rate_uc_set}")
    logging.info(f"gamma values: {discount_uc_set}")
    logging.info(f"alpha_trust values: {learning_rate_trust_set}")
    logging.info(f"gamma_trust values: {discount_trust_set}")

    dataset.append(df)

logging.info(f"***********************************")
logging.info(f"*****  plotting... ****************")
logging.info(f"***********************************")

my_dpi = 72
figsize = (528 / my_dpi, (864 / 6) / my_dpi)
ax = util.set_plotting_conf(plt, sns, 1, 3, figsize, my_dpi)

for i in range(2):
    # Figure i
    if i == 0:
        title = r"(a) ($\hat{\gamma} = 0.9$)"
    else:
        title = r"(b) ($\hat{\gamma} = 0.99$)"
    hue = "complexity"
    legend_title = r"complexity"
    palette = sns.color_palette(params["palette"], n_colors=len(pd.unique(df[hue])))
    sns.lineplot(
        ax=ax[i],
        data=dataset[i],
        x="episode_plus1",
        y=params["metric"],
        hue=hue,
        style=hue,
        palette=palette,
        linewidth=1,
    )

    # Rendering
    ax[i].set_xlabel(r"Episode")
    ax[i].set(xlim=(0, 60))
    ax[i].xaxis.set_ticks(range(0, 70, 10))
    ax[i].set(ylim=(10, 80))
    if i == 0:
        ax[i].set_ylabel(r"Self-Trust (\%)")
    else:
        ax[i].set_ylabel("")
    ax[i].legend(
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        framealpha=0.6,
        fontsize=6,
    )
    ax[i].set_title(label=title)

# Figure 2
df2 = pd.DataFrame(columns=["complexity", "10mean_trust2", "discount_trust"])
df2["complexity"] = dataset[0]["complexity"]
df2["10mean_trust2"] = dataset[0]["10mean_trust2"]
df2["discount_trust"] = np.repeat(0.9, len(dataset[0].index),)

df3 = pd.DataFrame(columns=["complexity", "10mean_trust2", "discount_trust"])
df3["complexity"] = dataset[1]["complexity"]
df3["10mean_trust2"] = dataset[1]["10mean_trust2"]
df3["discount_trust"] = np.repeat(0.99, len(dataset[1].index),)

df2 = df2.append(df3, ignore_index=True)

title = r"(c)"
legend_title = r"$\hat{\gamma}$"

hue = "discount_trust"
palette = sns.color_palette(params["palette"], n_colors=len(pd.unique(df2[hue])))
sns.lineplot(
    ax=ax[2],
    data=df2,
    x="complexity",
    y="10mean_trust2",
    hue=hue,
    style=hue,
    palette=palette,
    linewidth=1,
)

# Rendering
ax[2].set_xlabel(r"Complexity")
ax[2].set(xlim=(18, 88))
ax[2].xaxis.set_ticks(range(18, 98, 10))
ax[2].set(ylim=(10, 80))
ax[2].set_ylabel("")
ax[2].legend(title=legend_title, loc="lower right", framealpha=0.6, fontsize=6)
ax[2].set_title(label=title)

# Tight Layout and Save figure
util.save_fig(plt, figure_output_filename)
