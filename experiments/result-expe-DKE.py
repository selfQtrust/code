# Showing results of DKE simulation with SelfQTrust

# SET PARAMS
params = {
    "results_storage_dir": "datasets/expe_DKE",
    "tex_images_path": "figures",
    "generate_datas": True,
    "env_names": [r"Maze $(6x6)$", r"Maze $(10x10)$", r"Maze $(30x30)$"],
    "metric": ["10mean_trust1", "10mean_trust2", "10max_trust"],
}

import logging
import os
from qtrusttesting import Util

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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

name = params["results_storage_dir"].split(os.path.sep)[-1]
figure_output_filename = os.path.join(
    working_directory, params["tex_images_path"], f"{name}.pdf"
)
data_file = os.path.join(working_directory, params["tex_images_path"], f"{name}.pkl",)

# retrieve experiments_filenames
all_experiments_filenames = [
    f
    for f in os.listdir(params["results_storage_dir"])
    if os.path.isfile(os.path.join(working_directory, params["results_storage_dir"], f))
]
experiments_filenames = [
    f for f in all_experiments_filenames if f.split("_")[9] == "meantrust"
]

# instanciate Util class for post-treatment of experiments_data_files
util = Util(
    os.path.join(working_directory, params["results_storage_dir"]),
    experiments_filenames,
    figure_output_filename,
)

# generate qtrust dataframes or load it if already generated
if params["generate_datas"]:
    df, hyperparams = util.generate_dataframe()
    util.save_datas(data_file, df, hyperparams)
else:
    df, hyperparams = util.load_datas(data_file)
# add a column with the name of the environment
"""
env_name_dict = dict(zip([1, 2, 3], params["env_names"],))
df["env_name"] = df["env_ID"].map(env_name_dict)
"""


# plot and save figure

learning_rate_uc_set = set(hyperparams["learning_rate_uc"])
discount_uc_set = set(hyperparams["discount_uc"])
learning_rate_trust_set = set(hyperparams["learning_rate_trust"])
discount_trust_set = set(hyperparams["discount_trust"])

logging.info(f"***********************************")
logging.info(f"*****  hyperparams  ***************")
logging.info(f"alpha values: {learning_rate_uc_set}")
logging.info(f"gamma values: {discount_uc_set}")
logging.info(f"alpha_trust values: {learning_rate_trust_set}")
logging.info(f"gamma_trust values: {discount_trust_set}")

my_dpi = 72
figsize = (528 / my_dpi, (864 / 6) / my_dpi)
ax = util.set_plotting_conf(plt, sns, 1, len(params["metric"]), figsize, my_dpi)

for i in range(len(params["metric"])):
    palette = sns.color_palette("rocket", n_colors=len(pd.unique(df["env_ID"])))
    sns.lineplot(
        ax=ax[i],
        data=df,
        x="episode_plus1",
        y=params["metric"][i],
        hue="env_ID",
        style="env_ID",
        palette=palette,
        linewidth=1,
    )

for i in range(len(params["metric"])):
    ax[i].set_title(r"$M" + str(i + 1) + "$", loc="center")
    ax[i].set_xlabel(r"Episode (log scale)")
    ax[i].set(xscale="log")
    ax[i].set(xlim=(1, 250))

    ax[i].set(ylim=(0, 100))
    ax[i].yaxis.set_ticks(range(0, 120, 20))

    if i == 0:
        ax[i].set_ylabel(r"Self-Trust (\%)")
    else:
        ax[i].set_ylabel("")

    if i == 2:
        ax[i].legend(labels=params["env_names"], loc=4, framealpha=0.6, fontsize=6)
    else:
        ax[i].legend(labels=params["env_names"], loc=1, framealpha=0.6, fontsize=6)


# Tight Layout and Save figure
util.save_fig(plt, figure_output_filename)
