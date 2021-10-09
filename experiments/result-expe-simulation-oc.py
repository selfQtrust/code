# Showing results of sensitivity of SelfQTrust to complexity of environment

# SET PARAMS
params = {
    "results_storage_dir": "datasets/simulate_sanchez_dunning_oc_60",
    "figure_name": "simulate_sanchez_dunning_oc_60",
    "tex_images_path": "figures",
    "generate_datas": True,
    "metric": "10mean_trust2",
    "palette": "rocket",
    "palette_r": "rocket_r",
}

import logging
import errno
import os
from qtrusttesting import Util

import pandas as pd
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

name = params["results_storage_dir"].split(os.path.sep)[-1]
data_file = os.path.join(working_directory, params["tex_images_path"], f"{name}.pkl",)

# retrieve experiments_filenames
all_experiments_filenames = [
    f
    for f in os.listdir(os.path.join(working_directory, params["results_storage_dir"]))
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

logging.info(f"***********************************")
logging.info(f"*****  plotting... ****************")
logging.info(f"***********************************")

# load datas from sanchez and dunning
# load datas from pkl file
data_file = os.path.join(
    working_directory,
    "datasets",
    "sanchez_dunning_overconfidence",
    "sanchez2018overconfidence-study1234.pkl",
)

# load datas if available
if os.path.isfile(data_file):
    df1 = pd.read_pickle(data_file)
else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_file)

# prendre 50 lignes au hasard de df1

my_dpi = 72
figsize = ((528 / 4) / my_dpi, (864 / 6) / my_dpi)
ax = util.set_plotting_conf(plt, sns, 1, 1, figsize, my_dpi)

title = r"Simulation of OC Effect"
legend_title = r"Experiments"
palette = sns.color_palette(params["palette"], n_colors=2)
sns.lineplot(
    ax=ax,
    data=df1[df1["Study"] == 4],
    x="Trial",
    y="Confidence",
    label=r"Sanchez \& Dunning Study 4",
    color=palette[0],
    linestyle="--",
    linewidth=1,
)
sns.lineplot(
    ax=ax,
    data=df,
    x="episode_plus1",
    y=params["metric"],
    label=r"SelfQ-Trust",
    color=palette[1],
    linewidth=1,
)

# Rendering
ax.set_xlabel(r"Episode")
ax.set(xlim=(0, 60))
ax.xaxis.set_ticks(range(0, 70, 10))
ax.set(ylim=(10, 80))
ax.set_ylabel(r"Self-Trust (\%)")
ax.legend(
    title=legend_title, loc="lower right", framealpha=0.6, fontsize=6,
)
ax.set_title(label=title)

# Tight Layout and Save figure
util.save_fig(plt, figure_output_filename)
