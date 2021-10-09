# Showing results of sensitivity of SelfQTrust to hyperparameters

# SET PARAMS
params = {
    "results_storage_dir": "datasets/sensitivity_to_hyperparams",
    "tex_images_path": "figures",
    "generate_datas": True,
    "alpha": 0.1,
    "gamma": 0.9,
    "alpha_trust": 0.1,
    "gamma_trust": 0.9,
    "metric": "10mean_trust2",
    "palette": "rocket",
    "palette_r": "rocket_r",
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

alpha = params["alpha"]
gamma = params["gamma"]
alpha_trust = params["alpha_trust"]
gamma_trust = params["gamma_trust"]

logging.info(f"***********************************")
logging.info(f"*****  setting hyperparams*********")
logging.info(f"alpha = {alpha}")
logging.info(f"gamma = {gamma}")
logging.info(f"alpha_trust = {alpha_trust}")
logging.info(f"gamma_trust = {gamma_trust}")


expe_title = []
expe_title.append(r"$\alpha$")
expe_title.append(r"$\gamma$")
expe_title.append(r"$\hat{\alpha}$")
expe_title.append(r"$\hat{\gamma}$")

my_dpi = 72
figsize = (528 / my_dpi, (1.1 * 864 / 6) / my_dpi)
ax = util.set_plotting_conf(plt, sns, 1, 4, figsize, my_dpi)

df1 = df[df["discount_uc"] == gamma]
df2 = df1[df1["learning_rate_trust"] == alpha_trust]
data = df2[df2["discount_trust"] == gamma_trust]
hue = "learning_rate_uc"
legend_title = r"$\alpha$"
palette = sns.color_palette(params["palette"], n_colors=len(pd.unique(data[hue])))
sns.lineplot(
    ax=ax[0],
    data=data,
    x="episode_plus1",
    y=params["metric"],
    hue=hue,
    style=hue,
    palette=palette,
    linewidth=1,
)

df1 = df[df["learning_rate_uc"] == alpha]
df2 = df1[df1["learning_rate_trust"] == alpha_trust]
data = df2[df2["discount_trust"] == gamma_trust]
hue = "discount_uc"
legend_title = r"$\gamma$"
palette_r = sns.color_palette(params["palette_r"], n_colors=len(pd.unique(data[hue])))
sns.lineplot(
    ax=ax[1],
    data=data,
    x="episode_plus1",
    y=params["metric"],
    hue=hue,
    style=hue,
    palette=palette_r,
    linewidth=1,
)

df1 = df[df["discount_uc"] == gamma]
df2 = df1[df1["learning_rate_uc"] == alpha]
data = df2[df2["discount_trust"] == gamma_trust]
hue = "learning_rate_trust"
legend_title = r"$\hat{\alpha}$"
palette = sns.color_palette(params["palette"], n_colors=len(pd.unique(data[hue])))
sns.lineplot(
    ax=ax[2],
    data=data,
    x="episode_plus1",
    y=params["metric"],
    hue=hue,
    style=hue,
    palette=palette,
    linewidth=1,
)

df1 = df[df["learning_rate_uc"] == alpha]
df2 = df1[df1["learning_rate_trust"] == alpha_trust]
data = df2[df2["discount_uc"] == gamma]
hue = "discount_trust"
legend_title = r"$\hat{\gamma}$"
palette_r = sns.color_palette(params["palette_r"], n_colors=len(pd.unique(data[hue])))
sns.lineplot(
    ax=ax[3],
    data=data,
    x="episode_plus1",
    y=params["metric"],
    hue=hue,
    style=hue,
    palette=palette_r,
    linewidth=1,
)

for i in range(4):
    ax[i].axes.set_xlabel(r"Episode")
    ax[i].set(xlim=(0, 60))
    ax[i].xaxis.set_ticks(range(0, 70, 10))
    if i == 0:
        ax[i].set(ylim=(-40, 75))
        ax[i].yaxis.set_ticks(range(-60, 70, 20))
    else:
        ax[i].set(ylim=(0, 70))
        ax[i].yaxis.set_ticks(range(0, 70, 20))
    if i == 0:
        ax[i].set_ylabel(r"Self-Trust (\%)")
    else:
        ax[i].set_ylabel("")
    ax[i].set_title(f"{expe_title[i]}", loc="center")
    ax[i].legend(loc=4, framealpha=0.6, fontsize=6)

# Tight Layout and Save figure
util.save_fig(plt, figure_output_filename)
