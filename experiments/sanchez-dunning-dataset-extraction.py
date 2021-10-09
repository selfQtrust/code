# datasetoffigure4
# extracted with https://automeris.io/WebPlotDigitizer/
# from figure 1 to 4 in
# @article{sanchez2018overconfidence,
# 	title={Overconfidence among beginners: Is a little learning a dangerous thing?},
# 	author={Sanchez, Carmen and Dunning, David},
# 	journal={Journal of Personality and Social Psychology},
# 	volume={114},
# 	number={1},
# 	pages={10},
# 	year={2018},
# 	publisher={American Psychological Association}
# }

import errno
import os

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.use("PDF")

from qtrusttesting import Util


def __load_datas(filename):

    data_file = os.path.join(os.getcwd(), "papers", "dataset", filename,)

    # load datas if available
    if os.path.isfile(data_file):
        df = pd.read_pickle(data_file)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    return df


def __get_df1_from_csv(
    path, n,
):

    csv_file = os.path.join(path, "sanchez2018overconfidence-study1234.csv",)

    df = pd.read_csv(csv_file)
    df = df.dropna()

    df1 = pd.DataFrame(columns=["Study", "Trial", "UserName", "Overconfidence"])

    np.random.seed(42)
    for index, row in df.iterrows():
        study = row["study"]
        trial = row["trial"]
        x = row["mean"]
        delta = row["ci95max"] - row["ci95min"]
        accuracy = row["accuracy"]
        std = math.sqrt(n) * delta / 4
        y = np.random.normal(loc=x, scale=std, size=n)
        y_standardised = (y - y.mean()) / y.std()
        y_scaled = y_standardised * std + x
        for i in range(n):
            df1 = df1.append(
                {
                    "Study": study,
                    "Trial": trial,
                    "UserName": i,
                    "Overconfidence": y_scaled[i],
                    "Confidence": y_scaled[i] + accuracy,
                },
                ignore_index=True,
            )

    return df1


# set paths and filenames
working_directory = os.getcwd()
if working_directory.split(os.path.sep)[-1] == "experiments":
    working_directory = os.path.dirname(working_directory)

figure_output_filename = os.path.join(working_directory, "figures", "sd_oc.pdf")

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
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)


util = Util(None, None, figure_output_filename)
my_dpi = 72
figsize = ((528 / 4) / my_dpi, (864 / 6) / my_dpi)
ax = [util.set_plotting_conf(plt, sns, 1, 1, figsize, my_dpi)]

# plot extracting datas
for i in range(len(ax)):
    sns.lineplot(
        ax=ax[i],
        data=df1[df1["Study"] == i + 1],
        x="Trial",
        y="Confidence",
        legend=False,
    )

# axes sizes, label and title
for i in range(len(ax)):
    ax[i].set_title(r"Study " + str(i + 1))
    ax[i].set(ylim=(40, 80))
    ax[i].set(xlim=(0, 60))
    ax[i].set_xlabel(r"Trial Number")
    ax[i].xaxis.set_ticks(range(0, 65, 5))
    ax[i].yaxis.set_ticks(range(40, 90, 10))
    if i == 0:
        ax[i].set_ylabel(r"Self-Trust (\%)")
    else:
        ax[i].set_ylabel("")
        ax[i].tick_params(labelleft=False)


util.save_fig(plt, figure_output_filename)
