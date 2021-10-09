# Running an experiment with SelfQTrust
# usage : python selfq-trust-expe.py -i <inputfile> -expeid <expe_id>
# with inputfile a json file containing the description of the experiments
# (in the same format as expe_params later in the code)
# with expeid an int refering to a valid expe_id in inputfile (see "expe_id" key in expe_params)

import sys, getopt
import logging
import time
import os
import errno
import json
import pandas as pd
from qtrusttesting import QTrustTesting


def run_expe(**kwargs):
    expe_id = 0
    expe_params = []
    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "expe_params":
                if os.path.isfile(value):
                    with open(value, "r") as read_file:
                        expe_params = json.load(read_file)
                else:
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), value
                    )
            elif key == "expe_id":
                expe_id = value
            else:
                raise ValueError(f"arg {value} not allowed for run_expe() function")

    if len(kwargs) == 0 or expe_params == []:
        # DEFAULT PARAMS FOR TEST PURPOSE
        # be aware that COMPUTATION TIME CAN BE VERY LONG
        # and that STORAGE CAPACITY CAN BE HUGE
        # we advise to evaluate computation time with a subset of params values before launching a new experiment

        expe_params = [
            {
                "expe_id": 0,
                "expe_label": "EXPE 0: script testing 1",
                "params": {
                    "results_storage_dir": "datasets/script_testing_1",
                    "maze_set_description": "experiments/mazes10x10/maze_set_description.pkl",
                    "complexity": [18],
                    "number_of_try": [5],
                    "episodes_uc": [60],
                    "learning_rate_uc": [0.1],
                    "discount_uc": [0.99],
                    "episodes_trust": [60],
                    "learning_rate_trust": [0.1],
                    "discount_trust": [0.99],
                },
            },
            {
                "expe_id": 1,
                "expe_label": "EXPE 0: script testing 2",
                "params": {
                    "results_storage_dir": "datasets/script_testing_2",
                    "maze_set_description": "experiments/mazesDKETest/maze_set_description.pkl",
                    "complexity": [0, 0, 0],
                    "number_of_try": [5],
                    "episodes_uc": [60],
                    "learning_rate_uc": [0.1],
                    "discount_uc": [0.99],
                    "episodes_trust": [60],
                    "learning_rate_trust": [0.1],
                    "discount_trust": [0.99],
                },
            },
        ]

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info(f"************************************")

    # get expe description
    expe_description = [x for x in expe_params if x["expe_id"] == expe_id][0]
    expe_label = expe_description["expe_label"]
    params = expe_description["params"]

    # set working_directory and expe path
    working_directory = os.getcwd()
    if working_directory.split(os.path.sep)[-1] == "experiments":
        working_directory = os.path.dirname(working_directory)
    expe_path = os.path.join(working_directory, params["results_storage_dir"])
    maze_set_description = os.path.join(
        working_directory, params["maze_set_description"]
    )

    logging.info(f"*******EXPE CONFIGURATION************************")
    logging.info(f"{expe_label}")
    logging.info(f"results_storage_dir: {expe_path}")
    logging.info(f"maze_set_description: {maze_set_description}")
    logging.info(f"complexity: " + str(params["complexity"]))
    logging.info(f"number_of_try: " + str(params["number_of_try"]))
    logging.info(f"episodes_uc: " + str(params["episodes_uc"]))
    logging.info(f"learning_rate_uc: " + str(params["learning_rate_uc"]))
    logging.info(f"discount_uc: " + str(params["discount_uc"]))
    logging.info(f"episodes_trust: " + str(params["episodes_trust"]))
    logging.info(f"learning_rate_trust: " + str(params["learning_rate_trust"]))
    logging.info(f"discount_trust: " + str(params["discount_trust"]))
    logging.info(f"************************************")

    # get df_maze_set_description from params["maze_set_description"]
    if os.path.isfile(os.path.join(working_directory, params["maze_set_description"])):
        df_maze_set_description = pd.read_pickle(
            os.path.join(working_directory, params["maze_set_description"])
        )
    else:
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            os.path.join(working_directory, params["maze_set_description"]),
        )

    if params["complexity"] is not None:
        # remove from df_maze_set_description all rows except those of complexity isin params["complexity"]
        df_maze_set_description = df_maze_set_description[
            df_maze_set_description["complexity"].isin(params["complexity"])
        ]

        # get a list of id of only the fist maze corresponding to each complexity value
        retained_id = []
        complexity = params["complexity"]
        for index, row in df_maze_set_description.iterrows():
            id = row["id"]
            complexity_value = row["complexity"]
            if complexity_value in complexity:
                retained_id.append(id)
                complexity.remove(complexity_value)

        # show remaining complexity values not found in df_maze_set_description
        logging.info(
            f"** remaining complexity values without an environement maze found ..."
        )
        logging.info(f"{complexity}")

        # remove from df_maze_set_description all rows except those of id isin retained_id
        df_maze_set_description = df_maze_set_description[
            df_maze_set_description["id"].isin(retained_id)
        ]

    # print df_maze_set_description and ask user to confirm starting the experiment
    logging.info(f"******** following environment will be tested ******************")
    logging.info(f"{df_maze_set_description}")
    input("Press any key to confirm starting experiments")

    # TODO: Run a few blank tests to assess the computing time and storage capacity required,
    # check that the PC has what it needs
    # and display the results before requesting confirmation to start the experiments

    logging.info(f"************************************")
    logging.info(f"*********TEST SETUP*****************")
    logging.info(f"expe_path = {expe_path}")
    input("Press any key to confirm starting experiments")

    # create dir for storing expe results
    if not os.path.isdir(expe_path):
        os.mkdir(expe_path)
    else:
        input("Do you really want to use existing folder? Press any key to confirm")

    # TODO: what an ugly code with 8 nested for loops, please improve it
    for index, row in df_maze_set_description.iterrows():

        # create empty QTrustTesting
        id = row["id"]
        maze_file = os.path.join(os.path.dirname(maze_set_description), row["filename"])
        complexity = row["complexity"]
        qtrusttesting = QTrustTesting(
            maze_file, None, None, None, None, None, None, None, None,
        )

        for number_of_try in params["number_of_try"]:
            for episodes_uc in params["episodes_uc"]:
                for learning_rate_uc in params["learning_rate_uc"]:
                    for discount_uc in params["discount_uc"]:
                        for episodes_trust in params["episodes_trust"]:
                            for learning_rate_trust in params["learning_rate_trust"]:
                                for discount_trust in params["discount_trust"]:
                                    # forge the filename for storing the result of the test
                                    storage_filename = "try_"
                                    storage_filename += str(id)
                                    storage_filename += "_"
                                    storage_filename += str(number_of_try).replace(
                                        ".", "A"
                                    )
                                    storage_filename += "_"
                                    storage_filename += str(episodes_uc).replace(
                                        ".", "A"
                                    )
                                    storage_filename += "_"
                                    storage_filename += str(learning_rate_uc).replace(
                                        ".", "A"
                                    )
                                    storage_filename += "_"
                                    storage_filename += str(discount_uc).replace(
                                        ".", "A"
                                    )
                                    storage_filename += "_"
                                    storage_filename += str(episodes_trust).replace(
                                        ".", "A"
                                    )
                                    storage_filename += "_"
                                    storage_filename += str(
                                        learning_rate_trust
                                    ).replace(".", "A")
                                    storage_filename += "_"
                                    storage_filename += str(discount_trust).replace(
                                        ".", "A"
                                    )

                                    # set values for test
                                    qtrusttesting.storage_filename = os.path.join(
                                        working_directory, expe_path, storage_filename
                                    )

                                    qtrusttesting.number_of_try = number_of_try
                                    qtrusttesting.episodes_uc = episodes_uc
                                    qtrusttesting.learning_rate_uc = learning_rate_uc
                                    qtrusttesting.discount_uc = discount_uc
                                    qtrusttesting.episodes_trust = episodes_trust
                                    qtrusttesting.learning_rate_trust = (
                                        learning_rate_trust
                                    )
                                    qtrusttesting.discount_trust = discount_trust

                                    # infos
                                    logging.info(
                                        f"************************************"
                                    )
                                    logging.info(
                                        f"*********** SETUP ******************"
                                    )
                                    logging.info(
                                        f"storage_filename #1 is {qtrusttesting.storage_filename}_meantrust_history.pkl"
                                    )
                                    logging.info(
                                        f"storage_filename #2 is {qtrusttesting.storage_filename}_trust_history.pkl"
                                    )

                                    logging.info(f"number_of_try = {number_of_try}")
                                    logging.info(f"episodes_uc = {episodes_uc}")
                                    logging.info(f"episodes_trust = {episodes_trust}")

                                    logging.info(f"alpha_uc = {learning_rate_uc}")
                                    logging.info(f"discount_uc = {discount_uc}")
                                    logging.info(f"alpha_trust = {learning_rate_trust}")
                                    logging.info(f"discount_trust = {discount_trust}")

                                    logging.info(
                                        f"************************************"
                                    )
                                    logging.info(
                                        f"*********** BEGIN ******************"
                                    )

                                    # run test
                                    qtrusttesting.play()

                                    # wait 5 seconds to be sure that df is correctly stored in qtrusttesting.storage_filename
                                    time.sleep(5)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:n:")
    except getopt.GetoptError:
        print("selfq-trust-expe.py -i <inputfile> -n <expe_id>")
        sys.exit(2)
    inputfile = ""
    n = 0
    for opt, arg in opts:
        if opt == "-h":
            print("selfq-trust-expe.py -i <inputfile> -n <expe_id>")
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-n"):
            n = int(arg)

    if inputfile == "":
        run_expe(expe_id=0)
        # run_expe(expe_id=1)
    else:
        run_expe(expe_params=inputfile, expe_id=n)


if __name__ == "__main__":
    main(sys.argv[1:])

