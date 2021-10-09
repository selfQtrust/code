## Who is this repository for?
This is the public repository of the implementation of the SelfQtrust algorithm in python and the scripts to reproduce the experiments, as well as the associated datasets.

Please clone it only if you intend to review the corresponding paper submitted to ICLR2022 (see https://openreview.net/forum?id=G1J5OYjoiWb) or if you are a member of the project.

## How to reproduce experiments?
1. Clone the repository in a local directory (say ~/projects).
2. Create a conda virtual env (say trustenv).
3. Install required python packages with pip (see requirements.txt).
4. Test it: if the output is error-free and looks like the following, the test should be ok.
```console
(base) user@bar:~/projects$ mkdir SelfQtrust-ICLR2022
(base) user@bar:~/projects$ cd SelfQtrust-ICLR2022/
(base) user@bar:~/projects$ git clone https://github.com/selfQtrust/code.git
(base) user@bar:~/projects$ cd code/
(base) user@bar:~/projects$ conda create -n trustenv python=3.8 anaconda
(base) user@bar:~/projects$ conda activate trustenv
(trustenv) user@bar:~/projects/SelfQtrust-ICLR2022/code$ pip install -r requirements.txt
(trustenv) user@bar:~/projects/SelfQtrust-ICLR2022/code$ cd experiments/
(trustenv) user@bar:~/projects/SelfQtrust-ICLR2022/code/experiments$ python selfq-trust-expe.py
14:22:38: ************************************
14:22:38: *******EXPE CONFIGURATION************************
14:22:38: EXPE 0: script testing 1
14:22:38: results_storage_dir: /home/user/projects/SelfQtrust-ICLR2022/code/datasets/script_testing_1
14:22:38: maze_set_description: /home/user/projects/SelfQtrust-ICLR2022/code/experiments/mazes10x10/maze_set_description.pkl
...
...
...
14:23:21: Main    : thread store started
14:23:21: Main    : thread Alice4 started
14:23:21: Main    : before joining thread store
14:23:27: Main    : thread store done
14:23:27: Main    : before joining thread Alice4
14:23:27: Main    : thread Alice4 done
14:23:27: Evaluated time remaining to complete the test : 0:00:00
14:23:27: computation time : 0:00:29.871398
```

If test is ok, you should have a new directory in ~/projects/SelfQtrust-ICLR2022/code/datasets named script_testing_1 and in that directory the two following files:
* try_17_5_60_0A1_0A99_60_0A1_0A99_meantrust_history.pkl
* try_17_5_60_0A1_0A99_60_0A1_0A99_trust_history.pkl

The script selfq-trust-expe.py is used to run the experiment #id_expe that is configured in a json file expe_config_json with the command:
```console
(trustenv) user@bar:~/projects/SelfQtrust-ICLR2022/code/experiments$ python selfq-trust-expe.py -i expe_config_json -n id_expe
```

8 experiments are configured in the expe_params.json file. Experiments with #id_expe in (2,3,4,5,6,8) are those reported in the paper (#id_expe=2 corresponds to figure 1; #id_expe=3 corresponds to figure 2; #id_expe=4,5,6 corresponds to figure 3; #id_expe=8 corresponds to figure 4).

For example, run the last experiment that simulates the results of Study 4 of Sanchez and Dunning's paper on the overconfidence effect. Type the command:
```console
(trustenv) user@bar:~/projects/SelfQtrust-ICLR2022/code/experiments$ python selfq-trust-expe.py -i expe_params.json -n 8
```
Computed dataset should be found in ~/projects/SelfQtrust-ICLR2022/code/datasets/simulate_sanchez_dunning_oc folder.

A least, you can reproduce figures of the paper using the four scripts named result-expe-*.py:
* result-expe-DKE.py computes figure 1 from a dataset generated with experiment #id_expe=2 configured in expe_params.json
* result-expe-hyperparams-sensitivity.py computes figure 2 from a dataset generated with experiment #id_expe=3 configured in expe_params.json
* result-expe-environment-sensitivity.py computes figure 3 from a dataset generated with the three experiments of #id_expe=4, #id_expe=5, #id_expe=6 configured in expe_params.json
* result-expe-simulation-oc.py computes figure 4 from a dataset generated with experiment #id_expe=8 configured in expe_params.json and using a dataset extraction from Sanchez & Dunning experiments generatd with the script sanchez-dunning-dataset-extraction.py (later on we will use the raw datas as Dr. Sanchez kindly just recently sent them to us by e-mail).

Output figures (\*.pdf) as well as aggregating datasets used to plot them (\*.pkl), experiments configurations (\*.json) are to be found in ~/projects/SelfQtrust-ICLR2022/code/figures folder.

For example, save the files in the figures folder corresponding to the last experiment and run the result-expe-simulation-oc.py script to re-generate the figure 4 of the paper with the results you computed earlier:
```console
(trustenv) user@bar:~/projects/SelfQtrust-ICLR2022/code/experiments$ cd ../figures/
(trustenv) user@bar:~/projects/SelfQtrust-ICLR2022/code/figures$ mv simulate_sanchez_dunning_oc_60.json simulate_sanchez_dunning_oc_60_backup.json
user@bar:~/projects/SelfQtrust-ICLR2022/code/figures$ mv simulate_sanchez_dunning_oc_60.pdf simulate_sanchez_dunning_oc_60_backup.pdf
user@bar:~/projects/SelfQtrust-ICLR2022/code/figures$ mv simulate_sanchez_dunning_oc_60.pkl simulate_sanchez_dunning_oc_60_backup.pkl
user@bar:~/projects/SelfQtrust-ICLR2022/code/figures$ cd ../experiments/
(trustenv) user@bar:~/projects/SelfQtrust-ICLR2022/code/experiments$ python selfq-trust-expe.py -i expe_params.json -n 8
```

If everything worked well, result is in ~/projects/SelfQtrust-ICLR2022/code/figures/simulate_sanchez_dunning_oc.pdf.

You can now try other hyperparameters of the model by configuring new experiments in the expe_params.json file.

## Contents 
The main files to read to understand the implementation of the SelfQ-Trust algorithm and the experiments are:
* framework/agent.py (agent implementation to be used as a python Thread object from threading package)
* framework/qlearning.py (QLearning algorithm + SelfQ-Trust algorithm extending QLearning - see QTrust class)
* experiments/qtrusttesting.py (The QTrustTesting class is used to create a population of agents as python Thread objects. The QTrustTesting.play() function uses the SelfQ-Trust algorithm implemented in the QTrust class to measure the trust that agents have in each other and in themselves to solve the underlying MDP.)

Here is a non-exhaustive view of the important files in the repository:

```
SelfQtrust-ICLR2022
└───framework
│   │   agent.py (agent implementation)
│   │   env.py (underlying MDP environment)
│   │   maze.py (maze implementation)
│   │   qlearning.py (QLearning algorithm + QTrust algorithm extending QLearning)
│   │   storage.py
│   │   trust_env.py (trust environment)
│
└───experiments
│   │   selfq-trust-expe.py (use this script to run an experiment)
│   │   qtrusttesting.py (used by selfq-trust-expe.py)
│   │   result-expe-*.py (4 scripts to generate figures from experiments)
│   │   maze-set-generation.py (use this script to generate a set of mazes and compute their complexities)
│
└───datasets (store computed datasets from experiments)
│
└───datasetsfigures (store generated figures from computed datasets)

```
