# TODO for future works: replace MazeEnv by gym-maze
# or any other gym-compatible maze usable by coax, MushroomRL, Dopamine

import sys

# If you run in a terminal, following file enables to find the framework package
sys.path.append("..")

# SET PARAMETERS FOR MAZE SET GENERATION
params = {
    "maze_size": (10, 10),
    "episodes": 1000,
    "learning_rate": 0.1,
    "discount": 0.9,
    "replay_number": 1,
    "number_maze_generated": 2,
}

import os
import logging
import random
import numpy as np
import pandas as pd
import queue
from framework.env import MazeEnv

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
logging.info(f"************************************")
logging.info(f"**** Maze Set Generation************")

# set directory to store a new maze folder
base_path = os.path.abspath(os.path.dirname(__file__))
maze_file_dir_name = f"mazes{params['maze_size'][0]}x{params['maze_size'][1]}"
if base_path.split(os.path.sep)[-1] != "experiments":
    input(
        "are you sure you want create a new maze set folder in {base_path}? Press any key to continue (Ctrl+C to exit)"
    )
maze_file_dir_base = os.path.join(base_path, maze_file_dir_name)
maze_file_dir = maze_file_dir_base
done = False
i = 0
while not done:
    if not os.path.exists(maze_file_dir):
        os.mkdir(maze_file_dir)
        done = True
    else:
        i += 1
        maze_file_dir = f"{maze_file_dir_base}_{i}"

logging.info(f"Creation of folder: maze_file_dir")

# set output filenames
generic_maze_file_name = (
    f"maze{params['maze_size'][0]}x{params['maze_size'][1]}_MAZEID.npy"
)
filename_maze_set_description = os.path.join(maze_file_dir, "maze_set_description.pkl",)

# init output description dataframe
df = pd.DataFrame(columns=["id", "filename", "complexity",])


def choose_action(agent_state):
    if np.random.random() > epsilon:
        action = int(np.argmax(q_table[agent_state]))
    else:
        action = random.choice(possible_actions)

    return action


def update_q_table(action, agent_state, agent_state_, reward, done):
    if not done:
        current_q = q_table[agent_state + (action,)]
        max_future_q = np.max(q_table[agent_state_])
        new_q = (1 - params["learning_rate"]) * current_q + params["learning_rate"] * (
            reward + params["discount"] * max_future_q
        )
        q_table[agent_state + (action,)] = new_q
    else:
        q_table[agent_state + (action,)] = reward


for id in range(params["number_maze_generated"]):
    # create environnement
    epsilon = 1
    maze_file_name = generic_maze_file_name.replace("MAZEID", str(id))
    maze_file = os.path.join(maze_file_dir, maze_file_name,)
    agent_name = "Alice"
    agent_dict = {
        agent_name: (136, 170, 255, 255),
    }
    pipeline = queue.Queue(maxsize=10)

    env = MazeEnv(
        agent_dict=agent_dict,
        pipeline=pipeline,
        maze_file=maze_file,
        maze_size=params["maze_size"],
        save=True,
        has_loops=False,
    )

    # variables
    agent_state = env.reset(agent_name)
    possible_actions = list(range(env.action_space.n))
    q_table = np.zeros(shape=([env.size[0], env.size[1]] + [env.action_space.n]))

    score = 10000000
    for j in range(params["replay_number"]):
        for episode in range(params["episodes"]):
            agent_state = env.reset(agent_name)
            done = False

            score_episode = 0
            while not done:
                # Choose an action using epsilon-greedy
                action = choose_action(agent_state)

                agent_state_, reward, done, _ = env.step(agent_name, action)
                update_q_table(action, agent_state, agent_state_, reward, done)
                agent_state = agent_state_

                score_episode += 1

            if episode <= params["episodes"] // 2:
                epsilon -= epsilon / (params["episodes"] // 2)  # Decay epsilon
            if score_episode < score:
                score = score_episode

            logging.info(
                f"episode: {episode} ; score_episode: {score_episode} ; score: {score}"
            )

    print(f"ID: {id} ; score: {score}")
    new_row = {"id": id, "filename": maze_file_name, "complexity": score}

    df = df.append(new_row, ignore_index=True)

df.to_pickle(filename_maze_set_description)
print(df)
