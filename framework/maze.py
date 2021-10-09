import random
import numpy as np
import os

# Customized implementation of a Maze Environment
# TODO for future works: replace by gym-maze
# or any other gym-compatible maze usable by coax, MushroomRL, Dopamine


class MazeView:
    class ActionSpace(object):
        n = 4

    def __init__(
        self, agent_dict, maze_file_path=None, maze_size=(10, 10), has_loops=False,
    ):
        # Agent list
        self.__agent_dict = agent_dict

        # Load a maze
        if maze_file_path is None:
            self.__maze = Maze(maze_size=maze_size, has_loops=has_loops)
        else:
            if not os.path.exists(maze_file_path):
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(dir_path, "maze_samples", maze_file_path)
                if os.path.exists(rel_path):
                    maze_file_path = rel_path
                else:
                    raise FileExistsError("Cannot find %s." % maze_file_path)
            self.__maze = Maze(maze_cells=Maze.load_maze(maze_file_path))

        self.__action_space = self.ActionSpace
        self.maze_size = self.__maze.maze_size

        # Set the starting point
        self.__entrance = np.zeros(2, dtype=int)

        # Set the Goal
        self.__goal = np.array(self.maze_size) - np.array((1, 1))

        # Create the Robots
        self.__agent_state = dict()
        for name in self.__agent_dict.keys():
            self.__agent_state[name] = self.entrance

    def move_agent(self, agent, dir):
        ACTION = ["N", "S", "E", "W"]

        if isinstance(dir, int):
            if dir in [0, 1, 2, 3]:
                dir = ACTION[dir]
            else:
                raise ValueError("if dir is an int, dir must be in [0,1,2,3]")

        if dir not in self.__maze.COMPASS.keys():
            raise ValueError(
                "dir cannot be %s. The only valid dirs are %s."
                % (str(dir), str(self.__maze.COMPASS.keys()))
            )

        if self.__maze.is_open(self.__agent_state[agent], dir):

            # move the agent
            self.__agent_state[agent] = self.__agent_state[agent] + np.array(
                self.__maze.COMPASS[dir]
            )

        return tuple(self.__agent_state[agent])

    def reset_agent(self, agent):
        self.__agent_state[agent] = np.zeros(2, dtype=int)
        return tuple(self.__agent_state[agent])

    @property
    def maze(self):
        return self.__maze

    @property
    def action_space(self):
        return self.__action_space

    @property
    def entrance(self):
        return self.__entrance

    @property
    def goal(self):
        return self.__goal

    @property
    def size(self):
        return self.maze_size

    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)

    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.maze.MAZE_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.maze.MAZE_H)


class Maze:

    COMPASS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}

    def __init__(self, maze_cells=None, maze_size=(10, 10), has_loops=True):

        # maze member variables
        self.maze_cells = maze_cells
        self.has_loops = has_loops

        # Use existing one if exists
        if self.maze_cells is not None:
            if (
                isinstance(self.maze_cells, (np.ndarray, np.generic))
                and len(self.maze_cells.shape) == 2
            ):
                self.maze_size = tuple(maze_cells.shape)
            else:
                raise ValueError("maze_cells must be a 2D NumPy array.")
        # Otherwise, generate a random one
        else:
            # maze's configuration parameters
            if not (isinstance(maze_size, (list, tuple)) and len(maze_size) == 2):
                raise ValueError("maze_size must be a tuple: (width, height).")
            self.maze_size = maze_size

            self._generate_maze()

    def save_maze(self, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(os.path.dirname(file_path)):
            raise ValueError("Cannot find the directory for %s." % file_path)

        else:
            np.save(file_path, self.maze_cells, allow_pickle=False, fix_imports=True)

    @classmethod
    def load_maze(cls, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(file_path):
            raise ValueError("Cannot find %s." % file_path)

        else:
            return np.load(file_path, allow_pickle=False, fix_imports=True)

    def _generate_maze(self):

        # initialize random generator
        # random.seed(seed)

        # list of all cell locations
        self.maze_cells = np.zeros(self.maze_size, dtype=int)

        # Initializing constants and variables needed for maze generation
        current_cell = (
            random.randint(0, self.MAZE_W - 1),
            random.randint(0, self.MAZE_H - 1),
        )
        num_cells_visited = 1
        cell_stack = [current_cell]

        # Continue until all cells are visited
        while cell_stack:

            # restart from a cell from the cell stack
            current_cell = cell_stack.pop()
            x0, y0 = current_cell

            # find neighbours of the current cells that actually exist
            neighbours = dict()
            for dir_key, dir_val in self.COMPASS.items():
                x1 = x0 + dir_val[0]
                y1 = y0 + dir_val[1]
                # if cell is within bounds
                if 0 <= x1 < self.MAZE_W and 0 <= y1 < self.MAZE_H:
                    # if all four walls still exist
                    if self.all_walls_intact(self.maze_cells[x1, y1]):
                        # if self.num_walls_broken(self.maze_cells[x1, y1]) <= 1:
                        neighbours[dir_key] = (x1, y1)

            # if there is a neighbour
            if neighbours:
                # select a random neighbour
                dir = random.choice(tuple(neighbours.keys()))
                x1, y1 = neighbours[dir]

                # knock down the wall between the current cell and the selected neighbour
                self.maze_cells[x1, y1] = self.__break_walls(
                    self.maze_cells[x1, y1], self.__get_opposite_wall(dir)
                )

                # push the current cell location to the stack
                cell_stack.append(current_cell)

                # make the this neighbour cell the current cell
                cell_stack.append((x1, y1))

                # increment the visited cell count
                num_cells_visited += 1

        if self.has_loops:
            self.__break_random_walls(0.2)

    def __break_random_walls(self, percent):
        # find some random cells to break
        num_cells = int(round(self.MAZE_H * self.MAZE_W * percent))
        cell_ids = random.sample(range(self.MAZE_W * self.MAZE_H), num_cells)

        # for each of those walls
        for cell_id in cell_ids:
            x = cell_id % self.MAZE_H
            y = int(cell_id / self.MAZE_H)

            # randomize the compass order
            dirs = random.sample(list(self.COMPASS.keys()), len(self.COMPASS))
            for dir in dirs:
                # break the wall if it's not already open
                if self.is_breakable((x, y), dir):
                    self.maze_cells[x, y] = self.__break_walls(
                        self.maze_cells[x, y], dir
                    )
                    break

    def is_open(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        # if cell is still within bounds after the move
        if self.is_within_bound(x1, y1):
            # check if the wall is opened
            this_wall = bool(
                self.get_walls_status(self.maze_cells[cell_id[0], cell_id[1]])[dir]
            )
            other_wall = bool(
                self.get_walls_status(self.maze_cells[x1, y1])[
                    self.__get_opposite_wall(dir)
                ]
            )
            return this_wall or other_wall
        return False

    def is_breakable(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        return not self.is_open(cell_id, dir) and self.is_within_bound(x1, y1)

    def is_within_bound(self, x, y):
        # true if cell is still within bounds after the move
        return 0 <= x < self.MAZE_W and 0 <= y < self.MAZE_H

    @property
    def MAZE_W(self):
        return int(self.maze_size[0])

    @property
    def MAZE_H(self):
        return int(self.maze_size[1])

    @classmethod
    def get_walls_status(cls, cell):
        walls = {
            "N": (cell & 0x1) >> 0,
            "E": (cell & 0x2) >> 1,
            "S": (cell & 0x4) >> 2,
            "W": (cell & 0x8) >> 3,
        }
        return walls

    @classmethod
    def all_walls_intact(cls, cell):
        return cell & 0xF == 0

    @classmethod
    def num_walls_broken(cls, cell):
        walls = cls.get_walls_status(cell)
        num_broken = 0
        for wall_broken in walls.values():
            num_broken += wall_broken
        return num_broken

    @classmethod
    def __break_walls(cls, cell, dirs):
        if "N" in dirs:
            cell |= 0x1
        if "E" in dirs:
            cell |= 0x2
        if "S" in dirs:
            cell |= 0x4
        if "W" in dirs:
            cell |= 0x8
        return cell

    @classmethod
    def __get_opposite_wall(cls, dirs):

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        opposite_dirs = ""

        for dir in dirs:
            if dir == "N":
                opposite_dir = "S"
            elif dir == "S":
                opposite_dir = "N"
            elif dir == "E":
                opposite_dir = "W"
            elif dir == "W":
                opposite_dir = "E"
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            opposite_dirs += opposite_dir

        return opposite_dirs

    @property
    def locations(self):
        return self.__locations
