from framework.maze import MazeView


class MazeEnv:
    def __init__(
        self,
        agent_dict,
        pipeline,
        maze_file=None,
        maze_size=None,
        save=False,
        has_loops=False,
    ):

        self.agent_dict = agent_dict

        if maze_size:
            self.maze = MazeView(
                agent_dict=agent_dict, maze_size=maze_size, has_loops=has_loops,
            )
            if save:
                self.maze.maze.save_maze(maze_file)
        elif maze_file:
            self.maze = MazeView(agent_dict=agent_dict, maze_file_path=maze_file,)
        else:
            raise AttributeError(
                "One must supply either a maze_file path (str) or the maze_size (tuple of length 2)"
            )

        self.size = self.maze.size
        self.goal_position = tuple(self.maze.goal)
        self.action_space = self.maze.action_space
        for agent_name in agent_dict.keys():
            self.reset(agent_name)

        self.pipeline = pipeline

        # controling variable
        self.learn_on = True
        self.stepbystep_on = False
        self.visu_on = True

    def step(self, agent_name, action):
        state = self.maze.move_agent(agent_name, action)

        if state == self.goal_position:
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        info = {}

        return state, reward, done, info

    def reset(self, agent_name):
        state = self.maze.reset_agent(agent_name)
        return state

    def run(self):
        while True:
            message = self.pipeline.get()
            agent_name = message[0]
            callback = message[1]
            params = message[2]

            # TODO: writing introspection should be more elegant
            if callback == "draw_macro_params":
                self.maze.draw_macro_params(agent_name, params)
            elif callback == "draw_trust_learning_params":
                self.maze.draw_trust_learning_params(agent_name, params)
            elif callback == "draw_episode":
                self.maze.draw_episode(agent_name, params)
            elif callback == "draw_trust_maps":
                self.maze.draw_trust_maps(agent_name, params)
            elif callback == "draw_trust_learning":
                self.maze.draw_trust_learning(agent_name, params)
            elif callback == "draw_trust_vs_score":
                self.maze.draw_trust_vs_score(agent_name, params)
            elif callback == "draw_score_vs_episode":
                self.maze.draw_score_vs_episode(agent_name, params)
            elif callback == "draw_trust_compromise" and self.visu_on:
                self.maze.draw_trust_compromise(agent_name, params)
            elif callback == "update_trust_maps" and self.visu_on:
                self.maze.update_trust_maps(agent_name, params)

    def control(self):
        while True:
            out = self.maze.keyboard_control_learning()
            if out == "learn_on_off":
                self.stepbystep_on = False
                self.learn_on = not self.learn_on
            if out == "step_by_step":
                self.stepbystep_on = True
                self.learn_on = True
            if out == "visu_on_off":
                self.visu_on = not self.visu_on
