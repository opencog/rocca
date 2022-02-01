# Toy Gym Environment.
#
# This is to be used to test opencog-gym on a trivial
# environment.

from enum import Enum

# Numpy
import numpy as np

# OpenAI Gym
import gym
import pyglet
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering

from rocca.envs.gym_chase.game_objects import Player, Board, Pellet

FPS = 30

SCALE = 500
WINDOW_W = 600
WINDOW_H = 400

LEFT_TRANS_X = -0.5
RIGHT_TRANS_X = 0.5

BOARD_TRANS_X = 0.0
BOARD_TRANS_Y = -0.3
PLAYER_TRANS_Y = -0.15


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    STAY = 2
    EAT = 3


class Position(Enum):
    LEFT = 0
    RIGHT = 1
    NONE = 2


class ChaseEnv(gym.Env):
    """
    Chase is board with 2 squares, Left and Right. At each step, if no food is
    present on the board, a food pellet appears on the opposite square of the
    one containing previous pellet, initially starting on the Left square.
    The agent can be positioned on either the Left or Right square and has 4
    actions, go-left and go-right, stay, eat. Upon eating the pellet the agent
    receives a reward of 1.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": FPS}

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(2), spaces.Discrete(3))  # Agent Position  # Food pellet
        )
        self.food_position = None
        self.player_position = None
        self.prev_food_position = None
        self.viewer = None
        self.reset()

    def _setup(self):
        self.player_position = Position(np.random.choice(2))
        self.food_position = Position.LEFT

    def get_player_transform(self):
        if self.player_position == Position.LEFT:
            return LEFT_TRANS_X, PLAYER_TRANS_Y
        else:
            return RIGHT_TRANS_X, PLAYER_TRANS_Y

    def get_pellet_transform(self):
        if self.food_position == Position.LEFT:
            return LEFT_TRANS_X, BOARD_TRANS_Y
        else:
            return RIGHT_TRANS_X, BOARD_TRANS_Y

    def _get_ob(self):
        return np.array([self.player_position.value, self.food_position.value])

    def _update_state(self, action):
        done = False
        reward = 0
        pfp = self.food_position

        if action == Action.EAT and self.player_position == self.food_position:
            reward = 1
            self.prev_food_position = self.food_position
            self.food_position = Position.NONE
        elif action == Action.LEFT:
            self.player_position = Position.LEFT
        elif action == Action.RIGHT:
            self.player_position = Position.RIGHT

        # wait til next step to reset food if eaten.
        if pfp == Position.NONE:
            if self.prev_food_position == Position.LEFT:
                self.food_position = Position.RIGHT
            else:
                self.food_position = Position.LEFT

        # TODO calculate termination criteria
        return reward, done

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        reward, done = self._update_state(Action(action))
        self.player.set_pos(*self.get_player_transform())
        self.pellet.set_pos(*self.get_pellet_transform())
        return self._get_ob(), reward, done, {}

    def reset(self):
        self._setup()
        self.board = Board(BOARD_TRANS_X, BOARD_TRANS_Y)
        self.player = Player(*self.get_player_transform())
        self.pellet = Pellet(*self.get_pellet_transform())
        return self._get_ob()

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.viewer.set_bounds(
                -WINDOW_W / SCALE, WINDOW_W / SCALE, -WINDOW_H / SCALE, WINDOW_H / SCALE
            )

        # TODO: display text about observation, reward, cumulative
        # reward and last action.  See
        # https://github.com/openai/gym/blob/58aeddb62fb9d46d2d2481d1f7b0a380d8c454b1/gym/envs/box2d/car_racing.py#L424
        self.board.draw(self.viewer)
        self.pellet.draw(self.viewer)
        self.player.draw(self.viewer)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([2])

    def key_press(k, mod):
        if k == key.LEFT:
            a[0] = 0
        elif k == key.RIGHT:
            a[0] = 1
        elif k == key.DOWN:
            a[0] = 2
        elif k == key.RETURN:
            a[0] = 3

        print("pressed {}".format(a))

    treward = 0
    nsteps = 0

    env = ChaseEnv()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.reset()
    while True:
        obs, r, done, info = env.step(a[0])
        a[0] = 2
        treward += r

        isopen = env.render()
        if isopen == False:
            break

        nsteps += 1
        if nsteps % 200 == 0:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in obs]))
            print("step {} total_reward {:+0.2f}".format(nsteps, treward))
            print("current action is {}".format(a[0]))

        if done or treward > 5:
            break
    env.close()
