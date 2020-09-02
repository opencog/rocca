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

FPS = 30

WINDOW_W = 500
WINDOW_H = 400

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

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(2), # Agent Position
            spaces.Discrete(3)  # Food pellet
        ))
        self.food_position = None
        self.player_position = None
        self.prev_food_position = None
        self.viewer = None

    def _setup(self):
        self.player_position = Position(np.random.choice(2))
        self.food_position = Position.LEFT

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
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        reward, done = self._update_state(Action(action))
        return self._get_ob(), reward, done, {}

    def reset(self):
        self._setup()
        return self._get_ob()

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
        # TODO Draw objects.
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
