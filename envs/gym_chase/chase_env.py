# Toy Gym Environment.
#
# This is to be used to test opencog-gym on a trivial
# environment.

from enum import Enum

# Numpy
import numpy as np

# OpenAI Gym
import gym
from gym import error, spaces, utils
from gym.utils import seeding

FPS = 30

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

    def step(self, action):
        pass

    def reset(self):
        self._setup()
        return self._get_ob()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
