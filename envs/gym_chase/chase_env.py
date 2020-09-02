# Toy Gym Environment.
#
# This is to be used to test opencog-gym on a trivial
# environment.

# OpenAI Gym
import gym
from gym import error, spaces, utils
from gym.utils import seeding

FPS = 30

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
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
