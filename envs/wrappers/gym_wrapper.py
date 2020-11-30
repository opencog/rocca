from functools import wraps
from builtins import range
import os
import sys
import time
import json
from gym import spaces as sp

from .utils import *
from .wrapper import Wrapper


class GymWrapper(Wrapper):
    def __init__(self, env, action_list=[]):
        super()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_list = action_list

    @staticmethod
    def restart_decorator(restart):
        pass

    @staticmethod
    def step_decorator(step):
        pass

    @restart_decorator.__func__
    def restart(self):
        self.env.render()
        return self.env.reset();

    @step_decorator.__func__
    def step(self, action):
        obs, r, done, _ = self.env.step(action)
        is_open = self.env.render()
        if not is_open:
            done = True
        return obs, r, done

    def close(self):
        self.env.close()
