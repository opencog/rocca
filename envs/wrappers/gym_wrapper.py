from functools import wraps
from builtins import range
import os
import sys
import time
import json
from gym import spaces as sp

from .utils import *
from .wrapper import Wrapper


def labeled_observations(space, obs):
    pass


class GymWrapper(Wrapper):
    def __init__(self, env, action_list=[]):
        super()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_list = action_list

    @staticmethod
    def parse_world_state(ospace, obs, reward, done):
        obs_list = labeled_observations(ospace, obs)
        return mk_evaluation("Reward", reward), \
               obs_list, \
               done

    @staticmethod
    def restart_decorator(restart):
        pass

    @staticmethod
    def step_decorator(step):
        @wraps(step)
        def wrapper(ref, action):
            if isinstance(ref.action_space, sp.Discrete):
                if not len(ref.action_list) == ref.action_space.n:
                    raise ValueError("Invalid action list.")
                action_name = action.out[0].name
                if not action_name in ref.action_list:
                    raise ValueError("Action {} not known in the environment.".format(action_name))
                action_name = action.out[0].name
                obs, r, done = step(ref, ref.action_list.index(action_name))
            elif isinstance(ref.action_space, sp.Dict):
                if not len(action) == 1:
                    raise NotImplementedError("Multiple actions not supported.")
                obs, r, done = step(ref, {action.out[0].name: action.out[1].name})
            else:
                raise NotImplementedError("Unknown action space.")
            return GymWrapper.parse_world_state(ref.observation_space, obs, r, done)

        return wrapper

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
