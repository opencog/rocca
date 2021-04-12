from functools import wraps

import numpy as np
from fastcore.basics import listify
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

    def labeled_observations(self, space, obs, sbs=""):
        if isinstance(space, sp.Tuple):
            obs_list = []
            for s in space:
                idx = len(obs_list)
                _sbs = sbs + "-" + str(idx) if sbs else str(idx)
                obs_list.append(*self.labeled_observations(s, obs[idx], _sbs))
            return obs_list
        elif isinstance(space, sp.Box):
            label = sbs + "-Box" if sbs else "Box"
            return [mk_evaluation(label, *obs)]
        elif isinstance(space, sp.Discrete):
            label = sbs + "-Discrete" if sbs else "Discrete"
            return [mk_evaluation(label, obs)]
        elif isinstance(space, sp.Dict):
            obs_list = []
            for k in obs.keys():
                label = sbs + "-" + k if sbs else k
                if isinstance(space[k], sp.Discrete):
                    obs_list.append(mk_evaluation(label, obs[k]))
                elif isinstance(space[k], sp.Box):
                    l = (
                        listify(obs[k].tolist())
                        if isinstance(obs[k], np.ndarray)
                        else obs[k]
                    )
                    obs_list.append(mk_evaluation(label, *l))
                elif isinstance(space[k], sp.Tuple):
                    _sbs = sbs + "-" + k if sbs else k
                    obs_list.extend(self.labeled_observations(space[k], obs[k], _sbs))
                elif isinstance(space[k], sp.Dict):
                    _sbs = sbs + "-" + k if sbs else k
                    obs_list.extend(self.labeled_observations(space[k], obs[k], _sbs))
                else:
                    raise NotImplementedError("ObservationSpace not implemented.")
            return obs_list
        else:
            raise NotImplementedError("Unknown Observation Space.")

    def parse_world_state(self, ospace, obs, reward, done):
        obs_list = self.labeled_observations(ospace, obs)
        return mk_evaluation("Reward", reward), obs_list, done

    @staticmethod
    def restart_decorator(restart):
        @wraps(restart)
        def wrapper(ref):
            obs = restart(ref)
            return ref.parse_world_state(ref.observation_space, obs, 0, False)

        return wrapper

    @staticmethod
    def step_decorator(step):
        @wraps(step)
        def wrapper(ref, action):
            if isinstance(ref.action_space, sp.Discrete):
                if not len(ref.action_list) == ref.action_space.n:
                    raise ValueError("Invalid action list.")
                action_name = action.out[0].name
                if not action_name in ref.action_list:
                    raise ValueError(
                        "Action {} not known in the environment.".format(action_name)
                    )
                action_name = action.out[0].name
                obs, r, done = step(ref, ref.action_list.index(action_name))
            elif isinstance(ref.action_space, sp.Dict):
                actions = listify(action)
                act_dict = {
                    action.out[0].name: to_python(action.out[1]) for action in actions
                }
                obs, r, done = step(ref, act_dict)
            else:
                raise NotImplementedError("Unknown action space.")
            return ref.parse_world_state(ref.observation_space, obs, r, done)

        return wrapper

    @restart_decorator.__func__
    def restart(self):
        # self.env.render()  FIXME: is this necessary here?
        return self.env.reset()

    @step_decorator.__func__
    def step(self, action):
        obs, r, done, _ = self.env.step(action)
        self.env.render()
        return obs, r, done

    def close(self):
        self.env.close()
