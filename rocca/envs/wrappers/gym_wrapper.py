from functools import wraps

import numpy as np
from fastcore.basics import listify
from gym import Env
from gym import spaces as sp
from gym.spaces import Space
from opencog.atomspace import Atom, AtomSpace
from opencog.utilities import set_default_atomspace

from .utils import *
from .wrapper import Wrapper


class GymWrapper(Wrapper):
    def __init__(self, env: Env, atomspace: AtomSpace, action_names: list[str] = []):
        super().__init__()

        self.atomspace = atomspace
        set_default_atomspace(self.atomspace)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_names = action_names

    def transform_percept(self, label: str, *args) -> list[Atom]:
        """A function to apply custom transforms on observations"""

        return [mk_evaluation(label, *args)]

    def labeled_observation(self, space: Space, obs, sbs="") -> list[Atom]:
        """The main processing block from Gym observations to Atomese

        Uses Gym's `Space` types to determine what kind of data structure is passed
        and produces a generic Atomese representation from it.
        Use the `sbs` argument to add more into resulting Atom names.

        Returns a list of created Atoms.
        """

        if isinstance(space, sp.Tuple):
            observation: List[Atom] = []
            for s in space:
                idx = len(observation)
                _sbs = sbs + "-" + str(idx) if sbs else str(idx)
                observation.extend(self.labeled_observation(s, obs[idx], _sbs))
            return observation
        elif isinstance(space, sp.Box):
            label = sbs + "-Box" if sbs else "Box"
            return self.transform_percept(label, *obs)
        elif isinstance(space, sp.Discrete):
            label = sbs + "-Discrete" if sbs else "Discrete"
            return self.transform_percept(label, obs)
        elif isinstance(space, sp.Dict):
            observation: List[Atom] = []
            for k in obs.keys():
                label = sbs + "-" + k if sbs else k
                if isinstance(space[k], sp.Discrete):
                    observation += self.transform_percept(label, obs[k])
                elif isinstance(space[k], sp.Box):
                    l = (
                        obs[k].tolist()
                        if isinstance(obs[k], np.ndarray)
                        else listify(obs[k])
                    )
                    observation += self.transform_percept(label, *l)
                elif isinstance(space[k], sp.Tuple):
                    _sbs = sbs + "-" + k if sbs else k
                    observation.extend(self.labeled_observation(space[k], obs[k], _sbs))
                elif isinstance(space[k], sp.Dict):
                    _sbs = sbs + "-" + k if sbs else k
                    observation.extend(self.labeled_observation(space[k], obs[k], _sbs))
                else:
                    raise NotImplementedError("ObservationSpace not implemented.")
            return observation
        else:
            raise NotImplementedError("Unknown Observation Space.")

    def parse_world_state(
        self, ospace: Space, obs, reward: int, done: bool
    ) -> Tuple[List[Atom], Atom, bool]:
        """Return a triple of `observation, reward, done` - in Atomese representation

        The `done` variable signifies whether the Gym environment is done.
        """

        observation = self.labeled_observation(ospace, obs)
        return observation, mk_evaluation("Reward", reward), done

    @staticmethod
    def restart_decorator(restart):
        @wraps(restart)
        def wrapper(ref: "GymWrapper"):
            obs = restart(ref)
            return ref.parse_world_state(ref.observation_space, obs, 0, False)

        return wrapper

    @staticmethod
    def step_decorator(step):
        @wraps(step)
        def wrapper(ref: "GymWrapper", action):
            if isinstance(ref.action_space, sp.Discrete):
                if not len(ref.action_names) == ref.action_space.n:
                    raise ValueError("Invalid action list.")
                action_name = action.out[0].name
                if not action_name in ref.action_names:
                    raise ValueError(
                        "Action {} not known in the environment.".format(action_name)
                    )
                action_name = action.out[0].name
                obs, r, done = step(ref, ref.action_names.index(action_name))
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
        return self.env.reset()

    @step_decorator.__func__
    def step(self, action):
        obs, r, done, _ = self.env.step(action)
        return obs, r, done

    def close(self):
        self.env.close()

    def render(self, mode: str = "human"):
        return self.env.render(mode)
