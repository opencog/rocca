from typing import *

from gym import Env
from gym.spaces import Space

from opencog.atomspace import Atom, AtomSpace
from opencog.type_constructors import *

from .gym_wrapper import GymWrapper


class CartPoleWrapper(GymWrapper):
    def __init__(self, env: Env, atomspace: AtomSpace):
        action_names = ["Go Left", "Go Right"]
        super().__init__(env, atomspace, action_names)

    def labeled_observation(self, space: Space, obs, sbs="") -> List[Atom]:
        """Translate gym observation to Atomese

        There are 4 observations (taken from CartPoleEnv help)

        Observation               Min             Max
        -----------               ---             ---
        Cart Position             -4.8            4.8
        Cart Velocity             -Inf            Inf
        Pole Angle                -24 deg         24 deg
        Pole Angular Velocity      -Inf            Inf

        They are represented in atomese as follows

        Evaluation
          Predicate "Cart Position"
          Number CP

        Evaluation
          Predicate "Cart Velocity"
          Number CV

        Evaluation
          Predicate "Pole Angle"
          Number PA

        Evaluation
          Predicate "Pole Angular Velocity"
          Number PVAT

        Note that the observations are neither tv-set nor
        timestamped. It is up to the caller to do it.

        A python list (not an atomese list) is returned with these 4
        Atomese observations.
        """

        cp = NumberNode(str(obs[0]))
        cv = NumberNode(str(obs[1]))
        pa = NumberNode(str(obs[2]))
        pvat = NumberNode(str(obs[3]))

        return [
            EvaluationLink(PredicateNode("Cart Position"), cp),
            EvaluationLink(PredicateNode("Cart Velocity"), cv),
            EvaluationLink(PredicateNode("Pole Angle"), pa),
            EvaluationLink(PredicateNode("Pole Angular Velocity"), pvat),
        ]
