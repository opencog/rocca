# OpenCog wrapper around OpenAI Gym.
#
# Highly experimental at this stage.  For now we write an ad-hoc
# wrapper for the CartPole-v0 env, latter on this wrapper will be
# generalized to any env, and the user will have to overload of the
# various methods of that wrapper for specific env.

from opencog.atomspace import AtomSpace, TruthValue
from opencog.atomspace import types
from opencog.type_constructors import *

a = AtomSpace()

# Tell the type constructors which atomspace to use.
set_default_atomspace(a)

def to_atomese(observation):
    """Translate gym observation to Atomese

    There are 4 observations (taken from CartPoleEnv help)

    Observation               Min             Max
    -----------               ---             ---
    Cart Position             -4.8            4.8
    Cart Velocity             -Inf            Inf
    Pole Angle                -24 deg         24 deg
    Pole Velocity At Tip      -Inf            Inf

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
      Predicate "Pole Velocity At Tip"
      Number PVAT

    Note that the observations are neither tv-set nor timestamped. It
    is up to the caller to do it.

    A python list (not an atomese list) is returned with these 4
    Atomese observations.
    """

    cp = NumberNode(str(observation[0]))
    cv = NumberNode(str(observation[1]))
    pa = NumberNode(str(observation[2]))
    pvat = NumberNode(str(observation[3]))

    return [EvaluationLink(PredicateNode("Cart Position"), cp),
            EvaluationLink(PredicateNode("Cart Velocity"), cv),
            EvaluationLink(PredicateNode("Pole Angle"), pa),
            EvaluationLink(PredicateNode("Cart Velocity At Tip"), pvat)]

def timestamp(atom, i):
    TRUE_TV = TruthValue(1.0, 1.0);
    return AtTimeLink(atom, TimeNode(str(i)), tv=TRUE_TV)

import gym
env = gym.make('CartPole-v0')
# Uncomment the following to get a description of env
# help(env.unwrapped)
observation = env.reset()
for i in range(20):
    env.render()
    action = env.action_space.sample()
    print("action =", action)
    observation, reward, done, info = env.step(action)
    print("observation =", observation)
    print("reward =", reward)
    print("info =", info)

    atomese_obs = to_atomese(observation)
    print("atomese_obs =", atomese_obs)

    # Timestamp the atomese observations
    timestamped_obs = list(map(lambda o : timestamp(o, i), atomese_obs))
    print("timestamped_obs =", timestamped_obs)
    
    # if done:
    #     observation = env.reset()
    #     break

env.close()
