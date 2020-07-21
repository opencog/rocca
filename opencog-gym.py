# OpenCog wrapper around OpenAI Gym.
#
# Highly experimental at this stage.

from opencog.atomspace import AtomSpace, TruthValue
# from opencog.atomspace import types
# from opencog.type_constructors import *

# a = AtomSpace()

# # Tell the type constructors which atomspace to use.
# set_type_ctor_atomspace(a)

import gym
env = gym.make('CartPole-v0')
observation = env.reset()
for _ in range(20):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("observation =", observation)
    print("reward =", reward)
    print("info =", info)

    # if done:
    #     observation = env.reset()
    #     break

env.close()
