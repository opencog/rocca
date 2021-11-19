import time

import gym

from opencog.atomspace import AtomSpace
from opencog.utilities import set_default_atomspace

from rocca.utils import *
from rocca.agents.utils import *
from rocca.envs.wrappers import CartPoleWrapper
from rocca.agents.cartpole import FixedCartPoleAgent


# Test if a simple CartPole run works
def test_cartpole():
    env = gym.make("CartPole-v1")

    # Set main atomspace
    atomspace = AtomSpace()
    set_default_atomspace(atomspace)

    # Wrap environment
    wrapped_env = CartPoleWrapper(env, atomspace)

    # Instantiate CartPoleAgent, and tune parameters
    cpa = FixedCartPoleAgent(wrapped_env, atomspace)
    cpa.delta = 1.0e-16

    # Run control loop
    while not cpa.control_cycle():
        time.sleep(0.1)
