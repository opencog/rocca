# Define an agent for the CartPole env with GymAgent

##############
# Initialize #
##############

# Python
import time
from typing import List

# OpenCog
from opencog.atomspace import AtomSpace
from opencog.utilities import set_default_atomspace
from opencog.type_constructors import *
from opencog.spacetime import *
from opencog.pln import *
from opencog.ure import ure_logger
from opencog.logger import log

# OpenAI Gym
import gym

env = gym.make("CartPole-v1")
# Uncomment the following to get a description of env
# help(env.unwrapped)

# OpenCog Gym
from rocca.agents.cartpole import FixedCartPoleAgent
from rocca.agents.utils import *
from rocca.envs.wrappers import GymWrapper, CartPoleWrapper

from rocca.utils import *


########
# Main #
########
def main():
    # Init loggers
    log.set_level("fine")
    log.set_sync(False)
    agent_log.set_level("fine")
    agent_log.set_sync(False)
    ure_logger().set_level("fine")
    ure_logger().set_sync(False)

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
        wrapped_env.render()
        time.sleep(0.1)
        log.info("cycle_count = {}".format(cpa.cycle_count))

    log_msg(agent_log, f"The final reward is {cpa.accumulated_reward}.")


if __name__ == "__main__":
    main()
