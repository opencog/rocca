# OpenCog wrapper around OpenAI Gym.
#
# Highly experimental at this stage.  For now we write an ad-hoc
# wrapper for the CartPole-v0 env, latter on this wrapper will be
# generalized to any env, and the user will have to overload of the
# various methods of that wrapper for specific env.

##############
# Initialize #
##############

# Python
import os
import time
from orderedmultidict import omdict

# OpenCog
from opencog.atomspace import AtomSpace, TruthValue
from opencog.atomspace import types
from opencog.atomspace import get_type, is_a
from opencog.exec import execute_atom
from opencog.type_constructors import *
from opencog.spacetime import *
from opencog.pln import *
from opencog.scheme_wrapper import *

# OpenAI Gym
import gym
env = gym.make('CartPole-v1')
# Uncomment the following to get a description of env
# help(env.unwrapped)

# OpenCog Gym
from opencog_gym.gymagent import GymAgent

##################
# CartPole Agent #
##################

class CartPoleAgent(GymAgent):
    def __init__(self):
        GymAgent.__init__(self, env)
    def atomese_action_space(self):
        return {SchemaNode("Go Left"), SchemaNode("Go Right")}

########
# Main #
########
def main():
    cpa = CartPoleAgent()
    while cpa.step():
        print("step_count = {}".format(cpa.step_count))

if __name__ == "__main__":
    main()
