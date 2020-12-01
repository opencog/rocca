# Define an agent for the Chase env with OpencogAgent

##############
# Initialize #
##############

# Python
import time

# OpenAI Gym
import gym
# OpenCog
from opencog.pln import *

# OpenCog Gym
from agent.OpencogAgent import OpencogAgent
from agent.utils import *
from envs.wrappers import GymWrapper

env = gym.make('Chase-v0')
# Uncomment the following to get a description of env
# help(env.unwrapped)

if __name__ == "__main__":
    atomspace = AtomSpace()
    set_default_atomspace(atomspace)
    # Wrap environment
    # Allowed_actions is not required if the gym environment's action
    # space is labeled a.k.a space.Dict.
    allowed_actions = ["Go Left", "Go Right", "Stay", "Eat"]
    wrapped_env = GymWrapper(env, allowed_actions)

    # Create Goal
    pgoal = EvaluationLink(PredicateNode("Reward"), NumberNode("1"))
    ngoal = EvaluationLink(PredicateNode("Reward"), NumberNode("0"))

    # Create Action Space. The set of allowed actions an agent can take.
    # TODO take care of action parameters.
    action_space = {ExecutionLink(SchemaNode("Go Left")),
                    ExecutionLink(SchemaNode("Go Right")),
                    ExecutionLink(SchemaNode("Stay")),
                    ExecutionLink(SchemaNode("Eat"))}

    # OpencogAgent
    oa = OpencogAgent(wrapped_env, action_space, pgoal, ngoal)
    lt_iterations = 2           # Number of learning-training iterations
    lt_period = 200             # Duration of a learning-training iteration
    for i in range(lt_iterations):
        par = oa.accumulated_reward # Keep track of the reward before
        # Discover patterns to make more informed decisions
        log.info("Start learning ({}/{})".format(i + 1, lt_iterations))
        oa.learn()
        # Run agent to accumulate percepta
        log.info("Start training ({}/{})".format(i + 1, lt_iterations))
        for j in range(lt_period):
            oa.step()
            time.sleep(0.01)
            log.info("step_count = {}".format(oa.step_count))
        nar = oa.accumulated_reward - par
        log.info("Accumulated reward during {}th iteration = {}".format(i + 1, nar))
