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
from opencog.ure import ure_logger

# OpenCog Gym
from agent.OpencogAgent import OpencogAgent
from agent.utils import *
from envs.wrappers import GymWrapper

env = gym.make('Chase-v0')
# Uncomment the following to get a description of env
# help(env.unwrapped)

###############
# Chase Agent #
###############

class ChaseAgent(OpencogAgent):
    def __init__(self, env, action_space, p_goal, n_goal):
        OpencogAgent.__init__(self, env, action_space, p_goal, n_goal)


if __name__ == "__main__":
    # Init loggers
    log.set_level("debug")
    log.set_sync(False)
    agent_log.set_level("fine")
    agent_log.set_sync(False)
    ure_logger().set_level("debug")
    ure_logger().set_sync(False)

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
    action_space = {ExecutionLink(SchemaNode(a)) for a in allowed_actions}

    # ChaseAgent
    ca = ChaseAgent(wrapped_env, action_space, pgoal, ngoal)

    # Training/learning loop
    lt_iterations = 2           # Number of learning-training iterations
    lt_period = 200             # Duration of a learning-training iteration
    for i in range(lt_iterations):
        ca.reset_action_counter()
        par = ca.accumulated_reward # Keep track of the reward before
        # Discover patterns to make more informed decisions
        agent_log.info("Start learning ({}/{})".format(i + 1, lt_iterations))
        ca.learn()
        # Run agent to accumulate percepta
        agent_log.info("Start training ({}/{})".format(i + 1, lt_iterations))
        for j in range(lt_period):
            ca.step()
            time.sleep(0.01)
            log.info("step_count = {}".format(ca.step_count))
        nar = ca.accumulated_reward - par
        agent_log.info("Accumulated reward during {}th iteration = {}".format(i + 1, nar))
        agent_log.info("Action counter during {}th iteration:\n{}".format(i+1, ca.action_counter))
