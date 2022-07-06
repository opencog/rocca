# Define an agent for the Chase env with OpencogAgent
#
# WARNING: unmaintained

##############
# Initialize #
##############

# Python
import time

# OpenCog
from opencog.logger import log
from opencog.pln import *
from opencog.type_constructors import *
from opencog.utilities import set_default_atomspace

from rocca.agents import OpencogAgent
from rocca.agents.utils import agent_log
from rocca.envs.malmo_demo.chase_env import mission_xml
from rocca.envs.wrappers import MalmoWrapper
from opencog.ure import ure_logger

###############
# Chase Agent #
###############
from rocca.envs.wrappers.utils import mk_action


class ChaseAgent(OpencogAgent):
    def __init__(self, env, atomspace, action_space, p_goal, n_goal):
        OpencogAgent.__init__(self, env, atomspace, action_space, p_goal, n_goal)
        env.step(mk_action("hotbar.9", 1))  # Press the hotbar key
        env.step(
            mk_action("hotbar.9", 0)
        )  # Release hotbar key - agent should now be holding diamond_pickaxe

    def eat(self, i):
        self.env.step(mk_action("move", 0))
        self.env.step(mk_action("hotbar.{}".format(i), 1))
        self.env.step(mk_action("hotbar.{}".format(i), 0))
        self.env.step(mk_action("use", 1))

    def wake(self):
        self.env.step(mk_action("use", 0))
        self.env.step(mk_action("hotbar.9", 1))
        self.env.step(mk_action("hotbar.9", 0))


if __name__ == "__main__":
    atomspace = AtomSpace()
    set_default_atomspace(atomspace)

    # Init loggers
    log.set_level("info")
    # log.set_sync(True)
    agent_log.set_level("debug")
    # agent_log.set_sync(True)
    ure_logger().set_level("debug")

    # Wrap environment
    wrapped_env = MalmoWrapper(missionXML=mission_xml, validate=True)

    # Create Goal
    pgoal = EvaluationLink(PredicateNode("Reward"), NumberNode("1"))
    ngoal = EvaluationLink(PredicateNode("Reward"), NumberNode("0"))

    # Create Action Space. The set of allowed actions an agent can take.
    # TODO take care of action parameters.
    action_space = {
        ExecutionLink(SchemaNode("tpz"), NumberNode("2.5")),
        ExecutionLink(SchemaNode("tpz"), NumberNode("-1.5")),
        ExecutionLink(SchemaNode("attack"), NumberNode("0")),
        ExecutionLink(SchemaNode("attack"), NumberNode("1")),
        ExecutionLink(SchemaNode("move"), NumberNode("0")),
        ExecutionLink(SchemaNode("move"), NumberNode("0.5")),
    }

    # ChaseAgent
    ca = ChaseAgent(wrapped_env, atomspace, action_space, pgoal, ngoal)

    # Eat some food.
    ca.eat(4)
    time.sleep(5)

    # Training/learning loop
    lt_iterations = 3  # Number of learning-training iterations
    lt_period = 200  # Duration of a learning-training iteration
    for i in range(lt_iterations):
        ca.reset_action_counter()
        par = ca.accumulated_reward  # Keep track of the reward before
        # Discover patterns to make more informed decisions
        agent_log.info("Start learning ({}/{})".format(i + 1, lt_iterations))
        ca.learn()
        ca.wake()
        # Run agent to accumulate percepta
        agent_log.info("Start training ({}/{})".format(i + 1, lt_iterations))
        for j in range(lt_period):
            ca.control_cycle()
            time.sleep(0.01)
            log.info("cycle_count = {}".format(ca.cycle_count))
        nar = ca.accumulated_reward - par
        agent_log.info(
            "Accumulated reward during {}th iteration = {}".format(i + 1, nar)
        )
        agent_log.info(
            "Action counter during {}th iteration:\n{}".format(i + 1, ca.action_counter)
        )
        ca.eat(8 - i)
