# Define an agent for the Chase env with OpencogAgent

##############
# Initialize #
##############

# Python
import time

# OpenAI Gym
import gym

# OpenCog
from opencog.type_constructors import (
    ExecutionLink,
    SchemaNode,
    EvaluationLink,
    PredicateNode,
    NumberNode,
    ConceptNode,
)
from opencog.utilities import set_default_atomspace
from opencog.pln import *
from opencog.ure import ure_logger
from opencog.logger import log

# ROCCA
from rocca.agents import OpencogAgent
from rocca.agents.utils import *
from rocca.envs.wrappers import GymWrapper

env = gym.make("Chase-v0")
# Uncomment the following to get a description of env
# help(env.unwrapped)

#################
# Chase Wrapper #
#################
# TODO: move to a library module.
class ChaseWrapper(GymWrapper):
    def __init__(self, env, atomspace):
        action_list = ["Go Left", "Go Right", "Stay", "Eat"]
        super().__init__(env, atomspace, action_list)

    def labeled_observation(self, space, obs, sbs=""):
        """Translate gym observation to Atomese

        There are 2 gym observations:

        Agent Position is 0 (left) or 1 (right)
        Pellet Positon is 0 (left), 1 (right) or 2 (none)

        Translated in Atomese as follows:

        Evaluation
          Predicate "Agent Position"
          AP

        where AP can be

        1. Concept "Left Square"
        2. Concept "Right Square"

        Evaluation
          Predicate "Pellet Position"
          PP

        where PP can be

        1. Concept "Left Square"
        2. Concept "Right Square"
        3. Concept "None"

        """

        to_atomese_position = {
            0: ConceptNode("Left Square"),
            1: ConceptNode("Right Square"),
            2: ConceptNode("None"),
        }
        ap = to_atomese_position[obs[0]]
        pp = to_atomese_position[obs[1]]
        return [
            EvaluationLink(PredicateNode("Agent Position"), ap),
            EvaluationLink(PredicateNode("Pellet Position"), pp),
        ]


###############
# Chase Agent #
###############


class ChaseAgent(OpencogAgent):
    def __init__(self, env, atomspace):
        set_default_atomspace(atomspace)

        # Create Action Space. The set of allowed actions an agent can take.
        # TODO take care of action parameters.
        action_space = {ExecutionLink(SchemaNode(a)) for a in env.action_names}

        # Create Goal
        pgoal = EvaluationLink(PredicateNode("Reward"), NumberNode("1"))
        ngoal = EvaluationLink(PredicateNode("Reward"), NumberNode("0"))

        # Call super ctor
        OpencogAgent.__init__(self, env, atomspace, action_space, pgoal, ngoal)

        # Overwrite some OpencogAgent parameters
        self.polyaction_mining = False
        self.monoaction_general_succeedent_mining = True
        self.temporal_deduction = True
        self.cogscm_minimum_strength = 0.9
        self.cogscm_maximum_shannon_entropy = 1
        self.cogscm_maximum_differential_entropy = 0
        self.cogscm_maximum_variables = 0


if __name__ == "__main__":
    # Set main atomspace
    atomspace = AtomSpace()
    set_default_atomspace(atomspace)

    # Init loggers
    log.set_level("info")
    # log.set_sync(True)
    agent_log.set_level("info")
    # agent_log.set_sync(True)
    ure_logger().set_level("info")
    # ure_logger().set_sync(True)
    miner_log = MinerLogger(atomspace)
    miner_log.set_level("info")
    # miner_log.set_sync(True)

    # Wrap environment
    wrapped_env = ChaseWrapper(env, atomspace)

    # ChaseAgent
    cag = ChaseAgent(wrapped_env, atomspace)

    # Log all parameters of cag, useful for debugging
    cag.log_parameters(level="debug")

    # Uncomment to load previously saved percepta or cogscms
    # atomspaces.
    # cag.load_cogscms_atomspace("chase_cogscms_atomspace.scm")
    # cag.load_percepta_atomspace("chase_percepta_atomspace.scm")

    # Training/learning loop
    lt_iterations = 2  # Number of learning-training iterations
    lt_period = 200  # Duration of a learning-training iteration
    max_cycle_count = cag.cycle_count + lt_iterations * lt_period
    for i in range(lt_iterations):
        wrapped_env.restart()
        cag.reset_action_counter()
        par = cag.accumulated_reward  # Keep track of the reward before

        # Discover patterns to make more informed decisions
        agent_log.info("Start learning ({}/{})".format(i + 1, lt_iterations))
        cag.learn()

        # Run agent to accumulate percepta
        agent_log.info("Start training ({}/{})".format(i + 1, lt_iterations))
        for j in range(lt_period):
            cag.control_cycle()
            wrapped_env.render()
            time.sleep(0.1)
            agent_log.info(
                "Control cycle ({}/{})".format(
                    cag.cycle_count, max_cycle_count
                )
            )
        nar = cag.accumulated_reward - par
        agent_log.info(
            "Accumulated reward during {}th iteration = {}".format(i + 1, nar)
        )
        agent_log.info(
            "Action counter during {}th iteration:\n{}".format(
                i + 1, cag.action_counter_to_str()
            )
        )

    # Log all agent's atomspaces at the end (at fine level)
    agent_log_atomspace(cag.atomspace, "fine", "cag.atomspace")
    agent_log_atomspace(cag.percepta_atomspace, "fine", "cag.percepta_atomspace")
    agent_log_atomspace(cag.cogscms_atomspace, "fine", "cag.cogscms_atomspace")
    agent_log_atomspace(cag.working_atomspace, "fine", "cag.working_atomspace")

    # Uncomment to save percepta or cogscms atomspaces
    # cag.save_percepta_atomspace("chase_percepta_atomspace.scm")
    # cag.save_cogscms_atomspace("chase_cogscms_atomspace.scm")
