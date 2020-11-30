# Define an agent for the Chase env with GymAgent

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
from opencog.utilities import is_closed
from opencog.type_constructors import *
from opencog.spacetime import *
from opencog.pln import *
from opencog.exec import execute_atom
from opencog.scheme_wrapper import scheme_eval, scheme_eval_h
from opencog.logger import Logger, log
from opencog.ure import ure_logger

# OpenAI Gym
import gym
from envs.gym_chase.chase_env import ChaseEnv
env = gym.make('Chase-v0')
# Uncomment the following to get a description of env
# help(env.unwrapped)

# OpenCog Gym
from agent.gymagent import GymAgent
from agent.utils import *

##################
# CartPole Agent #
##################

class ChaseAgent(GymAgent):
    def __init__(self):
        GymAgent.__init__(self, env)

        # Init loggers
        log.set_level("debug")
        # log.set_sync(True)
        # ure_logger().set_level("fine")
        # ure_logger().set_sync(True)

        # Load miner
        scheme_eval(self.atomspace, "(use-modules (opencog miner))")
        # scheme_eval(self.atomspace, "(miner-logger-set-level! \"fine\")")
        # scheme_eval(self.atomspace, "(miner-logger-set-sync! #t)")

        # Load PLN
        scheme_eval(self.atomspace, "(use-modules (opencog pln))")
        # scheme_eval(self.atomspace, "(pln-load-rule 'predictive-implication-scope-direct-introduction)")
        scheme_eval(self.atomspace, "(pln-load-rule 'predictive-implication-scope-direct-evaluation)")
        scheme_eval(self.atomspace, "(pln-log-atomspace)")

    def gym_observation_to_atomese(self, observation):
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

        to_atomese_position = {0 : ConceptNode("Left Square"),
                               1 : ConceptNode("Right Square"),
                               2 : ConceptNode("None")}
        ap = to_atomese_position[observation[0]]
        pp = to_atomese_position[observation[1]]
        return [EvaluationLink(PredicateNode("Agent Position"), ap),
                EvaluationLink(PredicateNode("Pellet Position"), pp)]

    def atomese_action_space(self):
        return {SchemaNode("Go Left"),
                SchemaNode("Go Right"),
                SchemaNode("Stay"),
                SchemaNode("Eat")}

    def atomese_action_to_gym(self, action):
        """Map atomese actions to gym actions

        SchemaNode("Go Left") -> 0
        SchemaNode("Go Right") -> 1
        SchemaNode("Say") -> 2
        SchemaNode("Eat") -> 3

        """

        if SchemaNode("Go Left") == action:
            return 0
        if SchemaNode("Go Right") == action:
            return 1
        if SchemaNode("Stay") == action:
            return 2
        if SchemaNode("Eat") == action:
            return 3

    def positive_goal(self):
        return EvaluationLink(PredicateNode("Reward"), NumberNode("1"))

    def negative_goal(self):
        return EvaluationLink(PredicateNode("Reward"), NumberNode("0"))

    def plan(self, goal, expiry):

        """Plan the next actions given a goal and its expiry time offset

        Return a python list of cognivite schematics.  Whole cognitive
        schematics are output (instead of action plans) in order to
        make a decision based on their truth values.

        The format for a cognitive schematic is as follows

        PredictiveImplicationScope <tv>
          <vardecl>
          <expiry>
          And (or SimultaneousAnd?)
            <context>
            Execution
              <action>
              <input> [optional]
              <output> [optional]
          <goal>

        """

        return self.plan_pln_xp(goal, expiry)


########
# Main #
########
def main():
    ca = ChaseAgent()
    lt_iterations = 2           # Number of learning-training iterations
    lt_period = 200             # Duration of a learning-training iteration
    for i in range(lt_iterations):
        par = ca.accumulated_reward # Keep track of the reward before
        # Discover patterns to make more informed decisions
        log.info("Start learning ({}/{})".format(i + 1, lt_iterations))
        ca.learn()
        # Run agent to accumulate percepta
        log.info("Start training ({}/{})".format(i + 1, lt_iterations))
        for j in range(lt_period):
            ca.step()
            time.sleep(0.01)
            log.info("step_count = {}".format(ca.step_count))
        nar = ca.accumulated_reward - par
        log.info("Accumulated reward during {}th iteration = {}".format(i + 1, nar))


if __name__ == "__main__":
    main()
