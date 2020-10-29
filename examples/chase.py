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
from opencog.type_constructors import *
from opencog.spacetime import *
from opencog.pln import *
from opencog.scheme_wrapper import scheme_eval, scheme_eval_h
from opencog.logger import Logger, log

# OpenAI Gym
import gym
from envs.gym_chase.chase_env import ChaseEnv
env = gym.make('Chase-v0')
# Uncomment the following to get a description of env
# help(env.unwrapped)

# OpenCog Gym
from agent.gymagent import GymAgent

##################
# CartPole Agent #
##################

class ChaseAgent(GymAgent):
    def __init__(self):
        GymAgent.__init__(self, env)
        # Load PLN
        scheme_eval(self.atomspace, "(use-modules (opencog pln))")
        scheme_eval(self.atomspace, "(pln-load 'empty)")
        log.info("PLN atomspace 1")
        scheme_eval(self.atomspace, "(pln-log-atomspace)")
        scheme_eval(self.atomspace, "(pln-load-rules \"temporal/predictive-implication-scope-direct-introduction\")")
        log.info("PLN atomspace 2")
        scheme_eval(self.atomspace, "(pln-log-atomspace)")
        scheme_eval(self.atomspace, "(pln-add-rule 'predictive-implication-scope-direct-introduction)")
        log.info("PLN atomspace 3")
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

    def plan(self, goal, expiry):
        """Plan the next actions given a goal and its expiry time offset

        Return a python list of cognivite schematics.  Whole cognitive
        schematics are output (instead of action plans) in order to
        make a decision based on their truth values.  Alternatively it
        could return a pair (action plan, tv), where tv has been
        evaluated to take into account the truth value of the context
        as well (which would differ from the truth value of rule in
        case the context is uncertain).

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

        # For now we provide a hardwired rule, which is if the pellet
        # is on the same square as the agent, then eating brings a
        # reward of 1 at the next iteration.
        #
        # This can be formalized by the following rule:
        #
        # PredictiveImplicationScope <high TV>
        #   TypedVariable
        #     Variable "$position"
        #     Type "ConceptNode"
        #   Time "1"
        #   And
        #     Evaluation
        #       Predicate "Agent Position"
        #       Variable "$position"
        #     Evaluation
        #       Predicate "Pellet Position"
        #       Variable "$position"
        #     Execution
        #       Schema "Eat"
        #   Evaluation
        #     Predicate "Reward"
        #     Number "1"

        agent_position = PredicateNode("Agent Position")
        pellet_position = PredicateNode("Pellet Position")
        position = VariableNode("$position")
        concept_t = TypeNode("ConceptNode")
        time_offset = NumberNode(str(expiry))
        eat = SchemaNode("Eat")
        reward = PredicateNode("Reward")
        unit = NumberNode("1")
        vhTV = TruthValue(1.0, 0.1)  # Very high TV

        eat_cogscm = \
            PredictiveImplicationScopeLink(
                TypedVariableLink(position, concept_t),
                time_offset,
                AndLink(
                    # Context
                    EvaluationLink(agent_position, position),
                    EvaluationLink(pellet_position, position),
                    # Action
                    ExecutionLink(eat)),
                # Goal
                EvaluationLink(reward, unit),
                # TV
                tv=vhTV)

        return [eat_cogscm]


########
# Main #
########
def main():
    ca = ChaseAgent()
    while (ca.step() or True):
        time.sleep(0.1)
        print("step_count = {}".format(ca.step_count))


if __name__ == "__main__":
    main()
