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

    # TODO: move to gymagent
    def learn(self):
        """Discover patterns in the world and in the self.

        """

        log.debug("learn()")

        # For now we only learn cognitive schematics

        # All resulting cognitive schematics
        cogscms = set()

        # For each action, mine there relationship to the goal,
        # positively and negatively.
        for action in self.atomese_action_space():
            goal = self.positive_goal()
            pos_srps = self.mine_action_patterns(action, goal, 1)
            cogscms.union(set(self.surprises_to_predictive_implications(pos_srps)))

            # TODO: For now the negative goal is hardwired
            neg_goal = self.negative_goal()
            neg_srps = self.mine_action_patterns(action, neg_goal, 1)
            cogscms.union(set(self.surprises_to_predictive_implications(neg_srps)))

        log.debug("cogscms = {}".format(cogscms))

    def plan_pln_xp(self, goal, expiry):
        # For now query existing PredictiveImplicationScope and update
        # their TVs based on evidence.

        mi = 1
        query = self.predictive_implication_scope_query(goal, expiry)
        command = "(pln-bc " + str(query) + " #:maximum-iterations " + str(mi) + ")"
        results = scheme_eval_h(self.atomspace, command)
        return results.out

    def get_pattern(self, surprise_eval):
        """Extract the pattern wrapped in a surprisingness evaluation.

        That is given

        Evaluation
          <surprisingness-measure>
          List
            <pattern>
            <db>

        return <pattern>.

        """

        return surprise_eval.out[1].out[0]

    def is_T(self, var):
        """Return True iff the variable is (Variable "$T").

        """

        return var == VariableNode("$T")

    def is_temporally_typed(self, tvar):
        """"Return True iff the variable is typed as temporal.

        For now a variable is typed as temporal if it is

        (Variable "$T")

        since variabled are not typed for now.

        """

        return self.is_T(tvar)

    def is_attime_T(self, clause):

        """Return True iff the clause is about an event occuring at time T.

        That is if it is of the form

            AtTime
              <event>
              Variable "$T"

        """

        return self.is_T(clause.out[1])

    def get_event(self, clause):
        """Return the event in a clause that is a timestamped event

        For instance if clause is

        AtTime
          <event>
          <time>

        return <event>

        """

        return clause.out[0]

    def get_pattern_antecedents(self, pattern):
        """Return the antecedent events of a temporal pattern.

        That is all propositions taking place at time T.

        """

        clauses = pattern.out[1].out
        return [self.get_event(clause) for clause in clauses
                if self.is_attime_T(clause)]

    def get_pattern_succedents(self, pattern):
        """Return the succedent events of a temporal pattern.

        That is the propositions taking place at time T+1.

        """

        clauses = pattern.out[1].out
        return [self.get_event(clause) for clause in clauses
                if not self.is_attime_T(clause)]

    def get_typed_variables(self, vardecl):
        """Get the list of possibly typed variables in vardecl.

        """

        vt = vardecl.type
        if is_a(vt, get_type("VariableList")) or is_a(vt, get_type("VariableSet")):
            return vardecl.out
        else:
            return [vardecl]

    def get_nt_vardecl(self, pattern):
        """Get the vardecl of pattern excluding the time variable.

        """

        vardecl = get_vardecl(pattern)
        tvars = self.get_typed_variables(vardecl)
        nt_tvars = [tvar for tvar in tvars if not self.is_temporally_typed(tvar)]
        return VariableList(*nt_tvars)

    def maybe_and(self, clauses):
        """Wrap an And if multiple clauses, otherwise return the only one.

        """

        return AndLink(*clauses) if 1 < len(clauses) else clauses[0]

    def predictive_implication_scope_query(self, goal, expiry):
        """Build a PredictiveImplicationScope query for PLN.

        """

        vardecl = VariableNode("$vardecl")
        antecedent = VariableNode("$antecedent")
        succedent = VariableNode("$succedent")
        query = QuoteLink(PredictiveImplicationScopeLink(UnquoteLink(vardecl),
                                                         to_nat(expiry),
                                                         UnquoteLink(antecedent),
                                                         UnquoteLink(succedent)))
        return query

    def to_predictive_implication(self, pattern):
        """Turn a given pattern into a predictive implication with its TV.

        If the pattern has a variable in addition to time, then it is
        turned into a predictive implication scope.

        For instance if the pattern is

        Lambda
          Variable "$T"
          Present
            AtTime
              Execution
                Schema "Eat"
              Variable "$T"
            AtTime
              Evaluation
                Predicate "Reward"
                Number 1
              S
                Variable "$T"

        then the resulting predictive implication is

        PredictiveImplication
          S Z
          Execution
            Schema "Eat"
          Evaluation
            Predicate "Reward"
            Number 1

        However if the pattern is

        Lambda
          VariableList
            Variable "$T"
            Variable "$A"
          Present
            AtTime
              Evaluation
                Predicate "Eatable"
                Variable "$X"
              Variable "$T"
            AtTime
              Execution
                Schema "Eat"
              Variable "$T"
            AtTime
              Evaluation
                Predicate "Reward"
                Number 1
              S
                Variable "$T"

        then the resulting predictive implication scope is

        PredictiveImplicationScope
          Variable "$X"
          S Z
          And
            Evaluation
              Predicate "Eatable"
              Variable "$X"
            Execution
              Schema "Eat"
          Evaluation
            Predicate "Reward"
            Number 1

        TODO: for now if the succedent is

          Evaluation
            Predicate "Reward"
            Number 0

        then the resulting predictive implication (scope) is

        PredictiveImplication <1 - s, c>
          <antecedant>
          Evaluation
            Predicate "Reward"
            Number 1

        that is the negative goal is automatically converted into a
        positive goal with low strength on the predictive implication.

        """

        log.debug("to_predictive_implication(pattern={})".format(pattern))

        # Get the predictive implication implicant and implicand
        # respecively
        pt = self.maybe_and(self.get_pattern_antecedents(pattern))
        pd = self.maybe_and(self.get_pattern_succedents(pattern))

        # TODO: big hack, pd is turned into positive goal
        if pd == self.negative_goal():
            pd = self.positive_goal()

        # Get lag, for now set to 1
        lag = SLink(ZLink())

        # If there is only a time variable return a predictive
        # implication
        if self.is_T(get_vardecl(pattern)):
            pis = PredictiveImplicationLink(lag, pt, pd)
        # Otherwise there are multiple variables, return a predictive
        # implication scope
        else:
            ntvardecl = self.get_nt_vardecl(pattern)
            pis = PredictiveImplicationScopeLink(ntvardecl, lag, pt, pd)

        # Calculate the truth value of the predictive implication
        mi = 2
        command = "(pln-bc " + str(pis) + " #:maximum-iterations " + str(mi) + ")"
        results = scheme_eval_h(self.atomspace, command)
        return results.out[0]

    def is_desirable(self, cogscm):
        """Return True iff the cognitive schematic is desirable.

        For now to be desirable a cognitive schematic must

        1. have its confidence above zero
        2. have its action fully grounded

        """

        log.debug("is_desirable(cogscm={})".format(cogscm))

        # ic =  is_closed(get_action(cogscm))

        # log.debug("ic = {})".format(ic))

        # return ic

        return is_scope(cogscm) and is_closed(get_action(cogscm))

    def surprises_to_predictive_implications(self, srps):
        """Like to_predictive_implication but takes surprises.

        """

        log.debug("surprises_to_predictive_implications(srps={})".format(srps))

        # Turn patterns into predictive implications
        cogscms = [self.to_predictive_implication(self.get_pattern(srp))
                   for srp in srps]
        log.debug("cogscms-1 = {}".format(cogscms))

        # Remove undesirable cognitive schematics
        cogscms = [cogscm for cogscm in cogscms if self.is_desirable(cogscm)]
        log.debug("cogscms-2 = {}".format(cogscms))

        return cogscms


    def mine_action_patterns(self, action, postctx, lag):
        """Given an action, a post-context and its lag, mine patterns.

        That is mine patterns relating pre-context, action and
        post-context, of the form

        Present
          AtTime
            X1
            T
          ...
          AtTime
            Xn
            T
          AtTime
            Execution
              <action>
            T
          AtTime
            <postctx>
            T + <lag>

        """

        log.debug("mine_action_patterns(action={}, postctx={}, lag={})".format(action, postctx, lag))

        # Set miner parameters
        scheme_eval(self.atomspace, "(define minsup 10)")
        scheme_eval(self.atomspace, "(define maxiter 1000)")
        scheme_eval(self.atomspace, "(define cnjexp #f)")
        scheme_eval(self.atomspace, "(define enfspe #t)")
        scheme_eval(self.atomspace, "(define mspc 4)")
        scheme_eval(self.atomspace, "(define maxvars 10)")
        scheme_eval(self.atomspace, "(define maxcjnts 4)")
        scheme_eval(self.atomspace, "(define surprise 'nisurp)")

        # Define initial pattern
        # NEXT: work for more than lag of 1
        scheme_eval(self.atomspace,
                    "(define initpat"
                    "  (Lambda"
                    "    (VariableSet"
                    "      (Variable \"$T\")"
                    "      (Variable \"$P\")"
                    "      (Variable \"$X\")"
                    "      (Variable \"$Q\")"
                    "      (Variable \"$Y\"))"
                    "    (Present"
                    "      (AtTime"
                    "        (Evaluation (Variable \"$P\") (Variable \"$X\"))"
                    "        (Variable \"$T\"))"
                    "      (AtTime"
                    "        (Evaluation (Variable \"$Q\") (Variable \"$Y\"))"
                    "        (Variable \"$T\"))"
                    "      (AtTime"
                    "        (Execution " + str(action) + ")"
                    "        (Variable \"$T\"))"
                    "      (AtTime"
                    "        " + str(postctx) +
                    "        (S (Variable \"$T\"))))))")

        # Launch pattern miner
        surprises = scheme_eval_h(self.atomspace,
                                  "(List"
                                  "  (cog-mine " + str(self.percepta_record) +
                                  "            #:minimum-support minsup"
                                  "            #:initial-pattern initpat"
                                  "            #:maximum-iterations maxiter"
                                  "            #:conjunction-expansion cnjexp"
                                  "            #:maximum-variables maxvars"
                                  "            #:maximum-conjuncts maxcjnts"
                                  "            #:maximum-spcial-conjuncts mspc"
                                  "            #:surprisingness surprise))")
        log.debug("surprises = {}".format(surprises))

        return surprises.out

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
    lt_period = 300             # Duration of a learning-training iteration
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
