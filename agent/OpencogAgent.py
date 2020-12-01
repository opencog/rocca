# Class with an agent to interact with an environment

##############
# Initialize #
##############

# Python
import os
import math

# OpenCog
from opencog.pln import *
from opencog.utilities import is_closed
from opencog.scheme import scheme_eval_h, scheme_eval

# OpencogAgent
from .utils import *


#########
# Class #
#########

class OpencogAgent:
    def __init__(self, env, action_space, p_goal, n_goal):
        self.atomspace = AtomSpace()
        set_default_atomspace(self.atomspace)
        self.env = env
        _, self.observation, _ = self.env.restart()
        self.step_count = 0
        self.accumulated_reward = 0
        self.percepta_record = ConceptNode("Percepta Record")
        self.action_space = action_space
        self.positive_goal = p_goal
        self.negative_goal = n_goal
        self.load_opencog_modules()

        # Parameters controlling learning and decision

        # Prior alpha and beta of beta-distributions corresponding to
        # truth values
        self.prior_a = 1.0
        self.prior_b = 1.0

        # Parameter to control the complexity penalty over the
        # cognitive schematics. Ranges from 0, no penalty to +inf,
        # infinit penalty. Affect the calculation of the cognitive
        # schematic prior.
        self.cpx_penalty = 1.0

        # Parameter to estimate the length of a whole model given a
        # partial model + unexplained data. Ranges from 0 to 1, 0
        # being no compressiveness at all of the unexplained data, 1
        # being full compressiveness.
        self.compressiveness = 0.9

    def __del__(self):
        self.env.close()

    def load_opencog_modules(self):
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

    def record(self, atom, i, tv=None):
        """Timestamp and record an atom to the Percepta Record.

        That is add the following in the atomspace

        MemberLink (stv 1 1)
          AtTimeLink <tv>
            <atom>
            <i>
          <self.percepta_record>

        """

        return MemberLink(timestamp(atom, i, tv), self.percepta_record, tv=TRUE_TV)

    def make_goal(self):

        """Define the goal of the current iteration.

        By default the goal of the current iteration is to have a
        reward of 1.

        Evaluation
          Predicate "Reward"
          Number 1

        """

        return EvaluationLink(PredicateNode("Reward"), NumberNode(str(1)))

    def learn(self):
        """Discover patterns in the world and in the self.

        """

        log.debug("learn()")

        # For now we only learn cognitive schematics

        # All resulting cognitive schematics
        cogscms = set()

        # For each action, mine there relationship to the goal,
        # positively and negatively.
        for action in self.action_space:
            lag = 1
            prectxs = [EvaluationLink(VariableNode("$P"), VariableNode("$X")),
                       EvaluationLink(VariableNode("$Q"), VariableNode("$Y")),
                       ExecutionLink(action)]
            postctxs = [self.positive_goal]
            pos_srps = self.mine_temporal_patterns(lag, prectxs, postctxs)
            cogscms.union(set(self.surprises_to_predictive_implications(pos_srps)))

            postctxs = [self.negative_goal]
            neg_srps = self.mine_temporal_patterns(lag, prectxs, postctxs)
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
        if pd == self.negative_goal:
            pd = self.positive_goal

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

    def mine_temporal_patterns(self, lag=1, prectxs=[], postctxs=[], vardecl=None):
        """Given a lag, pre and post contexts, mine temporal patterns.

        That is mine patterns specializing the following

        Present
          AtTime
            <prectx-1>
            T
          ...
          AtTime
            <prectx-n>
            T
          AtTime
            <postctx-1>
            T + <lag>
          ...
          AtTime
            <postctx-m>
            T + <lag>

        where
          prectxs = [prectx-1, ..., prectx-n]
          postctxs = [postctx-1, ..., postctx-n]

        If no vardecl is provided then it is assumed to be composed of
        all free variables in prectxs and postctxs.

        """

        log.debug("mine_temporal_patterns(lag={}, prectx={}, postctx={})".format(lag, prectxs, postctxs))

        # Set miner parameters
        minsup = 10
        maxiter = 1000
        cnjexp = "#f"
        enfspe = "#t"
        mspc = 4
        maxvars = 10
        maxcjnts = 4
        surprise = "'nisurp"

        # Define initial pattern
        # TODO: support any lag and vardecl
        T = VariableNode("$T")
        timed_prectxs = [AtTimeLink(prectx, T) for prectx in prectxs]
        timed_postctxs = [AtTimeLink(postctx, SLink(T)) for postctx in postctxs]
        if not vardecl:
            vardecl = VariableSet(T, VariableNode("$P"), VariableNode("$X"), VariableNode("$Q"), VariableNode("$Y"))
        if not vardecl:
            initpat = LambdaLink(PresentLink(*timed_prectxs, *timed_postctxs))
        else:
            initpat = LambdaLink(vardecl, PresentLink(*timed_prectxs, *timed_postctxs))

        # Launch pattern miner
        surprises = scheme_eval_h(self.atomspace,
                                  "(List"
                                  "  (cog-mine " + str(self.percepta_record) +
                                  "            #:minimum-support " + str(minsup) +
                                  "            #:initial-pattern " + str(initpat) +
                                  "            #:maximum-iterations " + str(maxiter) +
                                  "            #:conjunction-expansion " + cnjexp +
                                  "            #:enforce-specialization " + enfspe +
                                  "            #:maximum-variables " + str(maxvars) +
                                  "            #:maximum-conjuncts " + str(maxcjnts) +
                                  "            #:maximum-spcial-conjuncts " + str(mspc) +
                                  "            #:surprisingness " + surprise + "))")
        log.debug("surprises = {}".format(surprises))

        return surprises.out

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

        For now it is assumed that <action> is fully grounded.

        """

        return self.plan_pln_xp(goal, expiry)

    # TODO: move to its own class (MixtureModel or something)
    def get_all_uniq_atoms(self, atom):
        """Return the set of all unique atoms in atom.

        """

        # Base cases
        if atom.is_node():
            return {atom}

        # Recursive cases
        if atom.is_link():
            results = {atom}
            for o in atom.out:
                results.union(self.get_all_uniq_atoms(o))
            return results

    # TODO: move to its own class (MixtureModel or something)
    def complexity(self, atom):
        """Return the count of all unique atoms in atom.

        """

        return len(self.get_all_uniq_atoms(atom))

    # TODO: move to its own class (MixtureModel or something)
    def prior(self, length):
        """Given the length of a model, calculate its prior.

        Specifically

        exp(-cpx_penalty*length)

        where cpx_penalty is a complexity penalty parameter (0 for no
        penalty, +inf for infinit penalty), and length is the size of
        the model, the total number of atoms involved in its
        definition.

        The prior doesn't have to sum up to 1 because the probability
        estimates are normalized.

        """

        return math.exp(-self.cpx_penalty * length)

    # TODO: move to its own class (MixtureModel or something)
    def kolmogorov_estimate(self, remain_count):
        """Given the size of the data set that isn't explained by a model,
        estimate the complexity of a model that would explain them
        perfectly. The heuristic used here is

        remain_data_size^(1 - compressiveness)

        If compressiveness is null, then no compression occurs, the
        model is the data set itself, if compressiveness equals 1,
        then it return 1, which is the maximum compression, all data
        can be explained with just one bit.

        """

        return pow(remain_count, 1.0 - self.compressiveness)

    # TODO: move to its own class (MixtureModel or something)
    def prior_estimate(self, cogscm):
        """Calculate the prior probability of cogscm.

        """

        partial_complexity = self.complexity(cogscm)
        remain_data_size = self.data_set_size - cogscm.tv.count
        kestimate = self.kolmogorov_estimate(remain_data_size);
        return self.prior(partial_complexity + kestimate);

    # TODO: move to its own class (MixtureModel or something)
    def beta_factor(self, cogscm):
        """Return the beta factor as described in Eq.26 of

        https://github.com/ngeiswei/papers/blob/master/PartialBetaOperatorInduction/PartialBetaOperatorInduction.pdf

        Note that we do account for the normalizing factor (numerator
        of the fraction of Eq.22) as its not clear we want to
        normalize the BMA (we might want to replace rest by an unknown
        distribution, i.e. a prior beta-distribution).

        """

        a = tv_to_alpha_param(cogscm.tv, self.prior_a, self.prior_b)
        b = tv_to_beta_param(cogscm.tv, self.prior_a, self.prior_b)
        return sp.beta(a, b) / sp.beta(self.prior_a, self.prior_b)

    # TODO: move to its own class (MixtureModel or something)
    def weight(self, cogscm):
        """Calculate the weight of a cogscm for Model Bayesian Averaging.

        The calculation is based on

        https://github.com/ngeiswei/papers/blob/master/PartialBetaOperatorInduction/PartialBetaOperatorInduction.pdf

        It assumes self.data_set_size has been properly set.

        """

        return self.prior_estimate(cogscm) * self.beta_factor(cogscm)

    # TODO: move to its own class (MixtureModel or something)
    def infer_data_set_size(self, cogscms):
        """Infer the data set size by taking the max count of all models

        (it works assuming that one of them is complete).

        """

        if 0 < len(cogscms):
            return max(cogscms, key=lambda x: x.tv.count).tv.count
        else:
            return 0

    def w8d_cogscm_to_str(self, w8d_cogscm, indent=""):
        """Pretty print the given list of weighted cogscm"""

        weight = w8d_cogscm[0]
        cogscm = w8d_cogscm[1]
        tv = get_cogscm_tv(cogscm)
        idstr = cogscm.id_string() if cogscm else "None"
        s = "(weight={}, tv={}, id={})".format(weight, tv, idstr)
        return s

    def w8d_cogscms_to_str(self, w8d_cogscms, indent=""):
        """Pretty print the given list of weighted cogscms"""

        w8d_cogscms_sorted = sorted(w8d_cogscms, key=lambda x: x[0], reverse=True)

        s = ""
        for w8d_cogscm in w8d_cogscms_sorted:
            s += indent + self.w8d_cogscm_to_str(w8d_cogscm, indent + "  ") + "\n"
        return s

    def mxmdl_to_str(self, mxmdl, indent=""):
        """Pretty print the given mixture model of cogscms"""

        s = ""
        for act_w8d_cogscms in mxmdl.listitems():
            s += "\n" + indent + str(act_w8d_cogscms[0]) + "\n"
            s += self.w8d_cogscms_to_str(act_w8d_cogscms[1], indent + "  ")
        return s

    def deduce(self, cogscms):
        """Return an action distribution given a list cognitive schematics.

        The action distribution is actually a second order
        distribution, i.e. each action is weighted with a truth value.
        Such truth value, called the action truth value, corresponds
        to the second order probability of acheiving the goal if the
        action is taken right now.

        Formally the meaning of the action truth value can be
        expressed as follows:

        Subset <action-tv>
          SubjectivePaths
            AtTime
              Execution
                <action>
              <i>
          SubjectivePaths
            AtTime
              <goal>
              <i + offset>

        where

        SubjectivePaths
          <P>

        is the set of all subjective paths compatible with <P>, and a
        subjective path is a sequence of sujective states (atomspace
        snapshots) indexed by time.

        In order to infer such action truth value one needs to perform
        deduction (or modus ponens) on the cognitive schematics,
        combining the probability of the context being currently true.

        """

        log.debug("deduce(cogscms={})".format(cogscms))

        # For each cognitive schematic estimate the probability of its
        # context to be true and multiply it by the truth value of the
        # cognitive schematic, then calculate its weight based on
        # https://github.com/ngeiswei/papers/blob/master/PartialBetaOperatorInduction/PartialBetaOperatorInduction.pdf
        # and use it to perform a Bayesian Model Averaging to obtain
        # the second order distribution of each action.
        #
        # Important Notes:

        # 1. Adding an unknown component (with flat or such prior) in
        #    the BMA can flatten the resulting distribution and be
        #    used to user-tune exploration vs exploitation in a
        #    justified manner. This is probably equivalent to (or
        #    better than) Epsilon-best. Ultimately the weight of such
        #    unknown component should be calculated as the unknown
        #    rest in the Solomonoff mixture.
        #
        # 2. We actually don't need to build the mixture model, rather
        #    we just need to hand the convex combination of models to
        #    the decide function which will do the Thompson sampling.
        #
        # 3. It's unclear if the probability of the context should
        #    altern the model TV or its weight. We need to think of
        #    generalizing the inactive models as well.

        # For now discretize context truth into valid and invalid and
        # only consider valid cognitive schematics in the BMA
        # (Bayesian Model Averaging). It's not entirely clear what to
        # do with the invalid cognitive schematics, maybe they should
        # be taken into account to lower the confidence of the final
        # result, as they allegedly exert an unknown influence (via
        # their invalid parts).
        ctx_tv = lambda cogscm: \
            get_context_actual_truth(self.atomspace, cogscm, self.step_count)
        valid_cogscms = [cogscm for cogscm in cogscms if 0.9 < ctx_tv(cogscm).mean]
        log.debug("valid_cogscms = {}".format(valid_cogscms))

        # Size of the complete data set, including all observations
        # used to build the models. For simplicity we're gonna assume
        # that it is the max of all counts over the models. Meaning
        # that to do well, at least one model has to be complete,
        # however bad this model might be.
        #
        # Needs to be set before calling self.weight
        self.data_set_size = self.infer_data_set_size(valid_cogscms)

        # For each action, map a list of weighted valid cognitive
        # schematics.
        mxmdl = omdict([(get_action(cogscm), (self.weight(cogscm), cogscm))
                        for cogscm in valid_cogscms])

        # Add an unknown component for each action. For now its weight
        # is constant, delta, but ultimately is should be calculated
        # as a rest in the Solomonoff mixture.
        delta = 1.0e-3
        for action in self.action_space:
            mxmdl.add(action, (delta, None))

        return mxmdl

    def decide(self, mxmdl):
        """Select the next action to enact from a mixture model of cogscms.

        The action is selected from the action distribution, a list of
        pairs (action, tv), obtained from deduce.  The selection uses
        Thompson sampling leveraging the second order distribution to
        balance exploitation and exploration. See
        http://auai.org/uai2016/proceedings/papers/20.pdf for more
        details about Thompson Sampling.

        """

        # Select the pair of action and its first order probability of
        # success according to Thompson sampling
        (action, pblty) = thompson_sample(mxmdl, self.prior_a, self.prior_b)

        # Return the action (we don't need the probability for now)
        return (action, pblty)

    def step(self):
        """Run one step of observation, decision and env update
        """

        log.debug("atomese_obs = {}".format(self.observation))
        obs_record = [self.record(o, self.step_count, tv=TRUE_TV)
                      for o in self.observation]
        log.debug("obs_record = {}".format(obs_record))

        # Make the goal for that iteration
        goal = self.make_goal()
        log.debug("goal = {}".format(goal))

        # Plan, i.e. come up with cognitive schematics as plans.  Here the
        # goal expiry is 1, i.e. set for the next iteration.
        expiry = 1
        css = self.plan(goal, expiry)
        log.debug("css = {}".format(css))

        # Deduce the action distribution
        mxmdl = self.deduce(css)
        log.debug("mxmdl = {}".format(self.mxmdl_to_str(mxmdl)))

        # Select the next action
        action, pblty = self.decide(mxmdl)
        log.debug("(action={}, pblty={})".format(action, pblty))

        # Timestamp the action that is about to be executed
        action_record = self.record(action, self.step_count, tv=TRUE_TV)
        log.debug("action_record = {}".format(action_record))
        log.debug("action = {}".format(action))

        # Increase the step count and run the next step of the environment
        self.step_count += 1
        # TODO gather environment info.
        reward, self.observation, done = self.env.step(action)
        self.accumulated_reward += int(reward.out[1].name)
        log.debug("observation = {}".format(self.observation))
        log.debug("reward = {}".format(reward))
        log.debug("accumulated reward = {}".format(self.accumulated_reward))

        reward_record = self.record(reward, self.step_count, tv=TRUE_TV)
        log.debug("reward_record = {}".format(reward_record))

        if done:
            return False

        return True
