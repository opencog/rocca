# Class with an agent to interact with an environment

##############
# Initialize #
##############

# Python
import os
import math
from collections import Counter

# OpenCog
from opencog.pln import *
from opencog.utilities import is_closed
from opencog.scheme import scheme_eval_h, scheme_eval
from opencog.ure import ure_logger

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
        self.cognitive_schematics = set()
        self.load_opencog_modules()
        self.reset_action_counter()

        # Parameters controlling learning and decision

        # Expiry time to fulfill the goal. The system will not plan
        # beyond expiry.
        self.expiry = 2

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
        self.compressiveness = 0.75

        # Add an unknown component for each action. For now its weight
        # is constant, delta, but ultimately is should be calculated
        # as a rest in the Solomonoff mixture.
        self.delta = 1.0e-5


    def __del__(self):
        self.env.close()

    def load_opencog_modules(self):
        # Load miner
        scheme_eval(self.atomspace, "(use-modules (opencog miner))")
        scheme_eval(self.atomspace, "(miner-logger-set-level! \"fine\")")
        # scheme_eval(self.atomspace, "(miner-logger-set-sync! #t)")

        # Load PLN
        scheme_eval(self.atomspace, "(use-modules (opencog pln))")
        # scheme_eval(self.atomspace, "(pln-load-rule 'predictive-implication-scope-direct-introduction)")
        scheme_eval(self.atomspace, "(pln-load-rule 'predictive-implication-scope-direct-evaluation)")
        # No need of predictive implication for now
        # scheme_eval(self.atomspace, "(pln-load-rule 'predictive-implication-direct-evaluation)")
        scheme_eval(self.atomspace, "(pln-log-atomspace)")

    def reset_action_counter(self):
        self.action_counter = Counter({action: 0 for action in self.action_space})

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

    def pln_bc(self, query, maxiter):
        """Call PLN backward chainer with the given query and parameters.

        Return a python list of solutions.

        """

        agent_log.fine("pln_bc(query={}, maxiter={})".format(query, maxiter))

        command = "(pln-bc "
        command += str(query)
        command += " #:maximum-iterations " + str(maxiter)
        command += ")"
        return scheme_eval_h(self.atomspace, command).out

    def learn(self):
        """Discover patterns in the world and in the self.

        """

        # For now we only learn cognitive schematics

        # All resulting cognitive schematics
        cogscms = set()

        # For each action, mine its relationship to the goal,
        # positively and negatively, as well as more general
        # succedents.
        for action in self.action_space:
            lag = 1
            prectxs = [EvaluationLink(VariableNode("$P"), VariableNode("$X")),
                       EvaluationLink(VariableNode("$Q"), VariableNode("$Y")),
                       action]

            # Mine positive succedent goals
            postctxs = [self.positive_goal]
            pos_srps = self.mine_temporal_patterns((lag, prectxs, postctxs))
            pos_prdi = self.surprises_to_predictive_implications(pos_srps)
            agent_log.fine("pos_prdi = {}".format(pos_prdi))
            cogscms.update(set(pos_prdi))

            # Mine negative succedent goals
            postctxs = [self.negative_goal]
            neg_srps = self.mine_temporal_patterns((lag, prectxs, postctxs))
            neg_prdi = self.surprises_to_predictive_implications(neg_srps)
            agent_log.fine("neg_prdi = {}".format(neg_prdi))
            cogscms.update(set(neg_prdi))

            # Mine general succedents (only one for now)
            postctxs = [EvaluationLink(VariableNode("$R"), VariableNode("$Z"))]
            gen_srps = self.mine_temporal_patterns((lag, prectxs, postctxs))
            gen_prdi = self.surprises_to_predictive_implications(gen_srps)
            agent_log.fine("gen_prdi = {}".format(gen_prdi))
            cogscms.update(set(gen_prdi))

            # Mine positive succedent goals with multi-actions
            postctxs = [self.positive_goal]
            for snd_action in self.action_space:
                agent_log.fine("multiaction mining snd_action = {}".format(snd_action))
                ma_prectxs = (lag, prectxs, [snd_action])
                pos_multi_srps = self.mine_temporal_patterns((lag, ma_prectxs, postctxs))
                agent_log.fine("pos_multi_srps = {}".format(pos_multi_srps))
                pos_multi_prdi = self.surprises_to_predictive_implications(pos_multi_srps)
                cogscms.update(set(pos_multi_prdi))

        agent_log.fine("cogscms = {}".format(cogscms))
        self.cognitive_schematics.update(cogscms)

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

    def get_pattern_timed_clauses(self, pattern):
        """Return all timestamped clauses of a pattern.

        """

        return pattern.out[1].out

    def get_pattern_antecedent_events(self, pattern):
        """Return the antecedent events of a temporal pattern.

        That is all propositions not taking place at the latest time.

        """

        tclauses = get_early_clauses(self.get_pattern_timed_clauses(pattern))
        return [get_event(tc) for tc in tclauses]

    def get_pattern_succedent_events(self, pattern):
        """Return the succedent events of a temporal pattern.

        That is the propositions taking place at the lastest time.

        """

        tclauses = get_latest_clauses(self.get_pattern_timed_clauses(pattern))
        return [get_event(tc) for tc in tclauses]

    def get_typed_variables(self, vardecl):
        """Get the list of possibly typed variables in vardecl.

        """

        vt = vardecl.type
        if is_a(vt, types.VariableList) or is_a(vt, types.VariableSet):
            return vardecl.out
        else:
            return [vardecl]

    def get_nt_vardecl(self, pattern):
        """Get the vardecl of pattern excluding the time variable.

        """

        vardecl = get_vardecl(pattern)
        tvars = self.get_typed_variables(vardecl)
        nt_tvars = [tvar for tvar in tvars if not self.is_temporally_typed(tvar)]
        return VariableSet(*nt_tvars)

    def predictive_implication_scope_query(self, goal, expiry):
        """Build a PredictiveImplicationScope query for PLN.

        """

        vardecl = VariableNode("$vardecl")
        antecedent = VariableNode("$antecedent")
        # TODO: fix python PredictiveImplicationScopeLink binding!
        # query = QuoteLink(PredictiveImplicationScopeLink(UnquoteLink(vardecl),
        #                                                  to_nat(expiry),
        #                                                  UnquoteLink(antecedent),
        #                                                  goal))
        query = QuoteLink(scheme_eval_h(self.atomspace, "(PredictiveImplicationScopeLink " + str(UnquoteLink(vardecl)) + str(to_nat(expiry)) + str(UnquoteLink(antecedent)) + str(goal) + ")"))
        return query

    def predictive_implication_query(self, goal, expiry):
        """Build a PredictiveImplication query for PLN.

        """

        antecedent = VariableNode("$antecedent")
        query = PredictiveImplicationLink(to_nat(expiry), antecedent, goal)
        return query

    def to_sequential_and(self, timed_clauses):
        times = get_times(timed_clauses)
        if len(times) == 2:
            # Partition events in first and second time. For now lags
            # are assumed to be 1.
            early_clauses = get_early_clauses(timed_clauses)
            early_events = get_events(early_clauses)
            latest_clauses = get_latest_clauses(timed_clauses)
            latest_events = get_events(latest_clauses)
            return AltSequentialAndLink(to_nat(1),
                                        maybe_and(early_events),
                                        maybe_and(latest_events))
        else:
            agent_log.error("Not supported yet!")

    def to_predictive_implicant(self, pattern):
        """Turn a temporal pattern into predictive implicant.

        """

        agent_log.fine("to_predictive_implicant(pattern={})".format(pattern))

        timed_clauses = self.get_pattern_timed_clauses(pattern)
        early_clauses = get_early_clauses(timed_clauses)
        early_times = get_times(early_clauses)
        agent_log.fine("timed_clauses = {})".format(timed_clauses))
        agent_log.fine("early_clauses = {})".format(early_clauses))
        agent_log.fine("early_times = {})".format(early_times))
        if len(early_times) == 1:     # No need of SequentialAnd
            return maybe_and(get_events(early_clauses))
        else:
            return self.to_sequential_and(early_clauses)

    def to_predictive_implicand(self, pattern):
        """Turn a temporal pattern into predictive implicand.

        """

        return maybe_and(self.get_pattern_succedent_events(pattern))

    def to_predictive_implication_scope(self, pattern):
        """Turn a given pattern into a predictive implication scope with its TV.

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

        then the resulting predictive implication scope is

        PredictiveImplicationScope
          VariableList
          S Z
          Execution
            Schema "Eat"
          Evaluation
            Predicate "Reward"
            Number 1

        Note the empty variable declaration.

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
          <antecedent>
          Evaluation
            Predicate "Reward"
            Number 1

        that is the negative goal is automatically converted into a
        positive goal with low strength on the predictive implication.

        """

        agent_log.fine("to_predictive_implication_scope(pattern={})".format(pattern))

        # Get the predictive implication implicant and implicand
        # respectively
        pt = self.to_predictive_implicant(pattern)
        pd = self.to_predictive_implicand(pattern)

        agent_log.fine("pt = {}".format(pt))
        agent_log.fine("pd = {}".format(pd))

        # HACK: big hack, pd is turned into positive goal
        if pd == self.negative_goal:
            pd = self.positive_goal

        # Get lag, for now set to 1
        lag = SLink(ZLink())

        ntvardecl = self.get_nt_vardecl(pattern)
        # TODO: fix python PredictiveImplicationScopeLink binding!
        # preimp = PredictiveImplicationScopeLink(ntvardecl, lag, pt, pd)
        preimp = scheme_eval_h(self.atomspace, "(PredictiveImplicationScopeLink " + str(ntvardecl) + str(lag) + str(pt) + str(pd) + ")")
        # Make sure all variables are in the antecedent
        vardecl_vars = set(get_free_variables(ntvardecl))
        pt_vars = set(get_free_variables(pt))
        agent_log.fine("vardecl_vars = {}".format(vardecl_vars))
        agent_log.fine("pt_vars = {}".format(pt_vars))
        if vardecl_vars != pt_vars:
            return None

        agent_log.fine("preimp = {}".format(preimp))
        # Calculate the truth value of the predictive implication
        mi = 2
        return self.pln_bc(preimp, mi)[0]

    def is_desirable(self, cogscm):
        """Return True iff the cognitive schematic is desirable.

        For now to be desirable a cognitive schematic must have

        0. a well define atom
        1. its confidence above zero
        2. its action fully grounded
        3. all its variables in the antecedent

        """

        return cogscm \
            and has_non_null_confidence(cogscm) \
            and is_closed(get_t0_execution(cogscm)) \
            and has_all_variables_in_antecedent(cogscm)

    def surprises_to_predictive_implications(self, srps):
        """Like to_predictive_implication but takes surprises.

        """

        agent_log.fine("surprises_to_predictive_implications(srps={})".format(srps))

        # Turn patterns into predictive implication scopes
        cogscms = [self.to_predictive_implication_scope(self.get_pattern(srp))
                   for srp in srps]

        # Remove undesirable cognitive schematics
        #
        # TODO: its still in the atomspace, maybe should be move to
        # its own atomspace or labelled as such.
        cogscms = [cogscm for cogscm in cogscms if self.is_desirable(cogscm)]

        return cogscms

    def to_timed_clauses(self, lagged_antecedents_succedents, T):
        """Turn nested lagged, antecedents, succedents to AtTime clauses.

        For instance the input is

        (lag-2, (lag-1, X, Y), Z)

        then it is transformed into the following clauses

        [AtTime
           X
           T,
         AtTime
           Y
           T + lag-1,
         AtTime
           Z
           T + lag-1 + lag-2]

        Note that it will also output the max lag of such clauses,
        which in this example would be lag-1 + lag-2.  This is
        required by the caller in case such function is called
        recursively.

        Also, for now, and maybe ever, having nested succedents is not
        supported as in that case lags are not additive.

        """

        lag = lagged_antecedents_succedents[0]
        antecedents = lagged_antecedents_succedents[1]
        succedents = lagged_antecedents_succedents[2]

        if type(antecedents) is tuple:
            timed_clauses, reclag = self.to_timed_clauses(antecedents, T)
            lag += reclag
        else:
            timed_clauses = [AtTimeLink(ante, T) for ante in antecedents]

        lagnat = lag_to_nat(lag, T)
        timed_clauses += [AtTimeLink(succ, lagnat) for succ in succedents]
        return timed_clauses, lag

    def mine_temporal_patterns(self, lagged_antecedents_succedents, vardecl=None):
        """Given nested lagged, antecedents, succedents, mine temporal patterns.

        More precisely it takes a list of triples

        (lag, antecedents, succedents)

        where antecedents and succedents can themselves be triples of
        lagged antecedents and succedents.

        For instance the input is

        (lag-2, (lag-1, X, Y), Z)

        then it is transformed into the following initial pattern

        Present
          AtTime
            X
            T
          AtTime
            Y
            T + lag-1
          AtTime
            Z
            T + lag-1 + lag-2

        which the miner can start from.

        If no vardecl is provided then it is assumed to be composed of
        all free variables in all antecedents and succedents.

        """

        agent_log.fine("mine_temporal_patterns(lagged_antecedents_succedents={}, vardecl={})".format(lagged_antecedents_succedents, vardecl))

        # Set miner parameters
        minsup = 4
        maxiter = 1000
        cnjexp = "#f"
        enfspe = "#t"
        mspc = 6
        maxvars = 8
        maxcjnts = 6
        surprise = "'nisurp"
        T = VariableNode("$T")
        ignore = SetLink(T)

        # Define initial pattern
        # TODO: support any lag and vardecl
        timed_clauses, _ = self.to_timed_clauses(lagged_antecedents_succedents, T)
        agent_log.fine("timed_clauses = {}".format(timed_clauses))
        if not vardecl:
            variables = set([T])
            variables.update(get_free_variables_of_atoms(timed_clauses))
            vardecl = VariableSet(*variables)
        initpat = LambdaLink(vardecl, PresentLink(*timed_clauses))

        # Launch pattern miner
            # " #:ignore " + str(ignore) + \
        mine_query = "(cog-mine " + str(self.percepta_record) + \
            " #:minimum-support " + str(minsup) + \
            " #:initial-pattern " + str(initpat) + \
            " #:maximum-iterations " + str(maxiter) + \
            " #:conjunction-expansion " + cnjexp + \
            " #:enforce-specialization " + enfspe + \
            " #:maximum-variables " + str(maxvars) + \
            " #:maximum-conjuncts " + str(maxcjnts) + \
            " #:maximum-spcial-conjuncts " + str(mspc) + \
            " #:surprisingness " + surprise + ")"
        agent_log.fine("mine_query = {}".format(mine_query))
        surprises = scheme_eval_h(self.atomspace, "(List " + mine_query + ")")
        agent_log.fine("surprises = {}".format(surprises))

        return surprises.out

    def plan(self, goal, expiry):
        """Plan the next actions given a goal and its expiry time offset

        Return a python list of cognivite schematics meeting the
        expiry constrain.  Whole cognitive schematics are output in
        order to make a decision based on their truth values and
        priors.

        The format for a cognitive schematic is as follows

        PredictiveImplicationScope <tv>
          <vardecl>
          <lag-n>
          SequentialAnd [optional]
            <lag-n-1>
            ...
              SequentialAnd
                <lag-1>
                And
                  <context>
                  <execution-1>
            <execution-n>
          <goal>

        For now it is assumed that <execution-1> is fully grounded.

        The cognitive schematics meets the temporal constrain if the
        total lag is lower or equal to the expiry.

        """

        agent_log.fine("plan(goal={}, expiry={})".format(goal, expiry))
        agent_log.fine("self.cognitive_schematics = {}".format(self.cognitive_schematics))

        # Retrieve all cognitive schematics meeting the constrains which are
        #
        # 1. The total lag of the cognitive schematics is below or
        #    equal to the expiry.
        #
        # 2. The succedent matches the goal
        meet = lambda cogscm : get_total_lag(cogscm) <= expiry \
            and get_succedent(cogscm) == goal
        return [cogscm for cogscm in self.cognitive_schematics if meet(cogscm)]

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

        agent_log.fine("deduce(cogscms={})".format(cogscms))

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
        agent_log.fine("valid_cogscms = {}".format(valid_cogscms))

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
        mxmdl = omdict([(get_t0_execution(cogscm), (self.weight(cogscm), cogscm))
                        for cogscm in valid_cogscms])
        # Add delta (unknown) components
        for action in self.action_space:
            mxmdl.add(action, (self.delta, None))

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

        agent_log.debug("atomese_obs = {}".format(self.observation))
        obs_record = [self.record(o, self.step_count, tv=TRUE_TV)
                      for o in self.observation]
        agent_log.debug("obs_record = {}".format(obs_record))

        # Make the goal for that iteration
        goal = self.make_goal()
        agent_log.debug("goal = {}".format(goal))

        # Plan, i.e. come up with cognitive schematics as plans.  Here the
        # goal expiry is 2, i.e. must be fulfilled set for the next two iterations.
        cogscms = self.plan(goal, self.expiry)
        agent_log.debug("cogscms = {}".format(cogscms))

        # Deduce the action distribution
        mxmdl = self.deduce(cogscms)
        agent_log.debug("mxmdl = {}".format(mxmdl_to_str(mxmdl)))

        # Select the next action
        action, pblty = self.decide(mxmdl)
        agent_log.debug("action with probability of success = {}".format(act_pblt_to_str((action, pblty))))

        # Timestamp the action that is about to be executed
        action_record = self.record(action, self.step_count, tv=TRUE_TV)
        agent_log.debug("action_record = {}".format(action_record))
        agent_log.debug("action = {}".format(action))

        # Increment the counter for that action and log it
        self.action_counter[action] += 1
        agent_log.debug("action_counter = {}".format(self.action_counter))

        # Increase the step count and run the next step of the environment
        self.step_count += 1
        # TODO gather environment info.
        reward, self.observation, done = self.env.step(action)
        self.accumulated_reward += int(reward.out[1].name)
        agent_log.debug("observation = {}".format(self.observation))
        agent_log.debug("reward = {}".format(reward))
        agent_log.debug("accumulated reward = {}".format(self.accumulated_reward))

        reward_record = self.record(reward, self.step_count, tv=TRUE_TV)
        agent_log.debug("reward_record = {}".format(reward_record))

        if done:
            return False

        return True
