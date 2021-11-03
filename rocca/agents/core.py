# Class with an agent to interact with an environment

##############
# Initialize #
##############

# Python
import logging
import math
import multiprocessing
from collections import Counter
from typing import Any

# OpenCog
from opencog.atomspace import Atom, AtomSpace
from opencog.spacetime import AtTimeLink, TimeNode
from opencog.pln import SLink, ZLink, BackPredictiveImplicationScopeLink, BackSequentialAndLink
from opencog.scheme import scheme_eval, scheme_eval_h
from opencog.utilities import is_closed, set_default_atomspace
from opencog.type_constructors import ConceptNode, AndLink, EvaluationLink, PredicateNode, NumberNode, VariableNode, SetLink, LambdaLink, QuoteLink, UnquoteLink, MemberLink

from rocca.envs.wrappers import Wrapper

# OpencogAgent
from .utils import *

logging.basicConfig(filename="agent.log", format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

#########
# Class #
#########


class OpencogAgent:
    def __init__(
        self,
        env: Wrapper,
        atomspace: AtomSpace,
        action_space,
        p_goal,
        n_goal,
        log_level="debug",
    ):
        # Construct the various atomspaces
        self.atomspace = atomspace # Working atomspace
        self.percepta_atomspace = AtomSpace()
        self.cogscms_atomspace = AtomSpace()
        self.working_atomspace = AtomSpace()
        set_default_atomspace(self.atomspace)

        self.env = env
        self.observation, _, _ = self.env.restart()
        self.step_count = 0
        self.accumulated_reward = 0
        self.percepta_record_cpt = ConceptNode("Percepta Record")
        # The percepta_record is a list of sets of timestamped
        # percepta.  The list is ordered by timestamp, that each
        # element of that list is a set of percepta at the timestamp
        # corresponding to its index.
        self.percepta_record: list = []
        self.action_space = action_space
        self.positive_goal = p_goal
        self.negative_goal = n_goal
        self.cognitive_schematics: set = set()
        self.log_level = log_level
        self.load_opencog_modules()
        self.reset_action_counter()

        # User parameters controlling learning and decision

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

        # Enable mono-action pattern mining with general succedent
        # (not just about goal).  This is important to gather the
        # knowledge in order to make temporal deduction useful.
        self.monoaction_general_succedent_mining = True

        # Enable poly-action pattern mining
        self.polyaction_mining = True

        # Enable temporal deduction, to string together polyaction
        # plans from monoaction plans.
        self.temporal_deduction = True

        # Only consider cognitive schematics with mean of 1 (and non
        # null confidence)
        self.true_cogscm = False

        # Only consider cognitive schematics without variable (other
        # than temporal of course)
        self.empty_vardecl_cogscm = False

    def __del__(self):
        self.env.close()

    def load_opencog_modules(self):
        # Load miner
        scheme_eval(self.atomspace, "(use-modules (opencog miner))")

        # Load PLN.  All rules must be pre-loaded here
        scheme_eval(self.atomspace, "(use-modules (opencog pln))")
        scheme_eval(self.atomspace, "(use-modules (opencog spacetime))")
        rules = [
            "back-predictive-implication-scope-direct-evaluation",
            "back-predictive-implication-scope-conditional-conjunction-introduction",
            "back-predictive-implication-scope-deduction-cogscm",
        ]
        self.pln_load_rules(rules)
        # scheme_eval(self.atomspace, "(pln-log-atomspace)")

    def reset_action_counter(self):
        self.action_counter = Counter({action: 0 for action in self.action_space})

    def insert_to_percepta_record(self, timed_atom: Atom, i: int):
        """Insert a timestamped atom into self.percepta_record.

        The percepta_record is a list of sets of timestamped percepta.
        The list is ordered by timestamp, that each element of that
        list is a set of percepta at the timestamp corresponding to
        its index.

        """

        while len(self.percepta_record) <= i:
            self.percepta_record.append(set())
        self.percepta_record[i].add(timed_atom)

    def record(self, atom: Atom, i: int, tv=None) -> Atom:
        """Timestamp and record an atom to the Percepta Record.

        That is add the following in the atomspace

        MemberLink (stv 1 1)
          AtTimeLink <tv>
            <atom>
            <i>
          <self.percepta_record_cpt>

        As well as the AtTimeLink to self.percepta_record (see
        insert_to_percepta_record method for more info).

        """

        timed_atom = timestamp(atom, i, tv)
        mbr = MemberLink(timed_atom, self.percepta_record_cpt, tv=TRUE_TV)
        self.insert_to_percepta_record(timed_atom, i)
        return self.percepta_atomspace.add_atom(mbr)

    def make_goal(self) -> Atom:
        """Define the goal of the current iteration.

        By default the goal of the current iteration is to have a
        reward of 1.

        Evaluation
          Predicate "Reward"
          Number 1

        """

        return EvaluationLink(PredicateNode("Reward"), NumberNode(str(1)))

    def pln_load_rules(self, rules: list[str]=[]):
        """Load PLN rules.

        Take a list of rule scheme symbols (but without the single
        quote for the symbol), such as

        ["back-predictive-implication-scope-direct-evaluation",
         "back-predictive-implication-scope-deduction-cogscm"]

        """

        for rule in rules:
            scheme_eval(self.atomspace, "(pln-load-rule '" + rule + ")")

    def pln_fc(self,
               atomspace: AtomSpace,
               source: Atom,
               vardecl=None,
               maximum_iterations: int=10,
               full_rule_application: bool=False,
               rules: list[str]=[]) -> set[Atom]:
        """Call PLN forward chainer with the given source and parameters.

        The parameters are

        atomspace: the atomspace over which to do the reasoning. # NEXT: find out if it really does that
        source: the atom source to start from.
        maximum_iterations: the maximum number of iterations.
        rules: optional list of rule symbols.  If empty keep current rule set.

        Return a python list of solutions.

        """

        agent_log.fine("pln_fc(atomspace={}, source={}, maximum_iterations={}, full_rule_application={})".format(
            atomspace,
            source,
            maximum_iterations,
            full_rule_application,
        ))

        # Add rules (should be previously loaded)
        if rules:
            scheme_eval(self.atomspace, "(pln-rm-all-rules)")
            for rule in rules:
                er = scheme_eval(self.atomspace, "(pln-add-rule '" + rule + ")")
                # agent_log.fine("(pln-add-rule '" + rule + ")")
                # agent_log.fine("er = " + str(er))

        # Generate and run query
        command = "(pln-fc "
        command += str(source)
        command += ("#:vardecl " + str(vardecl)) if vardecl else ""
        command += " #:maximum-iterations " + str(maximum_iterations)
        command += " #:fc-full-rule-application " + str(full_rule_application)
        command += ")"
        return set(scheme_eval_h(atomspace, command).out)

    def pln_bc(self,
               atomspace: AtomSpace,
               target: Atom,
               vardecl=None,
               maximum_iterations: int=10,
               rules: list[str]=[]) -> set[Atom]:
        """Call PLN backward chainer with the given target and parameters.

        The parameters are

        maximum_iterations: the maximum number of iterations.
        rules: optional list of rule symbols.  If empty keep current rule set.

        Return a python list of solutions.

        """

        agent_log.fine("pln_bc(atomspace={}, target={}, maximum_iterations={})".format(
            atomspace,
            target,
            maximum_iterations
        ))

        # Add rules (should be previously loaded)
        if rules:
            scheme_eval(self.atomspace, "(pln-rm-all-rules)")
            for rule in rules:
                er = scheme_eval(self.atomspace, "(pln-add-rule '" + rule + ")")
                # agent_log.fine("(pln-add-rule '" + rule + ")")
                # agent_log.fine("er = " + str(er))

        # Generate and run query
        command = "(pln-bc "
        command += str(target)
        command += ("#:vardecl " + str(vardecl)) if vardecl else ""
        command += " #:maximum-iterations " + str(maximum_iterations)
        command += ")"
        return set(scheme_eval_h(atomspace, command).out)

    def mine_cogscms(self) -> set[Atom]:
        """Discover cognitive schematics via pattern mining.

        Return the set of mined cognitive schematics.

        """

        # All resulting cognitive schematics
        cogscms = set()

        # For each action, mine its relationship to the goal,
        # positively and negatively, as well as more general
        # succedents.
        for action in self.action_space:
            lag = 1
            prectxs = [
                EvaluationLink(VariableNode("$P"), VariableNode("$X")),
                EvaluationLink(VariableNode("$Q"), VariableNode("$Y")),
                action,
            ]

            # Mine positive succedent goals
            postctxs = [self.positive_goal]
            las = (lag, prectxs, postctxs)
            # NEXT: use percepta_atomspace
            pos_srps = self.mine_temporal_patterns(self.atomspace, las)
            pos_prdi = self.surprises_to_predictive_implications(pos_srps)
            agent_log.fine("pos_prdi = {}".format(pos_prdi))
            cogscms.update(set(pos_prdi))

            # Mine negative succedent goals
            postctxs = [self.negative_goal]
            las = (lag, prectxs, postctxs)
            # NEXT: use percepta_atomspace
            neg_srps = self.mine_temporal_patterns(self.atomspace, las)
            neg_prdi = self.surprises_to_predictive_implications(neg_srps)
            agent_log.fine("neg_prdi = {}".format(neg_prdi))
            cogscms.update(set(neg_prdi))

            # Mine general succedents (only one for now)
            if self.monoaction_general_succedent_mining:
                postctxs = [EvaluationLink(VariableNode("$R"), VariableNode("$Z"))]
                las = (lag, prectxs, postctxs)
                # NEXT: use percepta_atomspace
                gen_srps = self.mine_temporal_patterns(self.atomspace, las)
                gen_prdi = self.surprises_to_predictive_implications(gen_srps)
                agent_log.fine("gen_prdi = {}".format(gen_prdi))
                cogscms.update(set(gen_prdi))

            # Mine positive succedent goals with poly-actions
            if self.polyaction_mining:
                postctxs = [self.positive_goal]
                for snd_action in self.action_space:
                    agent_log.fine(
                        "polyaction mining snd_action = {}".format(snd_action)
                    )
                    ma_prectxs = (lag, prectxs, [snd_action])
                    las = (lag, ma_prectxs, postctxs)
                    # NEXT: use percepta_atomspace
                    pos_multi_srps = self.mine_temporal_patterns(
                        self.atomspace, las
                    )
                    agent_log.fine("pos_multi_srps = {}".format(pos_multi_srps))
                    pos_multi_prdi = self.surprises_to_predictive_implications(
                        pos_multi_srps
                    )
                    cogscms.update(set(pos_multi_prdi))

        agent_log.fine("Mined cognitive schematics = {}".format(cogscms))
        return cogscms

    def directly_evaluate_conjunction(self, conjuncts: set[Atom]) -> Atom:
        """Directly evaluate the TV of a conjunction of atoms.

        conjunctions is a set of atoms and it returns a conjunction of
        atoms with its properly updated.

        """

        agent_log.fine("directly_evaluate_conjunction(conjuncts={})".format(conjuncts))

        pos_count = 0
        for timed_events in self.percepta_record:
            agent_log.fine("timed_events = {}".format(timed_events))
            events = set(get_events(timed_events))
            if conjuncts <= events:
                pos_count += 1
            # NEXT: check timestamped TV
            # and has_one_mean(timed_percept) \
            # and has_non_null_confidence(timed_percept):

        mean = float(pos_count) / float(self.step_count)
        count = self.step_count
        # NEXT: make sure the atomspace is correct
        # return AndLink(conjuncts).truth_value(mean, count)
        atom = conjuncts.pop()
        atom.truth_value(mean, count)
        return atom

    def directly_evaluate_cogscms_ante_succ(self, atomspace: AtomSpace):
        """Directly evaluate the TVs of all cogscms outgoings."""

        agent_log.fine("directly_evaluate_cogscms_ante_succ()")
        for atom in atomspace:
            if not is_predictive_implication_scope(atom):
                continue
            self.directly_evaluate_conjunction({get_antecedent(atom)})
            self.directly_evaluate_conjunction({get_succedent(atom)})

    def infer_cogscms(self) -> set[Atom]:
        """Discover cognitive schematics via reasoning.

        For now only temporal deduction is implemented.

        Return the set of inferred cognitive schematics.

        """

        agent_log.fine("infer_cogscms()")

        # DEBUG: Log atomspaces to be sure it contains what we want
        agent_log.fine("self.percepta_atomspace [count={}] = {}".format(
            len(self.percepta_atomspace),
            atomspace_to_str(self.percepta_atomspace)
        ))
        agent_log.fine("self.percepta_record [count={}] = {}".format(
            len(self.percepta_record),
            self.percepta_record
        ))
        agent_log.fine("self.cogscms_atomspace [count={}] = {}".format(
            len(self.cogscms_atomspace),
            atomspace_to_str(self.cogscms_atomspace)
        ))
        agent_log.fine("self.cognitive_schematics [count={}] = {}".format(
            len(self.cognitive_schematics),
            self.cognitive_schematics
        ))
        agent_log.fine("self.atomspace [count={}] = {}".format(
            len(self.atomspace),
            atomspace_to_str(self.atomspace)
        ))

        # NEXT: Infer the TVs of all antecedents and consequents of
        # cognitive schematics.  This is required with the current
        # version of temporal deduction used.
        #
        # Note: this needs to be done after
        # back-predictive-implication-scope-conditional-conjunction-introduction!!!

        # NEXT: Maybe we want to clear the working atomspace and copy
        # the content of self.percepta_atomspace and
        # self.cognitive_schematics before launching that round of
        # reasoning.  Then later on we could use Linas atomspace
        # combinations instead of copying atomspace content.
        self.working_atomspace.clear()
        add_to_atomspace(self.percepta_atomspace, self.working_atomspace)
        add_to_atomspace(self.cogscms_atomspace, self.working_atomspace)
        agent_log.fine("self.working_atomspace [count={}] = {}".format(
            len(self.working_atomspace),
            atomspace_to_str(self.working_atomspace)
        ))

        # All resulting cognitive schematics
        cogscms: set[Atom] = set()

        # NEXT: we probably want to create this query in a child
        # atomspace of self.cogscms_atomspace
        #
        # cogscms_atomspace_child = AtomSpace(self.cogscms_atomspace)
        # set_default_atomspace(cogscms_atomspace_child)

        # Call PLN to infer new cognitive schematics by combining
        # existing ones
        V = VariableNode("$V")
        T = VariableNode("$T")
        P = VariableNode("$P")
        Q = VariableNode("$Q")
        source = \
            QuoteLink(
                BackPredictiveImplicationScopeLink(
                    UnquoteLink(V),
                    UnquoteLink(T),
                    UnquoteLink(P),
                    UnquoteLink(Q)))
        mi = 2
        rules = [
            "back-predictive-implication-scope-conditional-conjunction-introduction",
        ]

        # Split in 3 operations
        # 1. Infer conditional conjunctions
        # 2. Infer conjunctions TVs
        # 3. Infer temporal deductions

        cogscms = self.pln_fc(
            self.cogscms_atomspace,
            source,
            maxiter=mi,
            rules=rules
        )

        agent_log.fine("MID self.cogscms_atomspace [count={}] = {}".format(
            len(self.cogscms_atomspace),
            atomspace_to_str(self.cogscms_atomspace)
        ))
        agent_log.fine("MID self.working_atomspace [count={}] = {}".format(
            len(self.working_atomspace),
            atomspace_to_str(self.working_atomspace)
        ))

        # # Infer antecedents and consequents TVs of cognitive schematics
        self.directly_evaluate_cogscms_ante_succ(self.working_atomspace)

        # rules = [
        #     "back-predictive-implication-scope-deduction-cogscm",
        # ]
        # cogscms = self.pln_fc(
        #     # NEXT: use cogscms_atomspace
        #     self.working_atomspace,
        #     source,
        #     maxiter=mi,
        #     rules=rules
        # )

        agent_log.fine("AFTER self.cogscms_atomspace [count={}] = {}".format(
            len(self.cogscms_atomspace),
            atomspace_to_str(self.cogscms_atomspace)
        ))
        agent_log.fine("AFTER self.working_atomspace [count={}] = {}".format(
            len(self.working_atomspace),
            atomspace_to_str(self.working_atomspace)
        ))

        agent_log.fine("Inferred cognitive schematics = {}".format(cogscms))
        return cogscms

    def update_cognitive_schematics(self, new_cogscms: set[Atom]):
        """Insert the new cognitive schematics into the proper container."""

        add_to_atomspace(new_cogscms, self.cogscms_atomspace)
        self.cognitive_schematics.update(new_cogscms)

    def learn(self):
        """Discover patterns in the world and in itself."""

        # For now we only learn cognitive schematics.  Later on we can
        # introduce more types of knowledge, temporal and more
        # abstract.

        # Mine cognitive schematics
        mined_cogscms = self.mine_cogscms()
        self.update_cognitive_schematics(mined_cogscms)

        # Infer cognitive schematics (via temporal deduction)
        if self.temporal_deduction:
            inferred_cogscms = self.infer_cogscms()
            self.update_cognitive_schematics(inferred_cogscms)

    def get_pattern(self, surprise_eval: Atom) -> Atom:
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

    def is_T(self, var: Atom) -> bool:
        """Return True iff the variable is (Variable "$T")."""

        return var == VariableNode("$T")

    def is_temporally_typed(self, tvar: Atom) -> bool:
        """ "Return True iff the variable is typed as temporal.

        For now a variable is typed as temporal if it is

        (Variable "$T")

        since variabled are not typed for now.

        """

        return self.is_T(tvar)

    def is_attime_T(self, clause: Atom) -> bool:

        """Return True iff the clause is about an event occuring at time T.

        That is if it is of the form

            AtTime
              <event>
              Variable "$T"

        """

        return self.is_T(clause.out[1])

    def get_pattern_timed_clauses(self, pattern: Atom) -> Atom:
        """Return all timestamped clauses of a pattern."""

        return pattern.out[1].out

    def get_pattern_antecedent_events(self, pattern: Atom) -> list[Atom]:
        """Return the antecedent events of a temporal pattern.

        That is all propositions not taking place at the latest time.

        """

        tclauses = get_early_clauses(self.get_pattern_timed_clauses(pattern))
        return get_events(tclauses)

    def get_pattern_succedent_events(self, pattern: Atom) -> Atom:
        """Return the succedent events of a temporal pattern.

        That is the propositions taking place at the lastest time.

        """

        tclauses = get_latest_clauses(self.get_pattern_timed_clauses(pattern))
        return get_events(tclauses)

    def get_typed_variables(self, vardecl: Atom) -> list[Atom]:
        """Get the list of possibly typed variables in vardecl."""

        vt = vardecl.type
        if is_a(vt, types.VariableList) or is_a(vt, types.VariableSet):
            return vardecl.out
        else:
            return [vardecl]

    def get_nt_vardecl(self, pattern: Atom) -> Atom:
        """Get the vardecl of pattern excluding the time variable."""

        vardecl = get_vardecl(pattern)
        tvars = self.get_typed_variables(vardecl)
        nt_tvars = [tvar for tvar in tvars if not self.is_temporally_typed(tvar)]
        return VariableSet(*nt_tvars)

    def to_sequential_and(self, timed_clauses: list[Atom]) -> Atom:
        times = get_times(timed_clauses)
        if len(times) == 2:
            # Partition events in first and second time. For now lags
            # are assumed to be 1.
            early_clauses = get_early_clauses(timed_clauses)
            early_events = get_events(early_clauses)
            latest_clauses = get_latest_clauses(timed_clauses)
            latest_events = get_events(latest_clauses)
            return BackSequentialAndLink(
                to_nat(1), maybe_and(early_events), maybe_and(latest_events)
            )
        else:
            agent_log.error("Not supported yet!")

    def to_predictive_implicant(self, pattern: Atom) -> Atom:
        """Turn a temporal pattern into predictive implicant."""

        agent_log.fine("to_predictive_implicant(pattern={})".format(pattern))

        timed_clauses = self.get_pattern_timed_clauses(pattern)
        early_clauses = get_early_clauses(timed_clauses)
        early_times = get_times(early_clauses)
        agent_log.fine("timed_clauses = {})".format(timed_clauses))
        agent_log.fine("early_clauses = {})".format(early_clauses))
        agent_log.fine("early_times = {})".format(early_times))
        if len(early_times) == 1:  # No need of SequentialAnd
            return maybe_and(get_events(early_clauses))
        else:
            return self.to_sequential_and(early_clauses)

    def to_predictive_implicand(self, pattern: Atom) -> Atom:
        """Turn a temporal pattern into predictive implicand."""

        return maybe_and(self.get_pattern_succedent_events(pattern))

    def to_predictive_implication_scope(self, pattern: Atom) -> Atom:
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

        BackPredictiveImplicationScope
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

        BackPredictiveImplicationScope
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

        BackPredictiveImplication <1 - s, c>
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
        # TODO: fix python BackPredictiveImplicationScopeLink binding!
        # preimp = BackPredictiveImplicationScopeLink(ntvardecl, lag, pt, pd)
        preimp = scheme_eval_h(
            self.atomspace,
            "(BackPredictiveImplicationScopeLink "
            + str(ntvardecl)
            + str(lag)
            + str(pt)
            + str(pd)
            + ")",
        )
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
        rules = ["back-predictive-implication-scope-direct-evaluation"]
        return self.pln_bc(self.atomspace, preimp, maximum_iterations=mi, rules=rules).pop()

    def is_desirable(self, cogscm: Atom) -> bool:
        """Return True iff the cognitive schematic is desirable.

        For now to be desirable a cognitive schematic must have

        0. a well define atom
        1. its confidence above zero
        2. its action fully grounded
        3. all its variables in the antecedent
        4. A mean of one, if self.true_cogscm is true
        5. An empty vardecl, if self.empty_vardecl_cogscm is true

        """

        return (
            cogscm
            and has_non_null_confidence(cogscm)
            and is_closed(get_t0_execution(cogscm))
            and has_all_variables_in_antecedent(cogscm)
            and (not(self.true_cogscm) or has_one_mean(cogscm))
            and (not(self.empty_vardecl_cogscm) or has_empty_vardecl(cogscm))
        )

    def surprises_to_predictive_implications(self, srps : Atom) -> list[Atom]:
        """Like to_predictive_implication but takes surprises."""

        agent_log.fine("surprises_to_predictive_implications(srps={})".format(srps))

        # Turn patterns into predictive implication scopes
        cogscms = [
            self.to_predictive_implication_scope(self.get_pattern(srp)) for srp in srps
        ]

        # Remove undesirable cognitive schematics
        cogscms = [cogscm for cogscm in cogscms if self.is_desirable(cogscm)]

        return cogscms

    def to_timed_clauses(self,
                         lagged_antecedents_succedents: tuple,
                         T: Atom) -> tuple[list[Atom], int]:
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

    def mine_temporal_patterns(self, atomspace: AtomSpace,
                               las: tuple,
                               vardecl: Atom=None) -> list[Atom]:
        """Given nested lagged, antecedents, succedents, mine temporal patterns.

        More precisely it takes

        # NEXT: should it be the atomspace to mine
        # (which would bedetermined by Percepta Record anyway)?
        # Or the atomspace to dump the result into?
        1. an atomspace to mine

        2. las, a list of triples

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

        agent_log.fine(
            "mine_temporal_patterns(atomspace={}, las={}, vardecl={})".format(
                atomspace, las, vardecl
            )
        )

        # Set miner parameters
        minsup = 4
        maximum_iterations = 1000
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
        timed_clauses, _ = self.to_timed_clauses(las, T)
        agent_log.fine("timed_clauses = {}".format(timed_clauses))
        if not vardecl:
            variables = set([T])
            variables.update(get_free_variables_of_atoms(timed_clauses))
            vardecl = VariableSet(*variables)
        initpat = LambdaLink(vardecl, PresentLink(*timed_clauses))

        # Launch pattern miner
        mine_query = (
            "(cog-mine "
            + str(self.percepta_record_cpt)
            + " #:ignore "
            + str(ignore)
            + " #:minimum-support "
            + str(minsup)
            + " #:initial-pattern "
            + str(initpat)
            + " #:maximum-iterations "
            + str(maximum_iterations)
            + " #:conjunction-expansion "
            + cnjexp
            + " #:enforce-specialization "
            + enfspe
            + " #:maximum-variables "
            + str(maxvars)
            + " #:maximum-conjuncts "
            + str(maxcjnts)
            + " #:maximum-spcial-conjuncts "
            + str(mspc)
            + " #:surprisingness "
            + surprise
            + ")"
        )
        agent_log.fine("mine_query = {}".format(mine_query))
        surprises = scheme_eval_h(atomspace, "(List " + mine_query + ")")
        agent_log.fine("surprises = {}".format(surprises))

        return surprises.out

    def plan(self, goal: Atom, expiry: int) -> list[Atom]:
        """Plan the next actions given a goal and its expiry time offset

        Return a python list of cognivite schematics meeting the
        expiry constrain.  Whole cognitive schematics are output in
        order to make a decision based on their truth values and
        priors.

        A cognitive schematic is a knowledge piece of the form

        Context & Action ⇒ Goal

        See https://wiki.opencog.org/w/Cognitive_Schematic for more
        information.  The supported format for cognitive schematics
        are as follows

        BackPredictiveImplicationScope <tv>
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
        agent_log.fine(
            "self.cognitive_schematics = {}".format(self.cognitive_schematics)
        )

        # Retrieve all cognitive schematics meeting the constrains which are
        #
        # 1. The total lag of the cognitive schematics is below or
        #    equal to the expiry.
        #
        # 2. The succedent matches the goal
        meet = (
            lambda cogscm: get_total_lag(cogscm) <= expiry
            and get_succedent(cogscm) == goal
        )
        return [cogscm for cogscm in self.cognitive_schematics if meet(cogscm)]

    # TODO: move to its own class (MixtureModel or something)
    def get_all_uniq_atoms(self, atom: Atom) -> set[Atom]:
        """Return the set of all unique atoms in atom."""

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
    def complexity(self, atom: Atom) -> int:
        """Return the count of all unique atoms in atom."""

        return len(self.get_all_uniq_atoms(atom))

    # TODO: move to its own class (MixtureModel or something)
    def prior(self, length: float) -> float:
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
    def kolmogorov_estimate(self, remain_count: float) -> float:
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
    def prior_estimate(self, cogscm: Atom) -> float:
        """Calculate the prior probability of cogscm."""

        partial_complexity = self.complexity(cogscm)
        remain_data_size = self.data_set_size - cogscm.tv.count
        kestimate = self.kolmogorov_estimate(remain_data_size)
        return self.prior(partial_complexity + kestimate)

    # TODO: move to its own class (MixtureModel or something)
    def beta_factor(self, cogscm: Atom) -> float:
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
    def weight(self, cogscm: Atom) -> float:
        """Calculate the weight of a cogscm for Model Bayesian Averaging.

        The calculation is based on

        https://github.com/ngeiswei/papers/blob/master/PartialBetaOperatorInduction/PartialBetaOperatorInduction.pdf

        It assumes self.data_set_size has been properly set.

        """

        return self.prior_estimate(cogscm) * self.beta_factor(cogscm)

    # TODO: move to its own class (MixtureModel or something)
    def infer_data_set_size(self, cogscms: list[Atom]) -> float:
        """Infer the data set size by taking the max count of all models

        (it works assuming that one of them is complete).

        """

        if 0 < len(cogscms):
            return max(cogscms, key=lambda x: x.tv.count).tv.count
        else:
            return 0.0

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
        ctx_tv = lambda cogscm: get_context_actual_truth(
            self.atomspace, cogscm, self.step_count
        )
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
        mxmdl = omdict(
            [
                (get_t0_execution(cogscm), (self.weight(cogscm), cogscm))
                for cogscm in valid_cogscms
            ]
        )
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
        """Run one step of observation, decision and env update"""

        agent_log.debug("atomese_obs = {}".format(self.observation))
        obs_record = [
            self.record(o, self.step_count, tv=TRUE_TV) for o in self.observation
        ]
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
        agent_log.debug(
            "action with probability of success = {}".format(
                act_pblt_to_str((action, pblty))
            )
        )

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
        self.observation, reward, done = self.env.step(action)
        self.accumulated_reward += int(reward.out[1].name)
        agent_log.debug("observation = {}".format(self.observation))
        agent_log.debug("reward = {}".format(reward))
        agent_log.debug("accumulated reward = {}".format(self.accumulated_reward))

        reward_record = self.record(reward, self.step_count, tv=TRUE_TV)
        agent_log.debug("reward_record = {}".format(reward_record))

        return done
