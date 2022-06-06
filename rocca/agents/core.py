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
from orderedmultidict import omdict

# SciPy
import scipy.special as sp

# OpenCog
from opencog.atomspace import Atom, AtomSpace
from opencog.spacetime import AtTimeLink, TimeNode
from opencog.pln import (
    SLink,
    ZLink,
    BackPredictiveImplicationScopeLink,
    BackSequentialAndLink,
)
from opencog.scheme import scheme_eval, scheme_eval_h
from opencog.utilities import is_closed, set_default_atomspace
from opencog.type_constructors import (
    ConceptNode,
    AndLink,
    EvaluationLink,
    PredicateNode,
    NumberNode,
    VariableNode,
    SetLink,
    LambdaLink,
    QuoteLink,
    UnquoteLink,
    MemberLink,
)

from rocca.envs.wrappers import Wrapper

# ROCCA
from .utils import *

logging.basicConfig(filename="agent.log", format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

#########
# Class #
#########


class OpencogAgent:
    """Class for controlling an agent using OpenCog."""

    def __init__(
        self,
        env: Wrapper,
        atomspace: AtomSpace,
        action_space: set[Atom],
        p_goal: Atom,
        n_goal: Atom,
        log_level: str = "debug",
    ):
        # Construct the various atomspaces
        self.atomspace = atomspace  # Working atomspace
        self.percepta_atomspace = AtomSpace()  # TODO: make sure contains only percepta
        self.cogscms_atomspace = AtomSpace()  # TODO: make sure contains only cogscms
        self.working_atomspace = AtomSpace()
        set_default_atomspace(self.atomspace)

        self.env = env
        self.observation, _, _ = self.env.restart()
        self.cycle_count: int = 0
        self.total_count: int = 0
        self.accumulated_reward: int = 0
        self.percepta_record_cpt = ConceptNode("Percepta Record")
        # The percepta_record is a list of sets of timestamped
        # percepta.  The list is ordered by timestamp, that each
        # element of that list is a set of percepta at the timestamp
        # corresponding to its index.
        #
        # In Latin, percepta is the plurial of perceptum (percept in
        # English), according to
        # https://en.wiktionary.org/wiki/perceptus#Latin
        self.percepta_record: list[set[Atom]] = []
        self.action_space: set[Atom] = action_space
        self.positive_goal: Atom = p_goal
        self.negative_goal: Atom = n_goal
        self.cognitive_schematics: set[Atom] = set()
        self.log_level = log_level
        self.load_opencog_modules()
        self.reset_action_counter()

        ###################
        # User parameters #
        ###################

        # Prior alpha and beta of beta-distributions corresponding to
        # truth values
        self.prior_a = 1.0
        self.prior_b = 1.0

        # Construct mixture model for action decision.  See
        # MixtureModel class for user parameters pertaining to mixture
        # model.
        self.mixture_model = MixtureModel(self.action_space, self.prior_a, self.prior_b)

        # Expiry time to fulfill the goal. The system will not plan
        # beyond expiry.
        self.expiry = 2

        # Enable poly-action pattern mining
        self.polyaction_mining = True

        # Enable pattern mining with general succedent (not just about
        # goal).  This is important to gather the knowledge in order
        # to make temporal deduction useful.
        #
        # Note that general succedent mining is only done for
        # mono-action pattern for now.
        self.general_succedent_mining = True

        # Enable conditional conjunction introduction
        #
        # C ∧ A ↝ D
        # C ∧ A ↝ E
        # ⊢
        # C ∧ A ↝ D ∧ E
        #
        # This is useful when multiple consequents must be combined to
        # be used as antecedant of other cognitive schematics.
        self.conditional_conjunction_introduction = True

        # Enable temporal deduction
        #
        # C ∧ A₁ ↝ D
        # D ∧ A₂ ↝ E
        # ⊢
        # (C ∧ A₁) ⩘ A₂ ↝ E
        #
        # Useful to string together cognitive schematics into larger
        # ones.
        self.temporal_deduction = True

        # Filter out cognitive schematics with strengths below this
        # threshold.
        #
        # Beware that setting this parameter above zero makes ROCCA
        # blind to negative predictions (i.e. a given context and
        # action imply that a goal or subgoal is unlikely to be
        # reached) which are also important to make sound decisions.
        # Such parameter can however be useful for debugging or other
        # exceptional situations.
        self.cogscm_minimum_strength = 0.0

        # Filter out cognitive schematics with a Shannon entropy equal
        # to or lower than this parameter.  The Shannon entropy is
        # calculated based on the mean of the corresponding beta
        # distribution of the cognitive schematics truth value.  The
        # parameter ranges from 0 (strongest filter, only absolutely
        # true or absolutely false cognitive schematics passes
        # through) to 1 (all cognitive schematics passes through).
        #
        # For example: an atom with truth value (stv 0 0.1) has
        # Shannon entropy below 0.1, but an atom with truth value (stv
        # 0 0.01) has Shannon entropy above 0.1, and an atom with
        # truth value (stv 0 0.001) has a Shannon entropy above 0.9.
        self.cogscm_maximum_shannon_entropy = 1.0

        # Filter out cognitive schematics with a differential entropy
        # equal to or lower than this parameter.  The differential
        # entropy is calculated using the corresponding beta
        # distribution of the cognitive schematics truth value.  The
        # parameter ranges from -inf (strongest filter, only
        # absolutely true or absolutely false cognitive schematics
        # passes through) to 0 (all cognitive schematics passes
        # through).  Given a reasonably value, the effect is that only
        # cognitive schematics with both high or lower strength and
        # high confidence passes the filter.
        #
        # For example a threshold value of -0.1 will let pass through
        # cognitive schematics with truth value (stv 0.99 1e-3) or
        # (stv 0.1 1e-3), but not with truth value (stv 0.5 1e-3) or
        # (stv 0.99 1e-4).
        #
        # This parameter is comparable to
        # cogscm_maximum_shannon_entropy but better accounts for
        # confidence.
        self.cogscm_maximum_differential_entropy = 0.0

        # Filter out cognitive schematics with numbers of variables
        # above this threshold
        self.cogscm_maximum_variables = 1

        # Minimum count support for the pattern miner
        self.miner_minimum_support = 4

        # Maximum number of iterations for the pattern miner
        self.miner_maximum_iterations = 1000

        # Maximum number of variables for the pattern miner.  This
        # should not be confused with cogscm_maximum_variables that
        # represents the final maximum number of variables, while this
        # parameter represents the intermediate maximum number of
        # variables.  This parameter should always be equal to or
        # greater than cogscm_maximum_variables + 1 (for the temporal
        # variable that is explicitly represented in temporal
        # patterns).
        self.miner_maximum_variables = 8

        # Enable posting cogscms and selected action to the visualizer.
        self.visualize_cogscm = False

    def __del__(self):
        self.env.close()

    def log_parameters(self, level: str = "debug"):
        """Log all user parameters at the given log level."""

        li = agent_log.string_as_level(level)
        agent_log.log(li, "OpencogAgent parameters:")
        agent_log.log(li, "prior_a = {}".format(self.prior_a))
        agent_log.log(li, "prior_b = {}".format(self.prior_b))
        agent_log.log(
            li,
            "mixture_model.complexity_penalty = {}".format(
                self.mixture_model.complexity_penalty
            ),
        )
        agent_log.log(
            li,
            "mixture_model.compressiveness = {}".format(
                self.mixture_model.compressiveness
            ),
        )
        agent_log.log(li, "mixture_model.delta = {}".format(self.mixture_model.delta))
        agent_log.log(
            li,
            "mixture_model.weight_influence = {}".format(
                self.mixture_model.weight_influence
            ),
        )
        agent_log.log(li, "expiry = {}".format(self.expiry))
        agent_log.log(li, "polyaction_mining = {}".format(self.polyaction_mining))
        agent_log.log(
            li,
            "general_succedent_mining = {}".format(self.general_succedent_mining),
        )
        agent_log.log(li, "temporal_deduction = {}".format(self.temporal_deduction))
        agent_log.log(
            li, "cogscm_minimum_strength = {}".format(self.cogscm_minimum_strength)
        )
        agent_log.log(
            li,
            "cogscm_maximum_shannon_entropy = {}".format(
                self.cogscm_maximum_shannon_entropy
            ),
        )
        agent_log.log(
            li,
            "cogscm_maximum_differential_entropy = {}".format(
                self.cogscm_maximum_differential_entropy
            ),
        )
        agent_log.log(
            li, "cogscm_maximum_variables = {}".format(self.cogscm_maximum_variables)
        )
        agent_log.log(
            li, "miner_minimum_support = {}".format(self.miner_minimum_support)
        )
        agent_log.log(
            li, "miner_maximum_iterations = {}".format(self.miner_maximum_iterations)
        )
        agent_log.log(
            li, "miner_maximum_variables = {}".format(self.miner_maximum_variables)
        )

    def load_opencog_modules(self) -> None:
        # Load miner
        scheme_eval(self.atomspace, "(use-modules (opencog miner))")

        # Load PLN.  All rules must be pre-loaded here
        scheme_eval(self.atomspace, "(use-modules (opencog pln))")
        scheme_eval(self.atomspace, "(use-modules (opencog spacetime))")
        rules = [
            "back-predictive-implication-scope-direct-evaluation",
            "back-predictive-implication-scope-conditional-conjunction-introduction",
            "back-predictive-implication-scope-deduction-cogscm-Q-conjunction",
            "back-predictive-implication-scope-deduction-cogscm-Q-evaluation",
        ]
        self.pln_load_rules(rules)
        # scheme_eval(self.atomspace, "(pln-log-atomspace)")

    def reset_action_counter(self) -> None:
        self.action_counter: Counter[Atom] = Counter(
            {action: 0 for action in self.action_space}
        )

    def action_counter_to_str(self) -> str:
        """Pretty print self.action_counter."""

        ss: list[str] = [
            "{}: {}".format(to_human_readable_str(p[0]), p[1])
            for p in self.action_counter.items()
        ]
        return "\n".join(ss)

    def insert_to_percepta_record(self, timed_atom: Atom, i: int = -1) -> None:
        """Insert a timestamped atom into self.percepta_record.

        The percepta_record is a list of sets of timestamped percepta.
        The list is ordered by timestamp, that each element of that
        list is a set of percepta at the timestamp corresponding to
        its index.

        If it's index i is not provided (or is negative), then it is
        extracted from the timestamp.

        """

        if i < 0:
            i = to_int(get_time(timed_atom))

        while len(self.percepta_record) <= i:
            self.percepta_record.append(set())
        self.percepta_record[i].add(timed_atom)

    def timed_percepta_to_scheme_str(
        self, timed_percepta: set[Atom], with_member: bool = False
    ) -> str:
        """Convert percepta at a given cycle into a string in Scheme format.

        Percepta are preceded by a comment in human readable form.

        If with_member is set to True then each perceptum is wrapped with
        a member link from it to the percepta record concept.

        """

        # Wrap member to percepta record concept around each perceptum
        if with_member:
            timed_percepta = {
                self.add_to_percepta_atomspace(timed_perceptum)
                for timed_perceptum in timed_percepta
            }
        cmt = "\n".join(";; " + to_human_readable_str(tpm) for tpm in timed_percepta)
        scm = "\n".join(tpm.long_string() for tpm in timed_percepta)
        return "\n".join([cmt, scm])

    def percepta_record_to_scheme_str(self, with_member: bool = False) -> str:
        """Convert a percepta record into a string in Scheme format.

        Each perception is preceded by a comment in human readable form.

        """

        return "\n\n".join(
            [
                self.timed_percepta_to_scheme_str(tp, with_member)
                for tp in self.percepta_record
            ]
        )

    def add_to_percepta_atomspace(self, timed_atom: Atom) -> Atom:
        """Add member link around timestamped atom to Percepta Atomspace.

        Return the added member link.

        """

        mbr = MemberLink(timed_atom, self.percepta_record_cpt, tv=TRUE_TV)
        return self.percepta_atomspace.add_atom(mbr)

    def record(self, atom: Atom, i: int, tv=None) -> Atom:
        """Timestamp and record an atom to the Percepta Record.

        That is add the following in the percepta atomspace

        MemberLink (stv 1 1)
          AtTimeLink <tv>
            <atom>
            <i>
          <self.percepta_record_cpt>

        As well as the AtTimeLink to self.percepta_record (see
        insert_to_percepta_record method for more info).

        """

        timed_atom = timestamp(atom, i, tv)
        self.insert_to_percepta_record(timed_atom, i)
        return self.add_to_percepta_atomspace(timed_atom)

    def make_goal(self) -> Atom:
        """Define the goal of the current iteration.

        By default the goal of the current iteration is to have a
        reward of 1.

        Evaluation
          Predicate "Reward"
          Number 1

        """

        return EvaluationLink(PredicateNode("Reward"), NumberNode(str(1)))

    def pln_load_rules(self, rules: list[str] = []):
        """Load PLN rules.

        Take a list of rule scheme symbols (but without the single
        quote for the symbol), such as

        ["back-predictive-implication-scope-direct-evaluation",
         "back-predictive-implication-scope-deduction-cogscm"]

        """

        for rule in rules:
            scheme_eval(self.atomspace, "(pln-load-rule '" + rule + ")")

    def pln_fc(
        self,
        atomspace: AtomSpace,
        source: Atom,
        vardecl=None,
        maximum_iterations: int = 10,
        full_rule_application: bool = False,
        rules: list[str] = [],
    ) -> set[Atom]:
        """Call PLN forward chainer with the given source and parameters.

        The parameters are

        atomspace: the atomspace over which to do the reasoning. # TODO: find out if it really does that
        source: the atom source to start from.
        maximum_iterations: the maximum number of iterations.
        rules: optional list of rule symbols.  If empty keep current rule set.

        Return a python list of solutions.

        """

        agent_log.fine(
            "pln_fc(atomspace={}, source={}, maximum_iterations={}, full_rule_application={}, rules={})".format(
                atomspace,
                atom_to_idstr(source),
                maximum_iterations,
                full_rule_application,
                rules,
            )
        )

        # Add rules (should be previously loaded)
        if rules:
            scheme_eval(self.atomspace, "(pln-rm-all-rules)")
            for rule in rules:
                er = scheme_eval(self.atomspace, "(pln-add-rule '" + rule + ")")

        # Log the entire atomspace (fine level), uncomment to enable.
        # agent_log_atomspace(atomspace)

        # Generate and run query
        command = "(pln-fc "
        command += str(source)
        command += ("#:vardecl " + str(vardecl)) if vardecl else ""
        command += " #:maximum-iterations " + str(maximum_iterations)
        command += " #:fc-full-rule-application " + to_scheme_str(full_rule_application)
        command += ")"

        # Log FC query before running
        agent_log.fine("PLN forward chainer query:\n{}".format(command))

        # Run query and return results
        return set(scheme_eval_h(atomspace, command).out)

    def pln_bc(
        self,
        atomspace: AtomSpace,
        target: Atom,
        vardecl=None,
        maximum_iterations: int = 10,
        rules: list[str] = [],
    ) -> set[Atom]:
        """Call PLN backward chainer with the given target and parameters.

        The parameters are

        maximum_iterations: the maximum number of iterations.
        rules: optional list of rule symbols.  If empty keep current rule set.

        Return a python list of solutions.

        """

        agent_log.fine(
            "pln_bc(atomspace={}, target={}, maximum_iterations={}, rules={})".format(
                atomspace, atom_to_idstr(target), maximum_iterations, rules
            )
        )

        # Add rules (should be previously loaded)
        if rules:
            scheme_eval(self.atomspace, "(pln-rm-all-rules)")
            for rule in rules:
                er = scheme_eval(self.atomspace, "(pln-add-rule '" + rule + ")")

        # Log the entire atomspace (fine level), uncomment to enable.
        # agent_log_atomspace(atomspace)

        # Generate and run query
        command = "(pln-bc "
        command += str(target)
        command += ("#:vardecl " + str(vardecl)) if vardecl else ""
        command += " #:maximum-iterations " + str(maximum_iterations)
        command += ")"

        # Log BC query before running
        agent_log.fine("PLN backward chainer query:\n{}".format(command))

        # Run query and return results
        return set(scheme_eval_h(atomspace, command).out)

    def mine_cogscms(self) -> set[Atom]:
        """Discover cognitive schematics via pattern mining.

        Return the set of mined cognitive schematics.

        """

        # Log the percepta record in Scheme format, useful for
        # debugging the pattern miner
        agent_log.fine(
            "Percepta record:\n{}".format(self.percepta_record_to_scheme_str())
        )

        # TODO: hack, copy self.percepta_atomspace into self.atomspace
        # till we use self.percepta_atomspace
        copy_atomspace(self.percepta_atomspace, self.atomspace)

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
            # TODO: use percepta_atomspace
            pos_srps = self.mine_temporal_patterns(self.atomspace, las)
            pos_prdi = self.surprises_to_predictive_implications(pos_srps)
            agent_log.fine(
                "Mined positive goal cognitive schematics [count={}]:\n{}".format(
                    len(pos_prdi), atoms_to_scheme_str(pos_prdi)
                )
            )
            cogscms.update(set(pos_prdi))

            # Mine negative succedent goals
            postctxs = [self.negative_goal]
            las = (lag, prectxs, postctxs)
            # TODO: use percepta_atomspace
            neg_srps = self.mine_temporal_patterns(self.atomspace, las)
            neg_prdi = self.surprises_to_predictive_implications(neg_srps)
            agent_log.fine(
                "Mined negative goal cognitive schematics [count={}]:\n{}".format(
                    len(neg_prdi), atoms_to_scheme_str(neg_prdi)
                )
            )
            cogscms.update(set(neg_prdi))

            # Mine general succedents (only mono-action for now)
            if self.general_succedent_mining:
                postctxs = [EvaluationLink(VariableNode("$R"), VariableNode("$Z"))]
                las = (lag, prectxs, postctxs)
                # TODO: use percepta_atomspace
                gen_srps = self.mine_temporal_patterns(self.atomspace, las)
                gen_prdi = self.surprises_to_predictive_implications(gen_srps)
                agent_log.fine(
                    "Mined general succedent cognitive schematics [count={}]:\n{}".format(
                        len(gen_prdi), atoms_to_scheme_str(gen_prdi)
                    )
                )
                cogscms.update(set(gen_prdi))

            # Mine positive succedent goals with poly-actions
            if self.polyaction_mining:
                postctxs = [self.positive_goal]
                for snd_action in self.action_space:
                    agent_log.fine(
                        "polyaction mining snd_action = {}".format(snd_action)
                    )
                    ma_prectxs = (lag, prectxs, [snd_action])
                    compo_las = (lag, ma_prectxs, postctxs)
                    # TODO: use percepta_atomspace
                    pos_poly_srps = self.mine_temporal_patterns(
                        self.atomspace, compo_las
                    )
                    agent_log.fine(
                        "Mined positive goal poly-action cognitive schematics[count={}]\n:{}".format(
                            len(pos_poly_srps), atoms_to_scheme_str(pos_poly_srps)
                        )
                    )
                    pos_poly_prdi = self.surprises_to_predictive_implications(
                        pos_poly_srps
                    )
                    cogscms.update(set(pos_poly_prdi))

        agent_log.debug(
            "Mined cognitive schematics [count={}]:\n{}".format(
                len(cogscms), atoms_to_scheme_str(cogscms)
            )
        )
        return cogscms

    def directly_evaluate(self, atom: Atom) -> None:
        """Directly evaluate the TV of the given atoms.

        If the atom is a conjunction, then split that atom into
        conjuncts and directly evaluate their combination.

        """

        agent_log.fine("directly_evaluate(atom={})".format(atom_to_idstr(atom)))

        # Exit now to avoid division by zero
        if self.total_count == 0:
            return

        # Exit if the confidence will not increase
        conf = count_to_confidence(self.total_count)
        if conf <= atom.tv.confidence:
            return

        # Count the positive occurrences of atom across time (percepta
        # are assumed to be absolutely true).
        pos_count = 0
        for timed_events in self.percepta_record:
            events = set(get_events(timed_events))
            conjuncts = set(atom.out) if is_and(atom) else {atom}
            if conjuncts <= events:
                pos_count += 1

        # Update the TV of atom
        mean = float(pos_count) / float(self.total_count)
        atom.truth_value(mean, conf)

    def directly_evaluate_cogscms_ante_succ(self, atomspace: AtomSpace) -> None:
        """Directly evaluate the TVs of all cogscms outgoings in given atomspace."""

        for atom in atomspace:
            if not is_predictive_implication_scope(atom):
                continue
            self.directly_evaluate(get_antecedent(atom))
            self.directly_evaluate(get_succedent(atom))

    def apply_conditional_conjunction_introduction(
        self, atomspace: AtomSpace
    ) -> set[Atom]:
        """Infer conditional conjunctions from cogscms in given atomspace.

        Meaning, apply conditional conjunction introduction

        C ∧ A ↝ D
        C ∧ A ↝ E
        ⊢
        C ∧ A ↝ D ∧ E

        """

        # Call PLN to infer new cognitive schematics by combining
        # existing ones
        V = VariableNode("$V")
        T = VariableNode("$T")
        P = VariableNode("$P")
        Q = VariableNode("$Q")
        source = QuoteLink(
            BackPredictiveImplicationScopeLink(
                UnquoteLink(V), UnquoteLink(T), UnquoteLink(P), UnquoteLink(Q)
            )
        )
        mi = 1
        rules = [
            "back-predictive-implication-scope-conditional-conjunction-introduction",
        ]
        cogscms = self.pln_fc(
            self.cogscms_atomspace,
            source,
            maximum_iterations=mi,
            rules=rules,
        )
        return cogscms

    def apply_temporal_deduction(self, atomspace: AtomSpace) -> set[Atom]:
        """Infer temporal deduction from cogscms in given atomspace.

        Meaning apply temporal deduction

        C ∧ A₁ ↝ D
        D ∧ A₂ ↝ E
        ⊢
        (C ∧ A₁) ⩘ A₂ ↝ E

        """

        # TODO: apply all rules for now (till the unifier gets fixed)
        source = SetLink()
        mi = 1
        rules = [
            "back-predictive-implication-scope-deduction-cogscm-Q-conjunction",
            "back-predictive-implication-scope-deduction-cogscm-Q-evaluation",
        ]
        return self.pln_fc(
            self.cogscms_atomspace, source, maximum_iterations=mi, rules=rules
        )

    def infer_cogscms(self) -> set[Atom]:
        """Discover cognitive schematics via reasoning.

        For now only temporal deduction is implemented and the
        reasoning task is decomposed into 3 phases:

          1. Infer conditional conjunctions
          2. Infer TVs inside predictive implications (for temporal deduction)
          3. Infer temporal deductions

        then return the set of inferred cognitive schematics.

        """

        # All resulting cognitive schematics
        cogscms: set[Atom] = set()

        # 1. Infer conditional conjunctions

        if self.conditional_conjunction_introduction:
            agent_log.fine(
                "cogscms_atomspace before inferring conditional conjunctions [count={}]:\n{}".format(
                    len(self.cogscms_atomspace),
                    atomspace_to_str(self.cogscms_atomspace),
                )
            )
            inferred_cond_cjns = self.apply_conditional_conjunction_introduction(
                self.cogscms_atomspace
            )
            cogscms.update(inferred_cond_cjns)

        # 2. Infer antecedents and consequents TVs of cognitive schematics

        if self.temporal_deduction:  # Only needed if temporal deduction is enabled
            agent_log.fine(
                "cogscms_atomspace before inferring cognitive schematics outgoings [count={}]:\n{}".format(
                    len(self.cogscms_atomspace),
                    atomspace_to_str(self.cogscms_atomspace),
                )
            )
            self.directly_evaluate_cogscms_ante_succ(self.cogscms_atomspace)

        # 3. Infer temporal deductions

        if self.temporal_deduction:
            agent_log.fine(
                "cogscms_atomspace before temporal deduction [count={}]:\n{}".format(
                    len(self.cogscms_atomspace),
                    atomspace_to_str(self.cogscms_atomspace),
                )
            )
            inferred_cogscms = self.apply_temporal_deduction(self.cogscms_atomspace)
            cogscms.update(inferred_cogscms)

        # Log all inferred cognitive schematics
        agent_log.debug(
            "Inferred cognitive schematics [count={}]:\n{}".format(
                len(cogscms), atoms_to_scheme_str(cogscms)
            )
        )

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

        # Set the total count, will be used to calculate some TVs
        self.total_count = self.cycle_count

        # Mine cognitive schematics
        mined_cogscms = self.mine_cogscms()
        self.update_cognitive_schematics(mined_cogscms)

        # Infer cognitive schematics
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

        agent_log.fine(
            "to_predictive_implicant(pattern={})".format(atom_to_idstr(pattern))
        )

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

        agent_log.fine(
            "to_predictive_implication_scope(pattern={})".format(atom_to_idstr(pattern))
        )

        # Get the predictive implication implicant and implicand
        # respectively
        pt = self.to_predictive_implicant(pattern)
        pd = self.to_predictive_implicand(pattern)

        agent_log.fine("pt = {}".format(atom_to_idstr(pt)))
        agent_log.fine("pd = {}".format(atom_to_idstr(pd)))

        # HACK: big hack, pd is turned into positive goal to create a
        # predictive implication of such positive goal with low
        # strength.
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

        # Calculate the truth value of the predictive implication
        mi = 2
        rules = ["back-predictive-implication-scope-direct-evaluation"]
        # TODO: use working_atomspace
        return self.pln_bc(
            self.atomspace, preimp, maximum_iterations=mi, rules=rules
        ).pop()

    def is_desirable(self, cogscm: Atom) -> bool:
        """Return True iff the cognitive schematic is desirable.

        For now to be desirable a cognitive schematic must have

        0. a well define atom
        1. its confidence above zero
        2. its action fully grounded
        3. all its variables in the antecedent
        4. its Shannon entropy must be equal to or lower than
           cogscm_maximum_shannon_entropy
        5. its differential entropy must be equal to or lower than
           cogscm_maximum_differential_entropy
        6. a number of variables less than or equal to cogscm_maximum_variables

        """

        # For logging
        cogscm_str = atom_to_scheme_str(cogscm) if cogscm else str(cogscm)
        msg = "{} is undesirable because ".format(cogscm_str)

        # Check that cogscm is defined
        if not cogscm:
            agent_log.fine(msg + "is it undefined")
            return False

        # Check that it has confidence greater than 0
        if not has_non_null_confidence(cogscm):
            agent_log.fine(msg + "it has null confidence")
            return False

        # Check that it is closed
        if not is_closed(get_t0_execution(cogscm)):
            agent_log.fine(msg + "it is not closed")
            return False

        # Check that all variables are in the antecedent
        if not has_all_variables_in_antecedent(cogscm):
            agent_log.fine(msg + "some variables are not in its antecedent")
            return False

        # Check that its strength is above than or equal to the
        # minimum threshold
        st = cogscm.tv.mean
        if st < self.cogscm_minimum_strength:
            agent_log.fine(
                msg
                + "its strength {} is below {}".format(st, self.cogscm_minimum_strength)
            )
            return False

        # Check that its Shannon entropy is below the maximum threshold
        if self.cogscm_maximum_shannon_entropy < 1.0:
            se = shannon_entropy(cogscm, self.prior_a, self.prior_b)
            if self.cogscm_maximum_shannon_entropy < se:
                agent_log.fine(
                    msg
                    + "its Shannon entropy {} is greater than {}".format(
                        se, self.cogscm_maximum_shannon_entropy
                    )
                )
                return False

        # Check that its differential entropy is below the maximum threshold
        if self.cogscm_maximum_differential_entropy < 0.0:
            de = differential_entropy(cogscm, self.prior_a, self.prior_b)
            if self.cogscm_maximum_differential_entropy < de:
                agent_log.fine(
                    msg
                    + "its differential entropy {} is greater than {}".format(
                        de, self.cogscm_maximum_differential_entropy
                    )
                )
                return False

        # Check that it has no more variables than allowed
        mv = vardecl_size(get_vardecl(cogscm))
        if self.cogscm_maximum_variables < mv:
            agent_log.fine(
                msg
                + "its number of variables {} is greater than {}".format(
                    mv, self.cogscm_maximum_variables
                )
            )
            return False

        # Everything checks, it is desirable
        return True

    def surprises_to_predictive_implications(self, srps: list[Atom]) -> list[Atom]:
        """Like to_predictive_implication but takes surprises."""

        agent_log.fine("surprises_to_predictive_implications(srps={})".format(srps))

        # Turn patterns into predictive implication scopes
        cogscms = [
            self.to_predictive_implication_scope(self.get_pattern(srp)) for srp in srps
        ]

        # Remove undesirable cognitive schematics
        len_cogscms = len(cogscms)
        cogscms = [cogscm for cogscm in cogscms if self.is_desirable(cogscm)]
        len_desirable_cogscms = len(cogscms)
        agent_log.fine(
            "Among {} cognitive schematics, {} are desirable".format(
                len_cogscms, len_desirable_cogscms
            )
        )

        return cogscms

    def to_timed_clauses(
        self, las: tuple[int, Any, Any], T: Atom
    ) -> tuple[list[Atom], int]:
        """Turn nested lagged, antecedents, succedents to AtTime clauses.

        For instance the input is

        (lag-2, (lag-1, [X1, X2], [Y]), [Z])

        then it is transformed into the following clauses

        [AtTime
           X1
           T,
        AtTime
           X2
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

        agent_log.fine("to_timed_clauses(las={}, T={})".format(las, T))

        lag = las[0]
        antecedents = las[1]
        succedents = las[2]

        if type(antecedents) is list:
            timed_clauses = [AtTimeLink(ante, T) for ante in antecedents]
        else:
            timed_clauses, reclag = self.to_timed_clauses(antecedents, T)
            lag += reclag

        lagnat = lag_to_nat(lag, T)
        timed_clauses += [AtTimeLink(succ, lagnat) for succ in succedents]
        return timed_clauses, lag

    def mine_temporal_patterns(
        self, atomspace: AtomSpace, las: tuple[int, Any, Any], vardecl: Atom = None
    ) -> list[Atom]:
        """Given nested lagged, antecedents, succedents, mine temporal patterns.

        More precisely it takes

        1. an atomspace to mine (TODO: should it be the atomspace to
        mine (which would bedetermined by Percepta Record anyway)?  Or
        the atomspace to dump the result into?)

        2. las, a list of triples

        (lag, antecedents, succedents)

        where antecedents and succedents can themselves be triples of
        lagged antecedents and succedents.

        For instance the input is

        (lag-2, (lag-1, [X], [Y]), [Z])

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
        minsup = self.miner_minimum_support
        maxiter = self.miner_maximum_iterations
        maxvars = self.miner_maximum_variables
        cnjexp = "#f"
        enfspe = "#t"
        mspc = 6
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
            + " #:ignore-variables "
            + str(ignore)
            + " #:minimum-support "
            + str(minsup)
            + " #:initial-pattern "
            + str(initpat)
            + " #:maximum-iterations "
            + str(maxiter)
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
        agent_log.fine("Miner query:\n{}".format(mine_query))
        surprises = scheme_eval_h(atomspace, "(List " + mine_query + ")")
        agent_log.fine(
            "Surprising patterns [count={}]:\n{}".format(
                surprises.arity, surprises.long_string()
            )
        )

        return surprises.out

    def plan(self, goal: Atom, expiry: int) -> list[Atom]:
        """Plan the next actions given a goal and its expiry time offset

        Return a python list of cognivite schematics meeting the
        expiry constraint.  Whole cognitive schematics are output in
        order to make a decision based on their truth values and
        priors.

        A cognitive schematic is a knowledge piece of the form

        Context & Action ⇒ Goal

        See https://wiki.opencog.org/w/Cognitive_Schematic for more
        information.  The supported format for cognitive schematics
        is as follows

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
            "self.cognitive_schematics [count={}]:\n{}".format(
                len(self.cognitive_schematics),
                atoms_to_scheme_str(self.cognitive_schematics),
            )
        )

        # Retrieve all cognitive schematics meeting the constraints
        # which are
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

    def deduce(self, cogscms: list[Atom]) -> omdict[Atom, tuple[float, Atom]]:
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
            self.atomspace, cogscm, self.cycle_count
        )
        valid_cogscms = [cogscm for cogscm in cogscms if 0.9 < ctx_tv(cogscm).mean]
        agent_log.debug(
            "[cycle={}] Valid cognitive schematics [count={}]:\n{}".format(
                self.cycle_count,
                len(valid_cogscms),
                atoms_to_scheme_str(valid_cogscms, only_id=True),
            )
        )

        # Return the mixture model for that cycle
        return self.mixture_model.mk_mxmdl(valid_cogscms, self.total_count)

    def decide(
        self, mxmdl: omdict[Atom, tuple[float, Atom]]
    ) -> tuple[Atom, Atom, float]:
        """Select the next action to enact from a mixture model of cogscms.

        The action is selected from a mixture model, a multimap from
        action to weighted cognitive schematics obtained from deduce.
        The selection uses Thompson sampling leveraging the second
        order distribution to balance exploitation and
        exploration. See
        http://auai.org/uai2016/proceedings/papers/20.pdf for more
        details about Thompson Sampling.

        """

        # Select the action alongside its first order probability
        # estimate of success
        action, _, cogscm, pblty, _ = self.mixture_model.thompson_sample(mxmdl)

        # Return the action, the cognitive schematic it comes from,
        # and it probability estimate of success.
        return (action, cogscm, pblty)

    def control_cycle(self) -> bool:
        """Run one cycle of

        1. Observe
        2. Plan
        3. Act

        More precisely

        1. Observe
        1.1. Timestamp current observations in percepta record
        2. Plan
        2.1. Select the goal for that iteration
        2.2. Find plans for that goal
        2.3. Deduce a distribution of actions
        2.4. Select the next action
        3. Act
        3.1. Timestamp that action
        3.2. Run that action (update the environment)
        3.3. save the resulting observations and reward
        3.4. Timestamp the reward

        Return whether we're done for that session (as determined by
        the environment).

        Note that at the moment no learning is taking place during
        that cycle, eventhough in principle it could take place during
        the

        2.2. Find plans for that goal

        step.  For learning plans (and more), the learn() method takes
        care of that and should be called at appropriate times.

        """

        obs_record = [
            self.record(o, self.cycle_count, tv=TRUE_TV) for o in self.observation
        ]
        agent_log.fine(
            "[cycle={}] Timestamped observations:\n{}".format(
                self.cycle_count, atoms_to_scheme_str(obs_record)
            )
        )

        # Make the goal for that iteration
        goal = self.make_goal()
        agent_log.debug(
            "[cycle={}] Goal for that cycle:\n{}".format(
                self.cycle_count, atom_to_scheme_str(goal)
            )
        )

        # Plan, i.e. come up with cognitive schematics as plans.  Here
        # the goal expiry is 2, i.e. must be fulfilled set for the
        # next two iterations.
        cogscms = self.plan(goal, self.expiry)
        agent_log.debug(
            "[cycle={}] Planned cognitive schematics [count={}]:\n{}".format(
                self.cycle_count, len(cogscms), atoms_to_scheme_str(cogscms)
            )
        )

        # Deduce the action distribution
        mxmdl = self.deduce(cogscms)
        agent_log.debug(
            "[cycle={}] Mixture models:\n{}".format(
                self.cycle_count, self.mixture_model.mxmdl_to_str(mxmdl)
            )
        )

        # Select the next action
        action, cogscm, pblty = self.decide(mxmdl)
        agent_log.debug(
            "[cycle={}] Selected action {} from {} with probability of success {}".format(
                self.cycle_count,
                to_human_readable_str(action),
                atom_to_idstr(cogscm),
                pblty,
            )
        )

        # Timestamp the action that is about to be executed
        action_record = self.record(action, self.cycle_count, tv=TRUE_TV)
        agent_log.fine(
            "[cycle={}] Timestamped action:\n{}".format(
                self.cycle_count, atom_to_scheme_str(action_record)
            )
        )
        agent_log.debug(
            "[cycle={}] Action to execute:\n{}".format(
                self.cycle_count, atom_to_scheme_str(action)
            )
        )

        # Increment the counter for that action and log it
        self.action_counter[action] += 1
        agent_log.debug(
            "[cycle={}] Action counter [total={}]:\n{}".format(
                self.cycle_count,
                self.action_counter.total(),
                self.action_counter_to_str(),
            )
        )
        if self.visualize_cogscm:
            obs = " ".join([str(i) for i in obs_record]) if self.cycle_count > 0 else ""
            if cogscm:
                msg = "cogscm: {} \r\nSelected Action: {}".format(
                    to_human_readable_str(cogscm), to_human_readable_str(action)
                )
                obs_cogscm_act = obs + str(cogscm) + str(action_record)
            else:
                msg = "cogscm: N/A \r\nSelected Action: {}".format(
                    to_human_readable_str(action)
                )
                obs_cogscm_act = obs + str(action_record)

            vis_cogscm_info = {
                "cogscm": obs_cogscm_act,
                "cycle": self.cycle_count,
                "msg": msg,
            }
            post_to_restapi_scheme_endpoint(vis_cogscm_info)

        # Increase the step count and run the next step of the environment
        self.cycle_count += 1
        # TODO gather environment info.
        self.observation, reward, done = self.env.step(action)
        self.accumulated_reward += int(reward.out[1].name)
        agent_log.debug(
            "[cycle={}] Observations [count={}]:\n{}".format(
                self.cycle_count,
                len(self.observation),
                atoms_to_scheme_str(self.observation),
            )
        )
        agent_log.debug(
            "[cycle={}] Reward:\n{}".format(
                self.cycle_count, atom_to_scheme_str(reward)
            )
        )
        agent_log.debug(
            "[cycle={}] Accumulated reward = {}".format(
                self.cycle_count, self.accumulated_reward
            )
        )

        reward_record = self.record(reward, self.cycle_count, tv=TRUE_TV)
        agent_log.fine(
            "[cycle={}] Timestamped reward:\n{}".format(
                self.cycle_count, atom_to_scheme_str(reward_record)
            )
        )

        return done

    def save_percepta_atomspace(self, filepath: str, overwrite: bool = True) -> bool:
        """Save the percepta atomspace at the indicated filepath.

        If `overwrite` is set to True (the default), then the file is
        cleared before being written.

        The percepta atomspace is saved in Scheme format.

        Return False if it fails, True otherwise.

        """

        with open(filepath, "w" if overwrite else "a") as file:
            file.write(self.percepta_record_to_scheme_str(with_member=True) + "\n")
        return True

    def load_percepta_atomspace(
        self, filepath: str, overwrite: bool = True, fast: bool = True
    ) -> bool:
        """Load the percepta atomspace from the given filepath.

        The file should be in Scheme format.

        If `overwrite` is set to True (the default), then the
        percepta atomspace is cleared before being written.

        If `fast` is set to True (the default), then the atomspace is
        loaded with OpenCog's built-in function for fast loading.
        Note however that in that case the file should not contain any
        Scheme code beside Atomese constructs.  (WARNING: only
        fast==True is support for now).

        Return False if it fails, True otherwise.  (WARNING: not
        supported yet, always return True).  If successful it will
        automatically update the percepta record and the cycle count
        so that new percepts do not have the same timestamps as the
        just loaded ones.

        It is assumed that the percepta record concept is

        Concept "Percepta Record"

        """

        success = load_atomspace(self.percepta_atomspace, filepath, overwrite)
        if success:
            if overwrite:
                self.percepta_record.clear()
            pas_roots = atomspace_roots(self.percepta_atomspace)
            for mbr in pas_roots:
                self.insert_to_percepta_record(mbr.out[0])
            new_count = len(self.percepta_record)
            self.cycle_count = new_count - 1 if 0 < new_count else 0
        return success

    def save_cogscms_atomspace(self, filepath: str, overwrite: bool = True) -> bool:
        """Save the cogscm atomspace at the indicated filepath.

        The cogscm atomspace is saved in Scheme format.

        If `overwrite` is set to True (the default), then the file is
        cleared before being written.

        Return False if it fails, True otherwise. (WARNING: not
        supported yet, it always returns True).

        """

        with open(filepath, "w" if overwrite else "a") as file:
            file.write(atoms_to_scheme_str(self.cognitive_schematics) + "\n")
        return True

    def load_cogscms_atomspace(
        self, filepath: str, overwrite: bool = True, fast: bool = True
    ) -> bool:
        """Load the cogscm atomspace from the given filepath.

        The file should be in Scheme format.

        If `overwrite` is set to True (the default), then the
        cogscm atomspace is cleared before being written.

        If `fast` is set to True (the default), then the atomspace is
        loaded with OpenCog's built-in function for fast loading.
        Note however that in that case the file should not contain any
        Scheme code beside Atomese constructs.  (WARNING: only
        fast==True is support for now).

        Return False if it fails, True otherwise.  If successful it
        will automatically update the cognitive_schematics attribute.

        """

        success = load_atomspace(self.cogscms_atomspace, filepath, overwrite)
        if success:
            if overwrite:
                self.cognitive_schematics.clear()
            self.cognitive_schematics.update(atomspace_roots(self.cogscms_atomspace))
        return success


class MixtureModel:
    """Class holding a Mixture Model for action decision."""

    def __init__(self, action_space: set[Atom], prior_a: float, prior_b: float):
        # Set of actions the agent can do
        self.action_space = action_space

        # Prior alpha and beta of beta-distributions corresponding to
        # truth values
        self.prior_a = prior_a
        self.prior_b = prior_b

        # Total evidence count.  Negative means unset.
        self.data_set_size = -1.0

        ###################
        # User parameters #
        ###################

        # Parameter to control the complexity penalty over the
        # cognitive schematics. Ranges from 0, no penalty to +inf,
        # infinit penalty. Affect the calculation of the cognitive
        # schematic prior.
        self.complexity_penalty = 0.1

        # Parameter to estimate the length of a whole model given a
        # partial model + unexplained data. Ranges from 0 to 1, 0
        # being no compressiveness at all of the unexplained data, 1
        # being full compressiveness.
        self.compressiveness = 0.75

        # Add an unknown component for each action. For now its weight
        # is constant, delta, but ultimately is should be calculated
        # as a rest in the Solomonoff mixture.
        self.delta = 1.0e-5

        # Action selection can be influenced by the weight w of the
        # model where the first order probability p has been sampled
        # from, if so then the action that maximizes
        #
        # p * w^weight_influence
        #
        # is selected.  It is recommended to have weight_influence
        # range from 0 (weight has no influence) to 1 (weight is
        # linearly taken into account) but values beyond 1 can also be
        # used.
        self.weight_influence = 0.0

    def complexity(self, atom: Atom) -> int:
        """Return the count of all unique atoms in atom."""

        return len(get_uniq_atoms(atom))

    def prior(self, length: float) -> float:
        """Given the length of a model, calculate its prior.

        Specifically

        exp(-complexity_penalty*length)

        where complexity_penalty is a complexity penalty parameter (0
        for no penalty, +inf for infinit penalty), and length is the
        size of the model, the total number of atoms involved in its
        definition.

        The prior doesn't have to sum up to 1 because the probability
        estimates are normalized.

        """

        return math.exp(-self.complexity_penalty * length)

    def unexplained_data_size(self, cogscm: Atom) -> float:
        """Estimate the size of the unexplained data by cogscm.

        This is the number of observations that the antecedent of
        cogscm does not cover.

        """

        # Make sure data_set_size has been set
        assert 0 < self.data_set_size

        # Estimate of the size of unexplained data by cogscm
        return self.data_set_size - cogscm.tv.count

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

    def prior_estimate(self, cogscm: Atom) -> float:
        """Calculate the prior probability of cogscm."""

        agent_log.fine("prior_estimate(cogscm={})".format(atom_to_idstr(cogscm)))

        # Get the complexity (program size) of cogscm
        partial_complexity = self.complexity(cogscm)

        # Calculate the kolmogotov complexity estimate of the
        # data unexplained by cogscm
        unexplained_data_size = self.unexplained_data_size(cogscm)
        kestimate = self.kolmogorov_estimate(unexplained_data_size)

        agent_log.fine(
            "partial_complexity = {}, unexplained_data_size = {}, kestimate = {}".format(
                partial_complexity, unexplained_data_size, kestimate
            )
        )

        return self.prior(partial_complexity + kestimate)

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

    def weight(self, cogscm: Atom) -> float:
        """Calculate the weight of a cogscm for Model Bayesian Averaging.

        The calculation is based on

        https://github.com/ngeiswei/papers/blob/master/PartialBetaOperatorInduction/PartialBetaOperatorInduction.pdf

        It assumes self.data_set_size has been properly set.

        """

        return self.prior_estimate(cogscm) * self.beta_factor(cogscm)

    def weighted_probability(self, w8: float, pblt: float) -> float:
        """Return the weighted probability using X."""

        return pblt * w8 ** self.weight_influence

    def infer_data_set_size(self, cogscms: list[Atom], total_count: int) -> None:
        """Infer the data set size (universe size).

        For now it uses the max of the total_count and the max count
        of all cognitive schematics (to work around the fact that we
        may not have a complete model).

        """

        max_count = 0.0
        if 0 < len(cogscms):
            max_count = max(cogscms, key=lambda x: x.tv.count).tv.count
        self.data_set_size = max(max_count, float(total_count))

    def mk_mxmdl(
        self, valid_cogscms: list[Atom], total_count: int
    ) -> omdict[Atom, tuple[float, Atom]]:
        # Infer the size of the complete data set, including all
        # observations used to build the mixture model.  It needs to
        # be called before calling MixtureModel.weight
        self.infer_data_set_size(valid_cogscms, total_count)

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

    def thompson_sample(
        self, mxmdl: omdict[Atom, tuple[float, Atom]]
    ) -> tuple[Atom, float, Atom, float, float]:
        """Perform Thompson sampling over the mixture model.

        Meaning, for each action

        1. Select a TV according to its likelihood (derived from its
        cognitive schematic).

        2. From that TV, sample its second order distribution to
        obtain a first order probability variate

        3. Calculate the weighted probability

        Then return a tuple with

        1. the selected action
        2. the weight of the model it is coming from
        3. its probability estimate of success

        that maximizes the weighted probability.

        """

        # 1. For each action select its TV according its weight
        act_w8d_cogscm_seq = [
            (action, weighted_sampling(w8d_cogscms))
            for (action, w8d_cogscms) in mxmdl.listitems()
        ]
        agent_log.fine(
            "act_w8d_cogscm_seq:\n{}".format(
                self.act_w8d_cogscm_seq_to_str(act_w8d_cogscm_seq)
            )
        )

        # 2. For each action select its first order probability given its tv
        act_w8_cogscm_pblt_seq = [
            (
                action,
                w8_cogscm[0],
                w8_cogscm[1],
                tv_rv(get_cogscm_tv(w8_cogscm[1]), self.prior_a, self.prior_b),
            )
            for (action, w8_cogscm) in act_w8d_cogscm_seq
        ]
        agent_log.fine(
            "act_w8_cogscm_pblt_seq:\n{}".format(
                self.act_w8_cogscm_pblt_seq_to_str(act_w8_cogscm_pblt_seq)
            )
        )

        # 3. For each action calculate the weighted probability
        act_w8_cogscm_w8d_pblt_seq = [
            (action, w8, cogscm, p, self.weighted_probability(w8, p))
            for (action, w8, cogscm, p) in act_w8_cogscm_pblt_seq
        ]
        agent_log.fine(
            "act_w8_cogscm_w8d_plbt_seq:\n{}".format(
                self.act_w8_cogscm_w8d_pblt_seq_to_str(act_w8_cogscm_w8d_pblt_seq)
            )
        )

        # Return an action with highest probability of success (TODO: take
        # care of ties)
        return max(act_w8_cogscm_w8d_pblt_seq, key=lambda t: t[4])

    def mxmdl_to_str(
        self, mxmdl: omdict[Atom, tuple[float, Atom]], indent: str = ""
    ) -> str:
        """Pretty print the given mixture model of cogscms"""

        ss: list[str] = []
        for act_w8d_cogscms in mxmdl.listitems():
            action = act_w8d_cogscms[0]
            w8d_cogscms = act_w8d_cogscms[1]
            s = indent + to_human_readable_str(action)
            s += " [size={}]:\n".format(len(w8d_cogscms))
            s += self.w8d_cogscm_seq_to_str(w8d_cogscms, indent + "  ")
            ss.append(s)
        return "\n".join(ss)

    def w8d_cogscm_to_str(
        self, w8d_cogscm: tuple[float, Atom], indent: str = ""
    ) -> str:
        """Pretty print a pair (weight, cogscm)."""

        weight = w8d_cogscm[0]
        cogscm = w8d_cogscm[1]
        tv = get_cogscm_tv(cogscm)
        idstr = atom_to_idstr(cogscm)
        s = "(weight={}, tv={}, id={})".format(weight, tv, idstr)
        return s

    def w8d_cogscm_seq_to_str(
        self, w8d_cogscm_seq: list[tuple[float, Atom]], indent: str = ""
    ) -> str:
        """Pretty print the given list of weighted cogscms"""

        w8d_cogscm_seq_sorted = sorted(w8d_cogscm_seq, key=lambda x: x[0], reverse=True)

        ss: list[str] = []
        for w8d_cogscm in w8d_cogscm_seq_sorted:
            ss.append(indent + self.w8d_cogscm_to_str(w8d_cogscm, indent + "  "))
        return "\n".join(ss)

    def act_pblt_to_str(self, act_pblt: tuple[Atom, float], indent: str = "") -> str:
        action = act_pblt[0]
        pblt = act_pblt[1]
        return indent + "{}: {}".format(to_human_readable_str(action), pblt)

    def act_pblt_seq_to_str(
        self, act_pblt_seq: list[tuple[Atom, float]], indent: str = ""
    ) -> str:
        """Pretty print a list of pairs (action, probability)."""

        return "\n".join(
            [indent + self.act_pblt_to_str(act_pblt) for act_pblt in act_pblt_seq]
        )

    def act_w8_pblt_to_str(
        self, act_w8_pblt: tuple[Atom, float, float], indent: str = ""
    ) -> str:
        """Pretty print a triple (action, weight, probability)."""

        action = act_w8_pblt[0]
        weight = act_w8_pblt[1]
        pblt = act_w8_pblt[2]
        return indent + "{}: (weight={}, probability={})".format(
            to_human_readable_str(action), weight, pblt
        )

    def act_w8_pblt_seq_to_str(
        self, act_w8_pblt_seq: list[tuple[Atom, float, float]], indent: str = ""
    ) -> str:
        """Pretty print a list of triple (action, weight, probability)."""

        return "\n".join(
            [
                indent + self.act_w8_pblt_to_str(act_w8_pblt)
                for act_w8_pblt in act_w8_pblt_seq
            ]
        )

    def act_w8_cogscm_pblt_to_str(
        self, act_w8_cogscm_pblt: tuple[Atom, float, Atom, float], indent: str = ""
    ) -> str:
        """Pretty print a triple (action, weight, probability)."""

        action = act_w8_cogscm_pblt[0]
        weight = act_w8_cogscm_pblt[1]
        cogscm = act_w8_cogscm_pblt[2]
        pblt = act_w8_cogscm_pblt[3]
        return indent + "{}: (weight={}, cogscm={}, probability={})".format(
            to_human_readable_str(action), weight, atom_to_idstr(cogscm), pblt
        )

    def act_w8_cogscm_pblt_seq_to_str(
        self,
        act_w8_cogscm_pblt_seq: list[tuple[Atom, float, Atom, float]],
        indent: str = "",
    ) -> str:
        """Pretty print a list of tuple (action, weight, cogscm, probability)."""

        return "\n".join(
            [
                indent + self.act_w8_cogscm_pblt_to_str(act_w8_cogscm_pblt)
                for act_w8_cogscm_pblt in act_w8_cogscm_pblt_seq
            ]
        )

    def act_w8_w8d_pblt_to_str(
        self, act_w8_w8d_pblt: tuple[Atom, float, float, float], indent: str = ""
    ) -> str:
        """Pretty print a quadruple (action, weight, probability, weighted probability)."""

        action = act_w8_w8d_pblt[0]
        weight = act_w8_w8d_pblt[1]
        pblt = act_w8_w8d_pblt[2]
        w8d_pblt = act_w8_w8d_pblt[3]
        return (
            indent
            + "{}: (weight={}, probability={}, weighted probability={})".format(
                to_human_readable_str(action), weight, pblt, w8d_pblt
            )
        )

    def act_w8_w8d_pblt_seq_to_str(
        self,
        act_w8_w8d_pblt_seq: list[tuple[Atom, float, float, float]],
        indent: str = "",
    ) -> str:
        """Pretty print a list of quadruples (action, weight, probability, weighted probability)."""

        return "\n".join(
            [
                indent + self.act_w8_w8d_pblt_to_str(act_w8_w8d_pblt)
                for act_w8_w8d_pblt in act_w8_w8d_pblt_seq
            ]
        )

    def act_w8_cogscm_w8d_pblt_to_str(
        self,
        act_w8_cogscm_w8d_pblt: tuple[Atom, float, Atom, float, float],
        indent: str = "",
    ) -> str:
        """Pretty print a tuple (action, weight, cogscm, probability, weighted probability)."""

        action = act_w8_cogscm_w8d_pblt[0]
        weight = act_w8_cogscm_w8d_pblt[1]
        cogscm = act_w8_cogscm_w8d_pblt[2]
        pblt = act_w8_cogscm_w8d_pblt[3]
        w8d_pblt = act_w8_cogscm_w8d_pblt[4]
        return indent + "{}: (weight={}, cogscm={}, probability={}, weighted probability={})".format(
            to_human_readable_str(action),
            weight,
            atom_to_idstr(cogscm),
            pblt,
            w8d_pblt,
        )

    def act_w8_cogscm_w8d_pblt_seq_to_str(
        self,
        act_w8_cogscm_w8d_pblt_seq: list[tuple[Atom, float, Atom, float, float]],
        indent: str = "",
    ) -> str:
        """Pretty print a list of tuples (action, weight, cogscm, probability, weighted probability)."""

        return "\n".join(
            [
                indent + self.act_w8_cogscm_w8d_pblt_to_str(act_w8_cogscm_w8d_pblt)
                for act_w8_cogscm_w8d_pblt in act_w8_cogscm_w8d_pblt_seq
            ]
        )

    def act_w8d_cogscm_to_str(
        self, act_w8d_cogscm: tuple[Atom, tuple[float, Atom]], indent: str = ""
    ) -> str:
        """Pretty print a pair (action, (weight, cogscm))."""

        action = act_w8d_cogscm[0]
        w8d_cogscm = act_w8d_cogscm[1]
        s = (
            indent
            + to_human_readable_str(action)
            + ": "
            + self.w8d_cogscm_to_str(w8d_cogscm)
        )
        return s

    def act_w8d_cogscm_seq_to_str(
        self,
        act_w8d_cogscm_seq: list[tuple[Atom, tuple[float, Atom]]],
        indent: str = "",
    ) -> str:
        """Pretty print a list of pairs (action, (weight, cogscm))."""

        return "\n".join(
            [
                indent + self.act_w8d_cogscm_to_str(act_w8d_cogscm)
                for act_w8d_cogscm in act_w8d_cogscm_seq
            ]
        )
