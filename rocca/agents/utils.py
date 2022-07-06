# Util functions used by GymAgent

##############
# Initialize #
##############

# Python
import random
import math
from typing import Any
from functools import cmp_to_key

# SciPy
import scipy.stats as st

# OpenCog
from opencog.atomspace import (
    Atom,
    AtomSpace,
    get_type,
    get_type_name,
    is_a,
    types,
    createTruthValue,
)
from opencog.execute import execute_atom
from opencog.logger import create_logger
from opencog.pln import ZLink, SLink
from opencog.scheme import scheme_eval, scheme_eval_h, load_scm
from opencog.spacetime import AtTimeLink, TimeNode
from opencog.type_constructors import (
    VariableSet,
    AndLink,
    PresentLink,
    IsClosedLink,
    IsTrueLink,
    SatisfactionLink,
    TruthValue,
)
from opencog.utilities import get_free_variables, load_file

from requests import post, get, delete
import json
import re
import time

#############
# Constants #
#############

TRUE_TV = createTruthValue(1, 1)
DEFAULT_TV = createTruthValue(1, 0)

#############
# Variables #
#############

agent_log = create_logger("opencog.log")
agent_log.set_component("Agent")

#############
# Functions #
#############


def add_to_atomspace(atoms: set[Atom] | list[Atom], atomspace: AtomSpace) -> None:
    """Add all atoms to the atomspace."""

    for atom in atoms:
        atomspace.add_atom(atom)


def copy_atomspace(src: AtomSpace, dst: AtomSpace) -> None:
    """Copy the content of src into the dst.

    The copy does not clean dst.

    """

    add_to_atomspace(atomspace_roots(src), dst)


def fetch_cogscms(atomspace: AtomSpace) -> set[Atom]:
    """Fetch all cognitive schematics from an given atomspace."""

    pit = get_type("BackPredictiveImplicationScopeLink")
    return set(atomspace.get_atoms_by_type(pit))


def has_non_null_confidence(atom: Atom) -> bool:
    """Return True iff the given atom has a confidence above 0."""

    return 0 < atom.tv.confidence


def has_mean_geq(atom: Atom, mean: float) -> bool:
    """Return True iff `atom` has a mean greater than or equal to `mean`."""

    return mean <= atom.tv.mean


def is_true(atom: Atom) -> bool:
    """Return True iff the given has a tv equal to TRUE_TV."""

    return atom.tv == TRUE_TV


def shannon_entropy(atom: Atom, prior_a: float = 1.0, prior_b: float = 1.0) -> float:
    """Return the shannon entropy of `atom`.

    The shannon entropy is calculated based on the mean of the beta
    distribution corresponding to the truth value of `atom`.

    """

    # Avoid division by zero
    s = atom.tv.mean
    c = atom.tv.confidence
    if (math.isclose(s, 0) or math.isclose(s, 1)) and math.isclose(c, 1):
        return 0

    # Otherwise, calculate the actual Shannon entropy
    mean = tv_to_beta(atom.tv, prior_a, prior_b).mean()
    return st.entropy([mean, 1.0 - mean], base=2)


def differential_entropy(
    atom: Atom, prior_a: float = 1.0, prior_b: float = 1.0
) -> float:
    """Return the differential entropy of `atom`.

    The differential entropy is calculated based on the beta
    distribution corresponding to the truth value of `atom`.

    See
    https://en.wikipedia.org/wiki/Beta_distribution#Quantities_of_information_(entropy)
    for more information.

    """

    # Avoid division by zero
    s = atom.tv.mean
    c = atom.tv.confidence
    if (math.isclose(s, 0) or math.isclose(s, 1)) and math.isclose(c, 1):
        return -float("inf")

    # Otherwise, calculate the actual differential entropy
    return tv_to_beta(atom.tv, prior_a, prior_b).entropy()


def count_to_confidence(count: int) -> float:
    """Convert TV count to confidence."""

    K = 800.0
    return float(count) / (float(count) + K)


def tv_to_beta(
    tv: TruthValue, prior_a: float = 1.0, prior_b: float = 1.0
) -> st.rv_continuous:
    """Convert a truth value to a beta distribution.

    Given a truth value, return the beta distribution that best fits
    it.  Two optional parameters are provided to set the prior of the
    beta-distribution, the default values are prior_a=1 and prior_b=1
    corresponding to the Bayesian prior.

    """

    count = tv.count
    pos_count = count * tv.mean  # the mean is actually the mode
    a = prior_a + pos_count
    b = prior_b + count - pos_count
    return st.beta(a, b)


def tv_to_alpha_param(
    tv: TruthValue, prior_a: float = 1.0, prior_b: float = 1.0
) -> float:
    """Return the alpha parameter of a TV's beta-distribution."""

    count = tv.count
    pos_count = count * tv.mean  # the mean is actually the mode
    return prior_a + pos_count


def tv_to_beta_param(
    tv: TruthValue, prior_a: float = 1.0, prior_b: float = 1.0
) -> float:
    """Return the beta parameter of a TV's beta-distribution."""

    count = tv.count
    pos_count = count * tv.mean  # the mean is actually the mode
    return prior_b + count - pos_count


def tv_rv(tv: TruthValue, prior_a: float = 1, prior_b: float = 1) -> float:
    """Return a first order probability variate of a truth value.

    Return a random variate of the beta distribution
    representing the second order distribution of the truth value,
    which represents a first order probability.

    rv stands for Random Variate.

    """

    betadist = tv_to_beta(tv, prior_a, prior_b)
    return betadist.rvs()


def atom_to_idstr(atom: Atom) -> str:
    return atom.id_string() if atom else "None"


def get_cogscm_tv(cogscm: Atom) -> TruthValue:
    """Return the Truth Value of a cogscm or the default if it is None"""

    return cogscm.tv if cogscm else DEFAULT_TV


def weighted_sampling(weighted_items: list[tuple[float, Any]]) -> tuple[float, Any]:
    """Given list of pairs (weight, element) weight-randomly select an element.

    Return pair (w, e) of the weight associated to the selected element.

    """

    w8s = [weight for (weight, _) in weighted_items]
    return random.choices(weighted_items, weights=w8s)[0]


def weighted_average_tv(weighted_tvs):
    """Given a list of pairs (weight, tv) return the weighted average tv.

    Return a Simple Truth Value with mean and variance equivalent to
    that of a distributional truth value built with the weighted
    average of the input tvs.

    """

    # TODO: actually not needed
    return None


def get_vardecl(cogscm: Atom) -> Atom:
    """Extract the vardecl of a cognitive schematic.

    Given a cognitive schematic of that format

    BackPredictiveImplicationScope <tv>
      <vardecl>
      <expiry>
      And (or SimultaneousAnd?)
        <context>
        Execution
          <action>
          <input> [optional]
          <output> [optional]
      <goal>

    return <vardecl>.

    If the cognitive schematic is not a scope then return an empty
    VariableSet.

    """

    return cogscm.out[0] if is_scope(cogscm) else VariableSet()


def vardecl_size(vardecl: Atom) -> int:
    """Return the number of variables in a variable declaration."""

    is_variable_collection = is_variable_list(vardecl) or is_variable_set(vardecl)
    return vardecl.arity if is_variable_collection else 1


def has_variables_leq(cogscm: Atom, vc: int) -> bool:
    """Return True iff `cogscm` has a number of variables less than or equal to `vc`."""

    return vardecl_size(get_vardecl(cogscm)) <= vc


def is_ordered(atom: Atom) -> bool:
    """Return true iff the atom inherits from the OrderedLink type."""

    return is_a(atom.type, types.OrderedLink)


def is_unordered(atom: Atom) -> bool:
    """Return true iff the atom inherits from the UnorderedLink type."""

    return is_a(atom.type, types.UnorderedLink)


def is_virtual(clause: Atom) -> bool:
    """Return true iff the clause is virtual.

    For instance

    (GreaterThanLink (NumberNode "1") (NumberNode "2"))

    is virtual because it gets evaluated on the fly as opposed to be
    query against the atomspace.

    """

    # TODO: can be simplified with clause.is_a
    return is_a(clause.type, types.VirtualLink)


def get_context(cogscm: Atom) -> tuple[list[Atom], list[Atom]]:
    """Extract the context of a cognitive schematic.

    Since the context can be multiple clauses, virtual and
    non-virtual, it outputs a pair of two lists

    (present-clauses, virtual-clauses)

    For instance given a cognitive schematic of that format

    BackPredictiveImplicationScope <tv>
      <vardecl>
      <expiry>
      And
        <context>
        <execution>
      <goal>

    return ([<context>], []).

    Another example, given a cognitive schematic of that format

    BackPredictiveImplicationScope <tv>
      <vardecl>
      <expiry>
      SequentialAnd
        And
          <context>
          <execution>
        ...
      <goal>

    return ([<context>], [])

    """

    # Grab all clauses pertaining to context
    clauses = get_t0_clauses(get_antecedent(cogscm))
    no_exec_clauses = [x for x in clauses if not is_execution(x)]

    # Split them into present and virtual clauses
    present_clauses = [x for x in no_exec_clauses if not is_virtual(x)]
    virtual_clauses = [x for x in no_exec_clauses if is_virtual(x)]

    # Return pair of present and virtual clauses
    return (present_clauses, virtual_clauses)


def is_list(atom: Atom) -> bool:
    """Return True iff the atom is a ListLink."""

    return is_a(atom.type, types.ListLink)


def is_variable(atom: Atom) -> bool:
    """Return True iff the atom is a variable node."""

    return is_a(atom.type, types.VariableNode)


def is_variable_list(atom: Atom) -> bool:
    """Return True iff the atom is a VariableList."""

    return is_a(atom.type, types.VariableList)


def is_variable_set(atom: Atom) -> bool:
    """Return True iff the atom is a VariableSet."""

    return is_a(atom.type, types.VariableSet)


def is_empty_link(atom: Atom) -> bool:
    """Return True iff the atom is a link with empty outgoing set."""

    return atom.is_link() and atom.out == []


def is_scope(atom: Atom) -> bool:
    """Return True iff the atom is a scope link."""

    return is_a(atom.type, types.ScopeLink)


def is_predictive_implication(atom: Atom) -> bool:
    """Return True iff the atom is a predictive implication link."""

    return is_a(atom.type, get_type("BackPredictiveImplicationLink"))


def is_predictive_implication_scope(atom: Atom) -> bool:
    """Return True iff the atom is a predictive implication scope link."""

    return is_a(atom.type, get_type("BackPredictiveImplicationScopeLink"))


def is_and(atom: Atom) -> bool:
    """Return True iff the atom is an and link."""

    return is_a(atom.type, types.AndLink)


def is_sequential_and(atom: Atom) -> bool:
    """Return True iff atom is a sequential and.

    Also for now we use BackSequentialAndLink.

    """

    return is_a(atom.type, get_type("BackSequentialAndLink"))


def is_evaluation(atom: Atom) -> bool:
    """Return True iff the atom is an EvaluationLink."""

    return is_a(atom.type, types.EvaluationLink)


def is_execution(atom: Atom) -> bool:
    """Return True iff the atom is an ExecutionLink."""

    return is_a(atom.type, types.ExecutionLink)


def is_at_time(atom: Atom) -> bool:
    """Return True iff the atom in an AtTimeLink."""

    return is_a(atom.type, get_type("AtTimeLink"))


def is_Z(atom: Atom) -> bool:
    """Return True iff the atom is Z."""

    return atom.type == get_type("ZLink")


def is_S(atom: Atom) -> bool:
    """Return True iff the atom is S ..."""

    return atom.type == get_type("SLink")


def maybe_and(clauses: list[Atom]) -> Atom:
    """Wrap an And if multiple clauses, otherwise return the only one."""

    return AndLink(*clauses) if 1 < len(clauses) else clauses[0]


def get_antecedent(atom: Atom) -> Atom | None:
    """Return the antecedent of a temporal atom.

    For instance is the cognitive schematics is represented by

    BackPredictiveImplicationScope <tv>
      <vardecl>
      <expiry>
      <antecedent>
      <succedent>

    it returns <antecedent>

    """

    if is_predictive_implication_scope(atom):
        return atom.out[2]
    if is_predictive_implication(atom):
        return atom.out[1]
    if is_sequential_and(atom):
        return atom.out[1]
    return None


def get_succedent(atom: Atom):  # Atom | None
    """Return the succedent of a temporal atom.

    For instance is the cognitive schematics is represented by

    BackSequentialAnd <tv>
      <lag>
      <antecedent>
      <succedent>

    it returns <succedent>.

    If it is

    BackPredictiveImplicationScope <tv>
      <vardecl>
      <lag>
      <antecedent>
      <succedent>

    it also returns <succedent>.

    Note that for predictive implication, succedent is also called
    consequent.  However it does make much sense to call a succedent a
    consequent for a SequentialAnd, or another link that is not an
    implication, thus the use of the generic term succedent.

    """

    if is_predictive_implication_scope(atom):
        return atom.out[3]
    if is_predictive_implication(atom):
        return atom.out[2]
    if is_sequential_and(atom):
        return atom.out[2]
    return None


def get_lag(atom: Atom) -> int:
    """Given an temporal atom, return its lag component.

    For instance if it is a BackPredictiveImplicationScope

    BackPredictiveImplicationScope
      <vardecl>
      <lag>
      <antecedent>
      <succedent>

    return <lag>

    If it is a SequentialAnd

    SequentialAnd
      <lag>
      <A>
      <B>

    return <lag>

    """

    if is_predictive_implication_scope(atom):
        return to_int(atom.out[1])
    if is_predictive_implication(atom):
        return to_int(atom.out[0])
    if is_sequential_and(atom):
        return to_int(atom.out[0])
    return 0


def get_t0_clauses(antecedent: Atom) -> list[Atom]:
    """Return the list of clauses occuring at initial time.

    For instance if the cognitive schematics has the following format

    BackPredictiveImplicationScope <tv>
      <vardecl>
      <lag>
      SequentialAnd
        And
          <context-1>
          ...
          <context-n>
          Execution
            <action>
            <input> [optional]
            <output> [optional]
        ...
      <goal>

    it returns

    [<context-1>, ..., <context-n>, Execution ...]

    because they are all the clauses happening the initial time of the
    predictive implication.

    """

    antecedent
    if is_and(antecedent):
        return antecedent.out
    if is_sequential_and(antecedent):
        return get_t0_clauses(antecedent.out[1])
    else:
        return [antecedent]


def has_all_variables_in_antecedent(cogscm: Atom) -> bool:
    """Return True iff all variables are in the antecedent."""

    if is_scope(cogscm):
        vardecl_vars = set(get_free_variables(get_vardecl(cogscm)))
        antecedent_vars = set(get_free_variables(get_antecedent(cogscm)))
        return vardecl_vars == antecedent_vars
    else:
        return True


def get_free_variables_of_atoms(atoms: Atom) -> set[Atom]:
    """Get the set of all free variables in all atoms."""

    return set().union(*(set(get_free_variables(atom)) for atom in atoms))


def get_times(timed_atoms: list[Atom]) -> set[Atom]:
    """Given a list of timestamped clauses, return a set of all times."""

    if timed_atoms == []:
        return set()
    return set.union(set([get_time(timed_atoms[0])]), get_times(timed_atoms[1:]))


def get_events(timed_atoms: set[Atom] | list[Atom]) -> list[Atom]:
    """Given a container of timestamped clauses, return a list of all events."""

    return [get_event(ta) for ta in timed_atoms]


def get_latest_time(timed_clauses: list[Atom]) -> Atom:
    """Given a list of timestamped clauses, return the latest timestamp."""

    if timed_clauses == []:
        return ZLink()
    return nat_max(get_time(timed_clauses[0]), get_latest_time(timed_clauses[1:]))


def get_latest_clauses(timed_clauses: list[Atom]) -> list[Atom]:
    """Given a list of timestamped clauses, return the latest clauses.

    For instance if the timestamped clauses are

    [AtTime(A, T), AtTime(B, S(T)), AtTime(C, S(S(T))), AtTime(D, S(S(T)))]

    return

    [AtTime(C, S(S(T))), AtTime(D, S(S(T)))]

    """

    lt = get_latest_time(timed_clauses)
    return [tc for tc in timed_clauses if get_time(tc) == lt]


def get_early_clauses(timed_clauses: list[Atom]) -> list[Atom]:
    """Return all clauses that are not the latest."""

    lcs = set(get_latest_clauses(timed_clauses))
    return list(set(timed_clauses).difference(lcs))


def get_total_lag(atom: Atom) -> int:
    """Return the total lag between earliest and lastest subatoms of atom.

    For instance if the atom is

    BackPredictiveImplicationScope <tv>
      <vardecl>
      <lag-1>
      SequentialAnd
        <lag-2>
        SequentialAnd
          <lag-3>
          And
            <context>
            <execution>
          <other-execution>
      <goal>

    return lag-1 + lag-2 + lag3

    """

    lag = get_lag(atom)
    ant = get_antecedent(atom)
    tlg = lag + (get_total_lag(ant) if ant else 0)
    return tlg


def get_t0_execution(cogscm: Atom) -> Atom:
    """Extract the action of a cognitive schematic.

    Given a cognitive schematic of that formats

    BackPredictiveImplicationScope <tv>
      <vardecl>
      <expiry>
      And (or SimultaneousAnd?)
        <context>
        Execution
          <action>
          <input> [optional]
          <output> [optional]
      <goal>

    extract
        Execution
          <action>
          <input> [optional]
          <output> [optional]

    """

    cnjs = get_t0_clauses(get_antecedent(cogscm))
    execution = next(x for x in cnjs if is_execution(x))
    return execution


def get_context_actual_truth(atomspace: AtomSpace, cogscm: Atom, i: int) -> TruthValue:
    """Calculate tv of the context of cognitive schematic cogscm at time i.

    Given a cognitive schematic of that format

    BackPredictiveImplicationScope <tv>
      <vardecl>
      <lag>
      And (or SimultaneousAnd?)
        <context>
        Execution
          <action>
          <input> [optional]
          <output> [optional]
      <goal>

    calculate the truth value of <context> at time i, where the
    variables have been properly instantiated.

    For now the calculation is crisp, either a context is completely
    true (stv 1 1) or completely false (stv 0 1).

    """

    agent_log.fine(
        "get_context_actual_truth(atomspace={}, cogscm={}, i={}".format(
            atomspace, atom_to_scheme_str(cogscm), i
        )
    )

    # Build and run a query to check if the context is true
    vardecl = get_vardecl(cogscm)
    present_clauses, virtual_clauses = get_context(cogscm)
    stamped_present_clauses = [timestamp(pc, i) for pc in present_clauses]
    body = AndLink(
        PresentLink(*stamped_present_clauses),
        IsClosedLink(*stamped_present_clauses),
        IsTrueLink(*stamped_present_clauses),
        *virtual_clauses
    )
    query = SatisfactionLink(vardecl, body)
    agent_log.fine("query = {}".format(query))
    tv = execute_atom(atomspace, query)
    agent_log.fine("tv = {}".format(tv))
    return tv


def get_event(timed_atom: Atom) -> Atom:
    """Return the event in a clause that is a timestamped event

    For instance if clause is

    AtTime
      <event>
      <time>

    return <event>

    """

    return timed_atom.out[0]


def get_time(timed_atom: Atom) -> Atom:
    """Given (AtTime A T) return T."""

    return timed_atom.out[1]


def timestamp(atom: Atom, i: int, tv: TruthValue = None, nat: bool = True) -> Atom:
    """Timestamp a given atom.  Optionally set its TV

    AtTimeLink <tv>               # if tv is provided
      <atom>
      TimeNode <str(i)>

    if nat is True it uses a Natural instead of TimeNode (see to_nat).

    """

    time = to_nat(i) if nat else TimeNode(str(i))
    return AtTimeLink(atom, time, tv=tv)


def nat_max(n: Atom, m: Atom) -> Atom:

    """Return the max between two naturals (including if they wrap variables)."""

    if is_Z(n) or is_variable(n):
        return m
    if is_Z(m) or is_variable(m):
        return n
    return SLink(nat_max(n.out[0], m.out[0]))


def to_nat(i: int) -> Atom:
    """Convert i to a Natural.

    For instance if i = 3, then it returns

    S S S Z

    """

    # return ZLink() if i == 0 else SLink(to_nat(i - 1))
    ret = ZLink()
    for i in range(0, i):
        ret = SLink(ret)

    return ret


def lag_to_nat(i: int, T: Atom) -> Atom:
    """Given an int i and T, return as many SLinks wrapping around T.

    For instance if i=3 and T=VariableNode("$T") return

    S(S(S(T)))

    """

    # return T if i == 0 else SLink(lag_to_nat(i - 1, T))
    ret = T
    for i in range(0, i):
        ret = SLink(ret)

    return ret


def to_int(n: Atom) -> int:

    """Convert n to an Int.

    For instance if n = S S S Z, then it returns 3.

    """

    # Optimize
    #
    #    if is_Z(n):
    #        return 0
    #    if is_S(n):
    #        return 1 + to_int(n.out[0])
    #
    # to avoid stack overflow

    ret = 0
    link = n

    while is_S(link):
        ret += 1
        link = link.out[0]

    return ret


def to_scheme_str(vs: Any) -> str:
    """Takes a python value and convert it to a scheme value string"""

    if vs == True:
        return "#t"
    elif vs == False:
        return "#f"
    else:
        return str(vs)


# TODO: this should really be moved to the atomspace python bindings
def atomspace_roots(atomspace: AtomSpace) -> set[Atom]:
    """Return all roots of an atomspace."""

    return set(scheme_eval_h(atomspace, "(List (cog-get-all-roots))").out)


def atomspace_to_str(atomspace: AtomSpace) -> str:
    """Takes an atomspace and return its content as a string"""

    roots = atomspace_roots(atomspace)
    return "\n".join([root.long_string() for root in roots])


def agent_log_atomspace(
    atomspace: AtomSpace, level: str = "fine", msg_prefix: str = "atomspace"
) -> None:
    """Takes an atomspace and log its content (with size and address)"""

    li = agent_log.string_as_level(level)
    msg = "{} [address={}, size={}]:\n{}".format(
        msg_prefix, atomspace, len(atomspace), atomspace_to_str(atomspace)
    )
    agent_log.log(li, msg)


def get_uniq_atoms(atom: Atom) -> set[Atom]:
    """Return the set of all unique atoms in atom."""

    result = {atom}

    # Recursive cases
    if atom.is_link():
        for o in atom.out:
            result.update(get_uniq_atoms(o))
        return result

    # Base cases (atom is a node)
    return result


def syntax_lt(a1: Atom, a2: Atom) -> bool:
    """Custom less-than function for unordered links.

    It is used by to_human_readable_str to place the do(Action) to the
    right, so that for instance

    do(Eat) ∧ AgentPosition(RightSquare) ↝ Reward(1)

    becomes

    AgentPosition(RightSquare) ∧ do(Eat) ↝ Reward(1)

    which makes it easier to read and search.

    """

    return (not is_execution(a1) and is_execution(a2)) or a1 < a2


def syntax_cmp(a1: Atom, a2: Atom) -> int:
    """Compare function based on syntax_lt for to_human_readable_str."""

    if a1 == a2:
        return 0
    elif syntax_lt(a1, a2):
        return -1
    else:
        return 1


def syntax_precede(a1: Atom, a2: Atom) -> bool:

    """Return true iff a1 syntactically precedes a2.

    This function is used by to_human_readable_str to minimize the
    number of parenthesis.

    Precedence order is as follows

    1. Variable/Number/Concept/Predicate/Schema
    2. Evaluation/Execution/GreaterThan
    3. AtTime/Member
    4. Not
    5. And/Or
    6. SequentialAnd
    7. PredictiveImplication

    Thus

    op_precede(And(...), SequentialAnd(...))

    returns

    True

    """

    precedence = {
        types.VariableNode: 1,
        types.NumberNode: 1,
        types.ConceptNode: 1,
        types.PredicateNode: 1,
        types.SchemaNode: 1,
        types.TimeNode: 1,
        types.EvaluationLink: 2,
        types.ExecutionLink: 2,
        types.GreaterThanLink: 2,
        types.AtTimeLink: 3,
        types.MemberLink: 3,
        types.NotLink: 4,
        types.AndLink: 5,
        types.OrLink: 5,
        get_type("BackSequentialAndLink"): 6,
        get_type("BackPredictiveImplicationScopeLink"): 7,
    }

    return precedence[a1.type] < precedence[a2.type]


# TODO: add type annotation on ty
def type_to_human_readable_str(ty) -> str:
    """Convert atom type to human readable character.

    The conversion goes as follows

    AtTime                       -> @
    Member                       -> ∈
    And                          -> ∧
    Or                           -> ∨
    Not                          -> ¬
    GreaterThan                  -> >
    SequentialAnd                -> ⩘
    SequentialOr                 -> ⩗
    PredictiveImplication[Scope] -> ↝
    Execution                    -> do

    """

    to_hrs = {
        types.AtTimeLink: "@",
        types.MemberLink: "∈",
        types.NotLink: "¬",
        types.AndLink: "∧",
        types.OrLink: "∨",
        types.GreaterThanLink: ">",
        types.ExecutionLink: "do",
        get_type("BackSequentialAndLink"): "⩘",
        get_type("BackSequentialOrLink"): "⩗",
        get_type("BackPredictiveImplicationScope"): "↝",
        get_type("BackPredictiveImplicationScopeLink"): "↝",
    }

    return to_hrs[ty]


def to_human_readable_str(atom: Atom, parenthesis: bool = False) -> str:
    """Convert an atom into a compact human readable form.

    For instance, the timestamped perceptum

    (AtTime (stv 1 1)
      (EvaluationLink
        (PredicateNode "outside")
        (ListLink
          (ConceptNode "self")
          (ConceptNode "house")))
      (SLink
        (SLink
          (ZLink))))

    returns

    "outside(self, house) @ 2"

    Another example, the cognitive schematic

    (BackPredictiveImplicationScopeLink (stv 1 0.00990099)
      (VariableSet)
      (SLink
        (ZLink))
      (AndLink (stv 0.16 0.0588235)
        (EvaluationLink (stv 0.6 0.0588235)
          (PredicateNode "outside")
          (ListLink
            (ConceptNode "self")
            (ConceptNode "house")))
        (ExecutionLink
          (SchemaNode "go_to_key")))
      (EvaluationLink (stv 0.26 0.0588235)
        (PredicateNode "hold")
        (ListLink
          (ConceptNode "self")
          (ConceptNode "key"))))

    returns

    "outside(self, house) ∧ do(go_to_key) ↝ hold(self, key)"

    Note that for now lags and truth values are ignored in that
    cognitive schematics compact human readable form.

    The optional parenthesis flag wraps the resulting expression with
    parenthesis if set to true (this mostly useful for the recursive
    calls).

    Precedence order is defined in the syntax_precede function, so that

    C ∧ A₁ ⩘ A₂ ↝ G

    is equivalent to

    ((C ∧ A₁) ⩘ A₂) ↝ G

    Additionally ⩘ is left-associative (due to being a
    BackSequentialAnd, we would probably want it to be
    right-associative if it were a ForeSequentialAnd) so that

    A₁ ⩘ A₂ ⩘ A₃

    is equivalent to

    (A₁ ⩘ A₂) ⩘ A₃

    TODO: support action with arguments.

    """

    ##############
    # Base cases #
    ##############

    if atom.is_node():
        obj_str = atom.name
        obj_str = obj_str.replace(" ", "")  # Remove whitespace
        return "(" + obj_str + ")" if parenthesis else obj_str

    ###################
    # Recursive cases #
    ###################

    # By now atom must be a link
    assert atom.is_link()

    # By default the link is represented as an infix operator
    is_infix = True

    if (
        is_predictive_implication_scope(atom)
        or is_predictive_implication(atom)
        or is_sequential_and(atom)
    ):
        # If the atom is a sequential and, or a predictive
        # implication, ignore the lag and (possibly) the variable
        # declaration.
        out = [get_antecedent(atom), get_succedent(atom)]
        op_str = type_to_human_readable_str(atom.type)
    elif is_evaluation(atom):
        # If evaluation then the operator name is the predicate name,
        # and the arguments are the renaming outgoings (possibly
        # wrapped in a ListLink.
        op_str = atom.out[0].name
        op_str = op_str.replace(" ", "")  # Remove whitespace
        arg = atom.out[1]
        out = arg.out if is_list(arg) else [arg]
        is_infix = False
    elif is_execution(atom):
        op_str = type_to_human_readable_str(atom.type)
        out = [atom.out[0]]  # Only action with no argument is
        # supported
        # If execution then the operator is prefix
        is_infix = False
    elif is_at_time(atom):
        op_str = type_to_human_readable_str(atom.type)
        time = TimeNode(str(to_int(atom.out[1])))
        out = [atom.out[0], time]
    else:
        out = atom.out
        if is_unordered(atom):
            out = sorted(out, key=cmp_to_key(syntax_cmp))
        op_str = type_to_human_readable_str(atom.type)

    # Recursively convert outgoings to human readable strings, adding
    # parenthesis if necessary.
    wrap_parenthesis = lambda child, atom: is_infix and not syntax_precede(child, atom)
    hrs_out = [
        to_human_readable_str(child, wrap_parenthesis(child, atom)) for child in out
    ]

    # Construct the final string
    if is_infix:
        op_str = " " + op_str + " "  # Breathe
        final_str = op_str.join(hrs_out)
    else:
        # It is prefix then
        final_str = op_str + "(" + ", ".join(hrs_out) + ")"

    # Possibly wrap parenthesis around it and return
    return "(" + final_str + ")" if parenthesis else final_str


def atom_to_scheme_str(atom: Atom, only_id: bool = False) -> str:
    """Convert an atom to Scheme format string + human readable form.

    The human readable form of atom is commented out according to
    Scheme format, then on the next line its Scheme representation is
    appended.  If only_id is True, then only its ID, instead of the
    whole Scheme representation, is rendered.

    So for instance

    So for instance calling atom_to_scheme_str on

    (AtTime (stv 1 1)
      (EvaluationLink
        (PredicateNode "outside")
        (ListLink
          (ConceptNode "self")
          (ConceptNode "house")))
      (SLink
        (SLink
          (ZLink))))

    returns

    "
    ;; outside(self, house) @ 2
    (AtTimeLink (stv 1 1)
      (EvaluationLink
        (PredicateNode "outside") ; [72730412e28a734][2]
        (ListLink
          (ConceptNode "self") ; [40b11d11524bd751][2]
          (ConceptNode "house") ; [63eb9919f37daa5f][2]
        ) ; [aadca36fe9d1a468][2]
      ) ; [ca0c329fb1ab493b][2]
      (SLink
        (SLink
          (ZLink
          ) ; [800fbffffffe8ce4][2]
        ) ; [da5f815ba9d4009f][2]
      ) ; [f5363085cdea2ffe][2]
    ) ; [b37c7ec68e8c0c81][2]
    "

    """

    msg = ";; " + to_human_readable_str(atom) + "\n"
    msg += atom.id_string() if only_id else atom.long_string()
    return msg


def atoms_to_scheme_str(atoms: set[Atom] | list[Atom], only_id: bool = False) -> str:
    """Convert a collection of atoms to string in scheme format.

    Apply atom_to_scheme_str to a collection of atoms.

    """

    msgs = []
    for atom in atoms:
        msgs.append(atom_to_scheme_str(atom, only_id))
    return "\n".join(msgs)


def save_atomspace(atomspace: AtomSpace, filepath: str, overwrite: bool = True) -> bool:
    """Save the given atomspace at the indicated filepath.

    The atomspace is saved in Scheme format.

    If `overwrite` is set to True (the default), then the file is
    cleared before being written. (WARNING: not supported yet).

    Return False if it fails, True otherwise. (WARNING: not supported
    yet, always return True).

    """

    # TODO: support overwrite
    scheme_eval(atomspace, '(export-all-atoms "{}")'.format(filepath))
    # TODO: support status output
    return True


def load_atomspace(
    atomspace: AtomSpace, filepath: str, overwrite: bool = True, fast: bool = True
) -> bool:
    """Load the given atomspace from the given filepath.

    The file should be in Scheme format.

    If `overwrite` is set to True (the default), then the atomspace is
    cleared before being written.

    If `fast` is set to True (the default), then the atomspace is
    loaded with OpenCog's built-in function for fast loading.  Note
    however that in that case the file should not contain any Scheme
    code beside Atomese constructs.  (WARNING: only fast==True is
    support for now).

    Return False if it fails, True otherwise.  (WARNING: not fully
    support yet).

    """

    if overwrite:
        atomspace.clear()

    if fast:
        load_file(filepath, atomspace)
        return True  # TODO: wrap load_file in a try/catch
    else:
        # TODO: support not fast
        agent_log.warn("Normal loading not implemented!")
        return False


def pre_process_atoms(exp):
    # Processing empty variableset
    exp = re.sub(r"\(VariableSet\)", "", exp)
    exp = re.sub(r"\(VariableSet\s*\)", "", exp)
    # exp = re.sub(r'back', '', exp, flags=re.IGNORECASE)

    # Processing szlinks
    exp = re.sub(r";.*\n", "\n", exp, flags=re.MULTILINE)
    exp = re.sub(r"\(ZLink\s+", "(ZLink", exp)
    exp = re.sub(r"\)\s+\)", "))", exp)
    exp = re.sub(r"\)\s+\)", "))", exp)

    starting_indices_slinks = [m.start() for m in re.finditer("SLink", exp)]
    starting_indices_zlinks = [m.start() for m in re.finditer("ZLink", exp)]

    for i in range(len(starting_indices_zlinks)):
        # Get the index of its preceding zlink
        index_preceding_zlink = -1 if i == 0 else starting_indices_zlinks[i - 1]
        # Count the number of its parent links and the index of the start of its parent slink
        parent_count = 0
        index_parent_slink = 0
        for p in starting_indices_slinks:
            if p < starting_indices_zlinks[i] and p > index_preceding_zlink:
                parent_count += 1
                if parent_count == 1:
                    index_parent_slink = p
            if p > starting_indices_zlinks[i]:
                break
        # Form the new string
        if parent_count == 0:
            # ='s are added and removed later to keep the length of the string exp
            # the same while other szlinks are processed.
            # This will keep previously calculated sz indices valid
            exp = (
                exp[0 : starting_indices_zlinks[i] - 1]
                + "=" * 7
                + exp[starting_indices_zlinks[i] + 6 :]
            )
        else:
            len_szlinks = (
                (starting_indices_zlinks[i] + 4 + 1 + parent_count + 1 - 1)
                - (index_parent_slink - 1)
                + 1
            )
            len_time_node = len('(TimeNode "' + str(parent_count) + '")')
            # "`" is added and removed later to keep the length of the string exp
            # the same while other szlinks are processed.
            # This will keep previously calculated sz indices valid
            exp = (
                exp[0 : index_parent_slink - 1]
                + '(TimeNode "'
                + str(parent_count)
                + '")'
                + "`" * (len_szlinks - len_time_node)
                + exp[starting_indices_zlinks[i] + 4 + 1 + parent_count + 1 :]
            )

    exp = exp.replace("=" * 7, '(TimeNode "0")')
    exp = exp.replace("`", "")
    return exp


def post_to_restapi_scheme_endpoint(
    cogscm_info, PORT=5000, IP_ADDRESS="127.0.0.1", cogscm_as=False
):
    """
    Visualize cognitive schema and selected action in each cycle.

    cogscm_as: an atomspace containing cognitive schema.
    cogscm_info: a dictionary containing
            - The discovered cognitive schema cogscm
            - human readable form msg
            - cycle count

    """
    # Post data to the visualizer
    uri = "http://" + IP_ADDRESS + ":" + str(PORT) + "/api/v1.1/"
    headers = {"content-type": "application/json"}

    if cogscm_as:
        execute_command("(clear)", uri, headers)
        all_cogscms = cogscm_as.get_atoms_by_type(
            types.BackPredictiveImplicationScopeLink
        )
        print("Discovered {} cogscms".format(len(all_cogscms)))

        for bpi in all_cogscms:
            bpi = """
            {} (ConceptNode "cogscm:{}")""".format(
                bpi, to_human_readable_str(bpi)
            )
            bpi = pre_process_atoms(str(bpi))
            execute_command(bpi, uri, headers)
            time.sleep(5)
    else:
        cogscm = cogscm_info["cogscm"]
        cycle = cogscm_info["cycle"]
        msg = cogscm_info["msg"]

        if cycle > 49:
            # clear the visualizer atomspace
            execute_command("(clear)", uri, headers)
        else:
            # update the current selected action
            get_response = get(uri + "atoms", params={"type": "ConceptNode"})
            for atom in get_response.json()["result"]["atoms"]:
                if "cogscm:" in atom["name"]:
                    delete_response = delete(uri + "atoms/" + str(atom["handle"]))

        cogscm = """
            {} (ConceptNode "{}")
        """.format(
            cogscm, msg
        )
        cogscm = pre_process_atoms(str(cogscm))
        execute_command(cogscm, uri, headers)
        time.sleep(5)


def execute_command(scheme_command, uri, headers):
    """
    Execute scheme command
    command = {'command': '(ConceptNode "cat")'}
    """
    command = {"command": scheme_command}
    post_response = post(uri + "scheme", data=json.dumps(command), headers=headers)


class MinerLogger:

    """Quick and dirty miner logger Python bindings"""

    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        scheme_eval(self.atomspace, "(use-modules (opencog miner))")

    def set_level(self, level: str) -> None:
        cmd_str = '(miner-logger-set-level! "' + level + '")'
        scheme_eval(self.atomspace, cmd_str)

    def set_sync(self, sync: bool) -> None:
        cmd_str = "(miner-logger-set-sync! " + to_scheme_str(sync) + ")"
        scheme_eval(self.atomspace, cmd_str)
