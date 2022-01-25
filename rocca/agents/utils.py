# Util functions used by GymAgent

##############
# Initialize #
##############

# Python
import random
from typing import Any

from orderedmultidict import omdict

# SciPy
import scipy.special as sp
import scipy.stats as st

# OpenCog
from opencog.atomspace import (
    Atom,
    AtomSpace,
    get_type,
    is_a,
    types,
    createTruthValue,
)
from opencog.execute import execute_atom
from opencog.logger import create_logger
from opencog.pln import ZLink, SLink
from opencog.scheme import scheme_eval
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
from opencog.utilities import get_free_variables

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

# TODO: replace list[Atom] by set[Atom] | list[Atom] once we have
# completely moved to Python 3.10.  Indeed for instance
# OpencogAgent.update_cognitive_schematics uses set[Atom], not
# list[Atom].
def add_to_atomspace(atoms: list[Atom], atomspace: AtomSpace) -> None:
    """Add all atoms to the atomspace."""

    for atom in atoms:
        atomspace.add_atom(atom)


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


def w8d_cogscm_to_str(w8d_cogscm: tuple[float, Atom], indent: str = "") -> str:
    """Pretty print a pair (weight, cogscm)."""

    weight = w8d_cogscm[0]
    cogscm = w8d_cogscm[1]
    tv = get_cogscm_tv(cogscm)
    idstr = atom_to_idstr(cogscm)
    s = "(weight={}, tv={}, id={})".format(weight, tv, idstr)
    return s


def w8d_cogscms_to_str(w8d_cogscms: list[tuple[float, Atom]], indent: str = "") -> str:
    """Pretty print the given list of weighted cogscms"""

    w8d_cogscms_sorted = sorted(w8d_cogscms, key=lambda x: x[0], reverse=True)

    s = indent + "size = " + str(len(w8d_cogscms_sorted)) + "\n"
    for w8d_cogscm in w8d_cogscms_sorted:
        s += indent + w8d_cogscm_to_str(w8d_cogscm, indent + "  ") + "\n"
    return s


def action_to_str(action: Atom, indent: str = "") -> str:
    """Pretty print an action.

    For now it just outputs the schema corresponding to the action
    without the execution link, for conciseness.

    """

    return indent + str(action.out[0])


def act_pblt_to_str(act_pblt: tuple[Atom, float], indent: str = "") -> str:
    action = act_pblt[0]
    pblt = act_pblt[1]
    return indent + "({}, {})".format(action_to_str(action), pblt)


def act_pblts_to_str(act_pblts: tuple[Atom, float], indent: str = "") -> str:
    """Pretty print a list of pairs (action, probability)."""

    return "\n".join([indent + act_pblt_to_str(act_pblt) for act_pblt in act_pblts])


def act_w8d_cogscm_to_str(
    act_w8d_cogscm: tuple[Atom, tuple[float, Atom]], indent: str = ""
) -> str:
    """Pretty print a pair (action, (weight, cogscm))."""

    action = act_w8d_cogscm[0]
    w8d_cogscm = act_w8d_cogscm[1]
    s = indent + action_to_str(action) + ": " + w8d_cogscm_to_str(w8d_cogscm)
    return s


def act_w8d_cogscms_to_str(
    act_w8d_cogscms: list[tuple[Atom, tuple[float, Atom]]], indent: str = ""
) -> str:
    """Pretty print a list of pairs (action, (weight, cogscm))."""

    return "\n".join(
        [
            indent + act_w8d_cogscm_to_str(act_w8d_cogscm)
            for act_w8d_cogscm in act_w8d_cogscms
        ]
    )


def mxmdl_to_str(mxmdl: omdict, indent: str = "") -> str:
    """Pretty print the given mixture model of cogscms"""

    s = ""
    for act_w8d_cogscms in mxmdl.listitems():
        action = act_w8d_cogscms[0]
        w8d_cogscms = act_w8d_cogscms[1]
        s += "\n" + indent + str(action_to_str(action)) + "\n"
        s += w8d_cogscms_to_str(w8d_cogscms, indent + "  ")
    return s


def thompson_sample(
    mxmdl: omdict, prior_a: float = 1, prior_b: float = 1
) -> tuple[Atom, float]:
    """Perform Thompson sampling over the mixture model.

    Meaning, for each action

    1. Select a TV according to its likelihood (derived from its
    cognitive schematic).

    2. From that TV, sample its second order distribution to obtain a
    first order probability variate, and return the pair (action,
    pblty) corresponding to the highest variate.

    Then return the action with the highest probability of success.

    """

    agent_log.fine(
        "thompson_sample(mxmdl={}, prior_a={}, prior_b={})".format(
            mxmdl_to_str(mxmdl), prior_a, prior_b
        )
    )

    # 1. For each action select its TV according its weight
    act_w8d_cogscms = [
        (action, weighted_sampling(w8d_cogscms))
        for (action, w8d_cogscms) in mxmdl.listitems()
    ]
    agent_log.fine(
        "act_w8d_cogscms:\n{}".format(act_w8d_cogscms_to_str(act_w8d_cogscms))
    )

    # 2. For each action select its first order probability given its tv
    act_pblts = [
        (action, tv_rv(get_cogscm_tv(w8_cogscm[1]), prior_a, prior_b))
        for (action, w8_cogscm) in act_w8d_cogscms
    ]
    agent_log.fine("act_pblts:\n{}".format(act_pblts_to_str(act_pblts)))

    # Return an action with highest probability of success (TODO: take
    # case of ties)
    return max(act_pblts, key=lambda act_pblt: act_pblt[1])


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


def is_virtual(clause: Atom) -> bool:
    """Return true iff the clause is virtual.

    For instance

    (GreaterThanLink (NumberNode "1") (NumberNode "2"))

    is virtual because it gets evaluated on the fly as opposed to be
    query against the atomspace.

    """

    # TODO: can be simplified with clause.is_a
    return is_a(clause.type, types.VirtualLink)


def get_context(cogscm: Atom) -> tuple[Atom, Atom]:
    """Extract the context of a cognitive schematic.

    For instance given a cognitive schematic of that format

    BackPredictiveImplicationScope <tv>
      <vardecl>
      <expiry>
      And
        <context>
        <execution>
      <goal>

    return <context>.

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

    return <context>

    Since the context can be multiple clauses, virtual and
    non-virtual, it outputs a pair of two lists

    (present-clauses, virtual-clauses)

    """

    # Grab all clauses pertaining to context
    clauses = get_t0_clauses(get_antecedent(cogscm))
    no_exec_clauses = [x for x in clauses if not is_execution(x)]

    # Split them into present and virtual clauses
    present_clauses = [x for x in no_exec_clauses if not is_virtual(x)]
    virtual_clauses = [x for x in no_exec_clauses if is_virtual(x)]

    # Return pair of present and virtual clauses
    return (present_clauses, virtual_clauses)


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


def is_execution(atom: Atom) -> bool:
    """Return True iff the atom is an ExecutionLink."""

    return is_a(atom.type, types.ExecutionLink)


def is_Z(atom: Atom) -> bool:
    """Return True iff the atom is Z."""

    return atom.type == get_type("ZLink")


def is_S(atom: Atom) -> bool:
    """Return True iff the atom is S ..."""

    return atom.type == get_type("SLink")


def maybe_and(clauses: list[Atom]) -> Atom:
    """Wrap an And if multiple clauses, otherwise return the only one."""

    return AndLink(*clauses) if 1 < len(clauses) else clauses[0]


def get_antecedent(atom: Atom):  # TODO: requires Python 3.10 -> (Atom | None):
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


def get_events(timed_atoms: list[Atom]) -> list[Atom]:
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
    agent_log.fine("get_total_lag(atom={}) = {}".format(atom, tlg))
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
      <expiry>
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
    tv = execute_atom(atomspace, query)
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

    #    if is_Z(n):
    #        return 0
    #    if is_S(n):
    #        return 1 + to_int(n.out[0])

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


def atomspace_to_str(atomspace: AtomSpace) -> str:
    """Takes an atomspace and return its content as a string"""
    return str(scheme_eval(atomspace, "(cog-get-all-roots)").decode("utf-8"))


def agent_log_atomspace(
    atomspace: AtomSpace, level: str = "fine", msg_prefix: str = "atomspace"
) -> None:
    """Takes an atomspace and log its content (with size and address)"""

    msg = "{} [address={}, size={}]:\n{}".format(
        msg_prefix, atomspace, len(atomspace), atomspace_to_str(atomspace)
    )
    agent_log.log(level, msg)


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
