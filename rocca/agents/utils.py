# Util functions used by GymAgent

##############
# Initialize #
##############

# Python
import random
from orderedmultidict import omdict

# SciPy
import scipy.stats as st
import scipy.special as sp

# OpenCog
from opencog.atomspace import AtomSpace, TruthValue
from opencog.atomspace import types
from opencog.atomspace import get_type, is_a
from opencog.exec import execute_atom
from opencog.type_constructors import *
from opencog.spacetime import *
from opencog.pln import *
from opencog.utilities import is_closed, get_free_variables
from opencog.logger import create_logger

#############
# Constants #
#############

TRUE_TV = TruthValue(1, 1)
DEFAULT_TV = TruthValue(1, 0)

#############
# Variables #
#############

agent_log = create_logger("opencog.log")
agent_log.set_component("Agent")

#############
# Functions #
#############

def has_non_null_confidence(atom):
    """Return True iff the given atom has a confidence above 0."""

    return 0 < atom.tv.confidence


def tv_to_beta(tv, prior_a=1, prior_b=1):
    """Convert a truth value to a beta distribution.

    Given a truth value, return the beta distribution that best fits
    it.  Two optional parameters are provided to set the prior of the
    beta-distribution, the default values are prior_a=1 and prior_b=1
    corresponding to the Bayesian prior.

    """

    count = tv.count
    pos_count = count * tv.mean # the mean is actually the mode
    a = prior_a + pos_count
    b = prior_b + count - pos_count
    return st.beta(a, b)


def tv_to_alpha_param(tv, prior_a=1, prior_b=1):
    """Return the alpha parameter of a TV's beta-distribution.

    """

    count = tv.count
    pos_count = count * tv.mean # the mean is actually the mode
    return prior_a + pos_count


def tv_to_beta_param(tv, prior_a=1, prior_b=1):
    """Return the beta parameter of a TV's beta-distribution.

    """

    count = tv.count
    pos_count = count * tv.mean # the mean is actually the mode
    return prior_b + count - pos_count


def tv_rv(tv, prior_a=1, prior_b=1):

    """Return a first order probability variate of a truth value.

    Return a first order probability variate of the beta-distribution
    representing the second order distribution of tv.

    rv stands for Random Variate.

    """

    betadist = tv_to_beta(tv, prior_a, prior_b)
    return betadist.rvs()


def atom_to_idstr(atom):
    return atom.id_string() if atom else "None"


def w8d_cogscm_to_str(w8d_cogscm, indent=""):
    """Pretty print a pair (weight, cogscm)."""

    weight = w8d_cogscm[0]
    cogscm = w8d_cogscm[1]
    tv = get_cogscm_tv(cogscm)
    idstr = atom_to_idstr(cogscm)
    s = "(weight={}, tv={}, id={})".format(weight, tv, idstr)
    return s


def w8d_cogscms_to_str(w8d_cogscms, indent=""):
    """Pretty print the given list of weighted cogscms"""

    w8d_cogscms_sorted = sorted(w8d_cogscms, key=lambda x: x[0], reverse=True)

    s = indent + "size = " + str(len(w8d_cogscms_sorted)) + "\n"
    for w8d_cogscm in w8d_cogscms_sorted:
        s += indent + w8d_cogscm_to_str(w8d_cogscm, indent + "  ") + "\n"
    return s


def action_to_str(action, indent=""):
    """Pretty print an action.

    For now it just outputs the schema corresponding to the action
    without the execution link, for conciseness.

    """

    return indent + str(action.out[0])


def act_pblt_to_str(act_pblt, indent=""):
    action = act_pblt[0]
    pblt = act_pblt[1]
    return indent + "({}, {})".format(action_to_str(action), pblt)


# TODO: use join to optimize
def act_pblts_to_str(act_pblts, indent=""):
    """Pretty print a list of pairs (action, probability)."""

    s = ""
    for act_pblt in act_pblts:
        s += indent + act_pblt_to_str(act_pblt) + "\n"
    return s


def act_w8d_cogscm_to_str(act_w8d_cogscm, indent=""):
    """Pretty print a pair (action, (weight, cogscm))."""

    action = act_w8d_cogscm[0]
    w8d_cogscm = act_w8d_cogscm[1]
    s = indent + action_to_str(action) + ": " + w8d_cogscm_to_str(w8d_cogscm)
    return s


# TODO: use join to optimize
def act_w8d_cogscms_to_str(act_w8d_cogscms, indent=""):
    """Pretty print a list of pairs (action, (weight, cogscm))."""

    s = ""
    for act_w8d_cogscm in act_w8d_cogscms:
        s += indent + act_w8d_cogscm_to_str(act_w8d_cogscm) + "\n"
    return s


def mxmdl_to_str(mxmdl, indent=""):
    """Pretty print the given mixture model of cogscms"""

    s = ""
    for act_w8d_cogscms in mxmdl.listitems():
        action = act_w8d_cogscms[0]
        w8d_cogscms = act_w8d_cogscms[1]
        s += "\n" + indent + str(action_to_str(action)) + "\n"
        s += w8d_cogscms_to_str(w8d_cogscms, indent + "  ")
    return s


def thompson_sample(mxmdl, prior_a=1, prior_b=1):
    """Perform Thompson sampling over the mixture model.

    Meaning, for each action

    1. Select a TV according to its likelihood (derived from its
    cognitive schematic).

    2. From that TV, sample its second order distribution to obtain a
    first order probability variate, and return the pair (action,
    pblty) corresponding to the highest variate.

    Then return the action with the highest probability of success.

    """

    agent_log.fine("thompson_sample(mxmdl={}, prior_a={}, prior_b={})".format(mxmdl_to_str(mxmdl), prior_a, prior_b))

    # 1. For each action select its TV according its weight
    act_w8d_cogscms = [(action, weighted_sampling(w8d_cogscms))
                  for (action, w8d_cogscms) in mxmdl.listitems()]
    agent_log.fine("act_w8d_cogscms:\n{}".format(act_w8d_cogscms_to_str(act_w8d_cogscms)))

    # 2. For each action select its first order probability given its tv
    act_pblts = [(action, tv_rv(get_cogscm_tv(w8_cogscm[1]), prior_a, prior_b))
                 for (action, w8_cogscm) in act_w8d_cogscms]
    agent_log.fine("act_pblts:\n{}".format(act_pblts_to_str(act_pblts)))

    # Return an action with highest probability of success (TODO: take
    # case of ties)
    return max(act_pblts, key=lambda act_pblt: act_pblt[1])


def get_cogscm_tv(cogscm):
    """Return the Truth Value of a cogscm or the default if it is None"""

    return cogscm.tv if cogscm else DEFAULT_TV


def weighted_sampling(weighted_list):
    """Given list of pairs (weight, element) weight-randomly select an element.

    Return pair (w, e) of the weight associated to the selected element.

    """

    w8s = [weight for (weight, _) in weighted_list]
    return random.choices(weighted_list, weights=w8s)[0]


def weighted_average_tv(weighted_tvs):
    """Given a list of pairs (weight, tv) return the weighted average tv.

    Return a Simple Truth Value with mean and variance equivalent to
    that of a distributional truth value built with the weighted
    average of the input tvs.

    """

    # TODO: actually not needed
    return None


def get_vardecl(cogscm):
    """Extract the vardecl of a cognitive schematic.

    Given a cognitive schematic of that format

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

    return <vardecl>.

    If the cognitive schematic is an PredictiveImplicationLink then
    return an empty VariableSet.

    """

    return cogscm.out[0] if is_scope(cogscm) else VariableSet()


def is_virtual(clause):
    """Return true iff the clause is virtual.

    For instance

    (GreaterThanLink (NumberNode "1") (NumberNode "2"))

    is virtual because it gets evaluated on the fly as opposed to be
    query against the atomspace.

    """

    # TODO: can be simplified with clause.is_a
    return is_a(clause.type, types.VirtualLink)


def get_context(cogscm):
    """Extract the context of a cognitive schematic.

    For instance given a cognitive schematic of that format

    PredictiveImplicationScope <tv>
      <vardecl>
      <expiry>
      And
        <context>
        <execution>
      <goal>

    return <context>.

    Another example, given a cognitive schematic of that format

    PredictiveImplicationScope <tv>
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


def is_variable(atom):
    """Return True iff the atom is a variable node."""

    return is_a(atom.type, types.VariableNode)


def is_scope(atom):
    """Return True iff the atom is a scope link."""

    return is_a(atom.type, types.ScopeLink)


def is_predictive_implication(atom):
    """Return True iff the atom is a predictive implication link."""

    return is_a(atom.type, get_type("PredictiveImplicationLink"))


def is_predictive_implication_scope(atom):
    """Return True iff the atom is a predictive implication scope link."""

    return is_a(atom.type, get_type("PredictiveImplicationScopeLink"))


def is_and(atom):
    """Return True iff the atom is an and link."""

    return is_a(atom.type, types.AndLink)


def is_sequential_and(atom):
    """Return True iff atom is a sequential and.

    Also for now we use AltSequentialAndLink.

    """

    return is_a(atom.type, get_type("AltSequentialAndLink"))

def is_execution(atom):
    """Return True iff the atom is an ExecutionLink."""

    return is_a(atom.type, types.ExecutionLink)


def is_Z(atom):
    """Return True iff the atom is Z."""

    return atom.type == get_type("ZLink")


def is_S(atom):
    """Return True iff the atom is S ..."""

    return atom.type == get_type("SLink")


def maybe_and(clauses):
    """Wrap an And if multiple clauses, otherwise return the only one.

    """

    return AndLink(*clauses) if 1 < len(clauses) else clauses[0]


def get_antecedent(atom):
    """Return the antecedent of a temporal atom.

    For instance is the cognitive schematics is represented by

    PredictiveImplicationScope <tv>
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


def get_succedent(atom):
    """Return the succedent of a temporal atom.

    For instance is the cognitive schematics is represented by

    PredictiveImplicationScope <tv>
      <vardecl>
      <expiry>
      <antecedent>
      <succedent>

    it returns <succedent>

    """

    if is_predictive_implication_scope(atom):
        return atom.out[3]
    if is_predictive_implication(atom):
        return atom.out[2]
    if is_sequential_and(atom):
        return atom.out[2]
    return None


def get_lag(atom):
    """Given an temporal atom, return its lag component.

    For instance if it is a PredictiveImplicationScope

    PredictiveImplicationScope
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


def get_t0_clauses(antecedent):
    """Return the list of clauses occuring at initial time.

    For instance if the cognitive schematics has the following format

    PredictiveImplicationScope <tv>
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


def has_all_variables_in_antecedent(cogscm):
    """Return True iff all variables are in the antecedent."""

    if is_scope(cogscm):
        vardecl_vars = set(get_free_variables(get_vardecl(cogscm)))
        antecedent_vars = set(get_free_variables(get_antecedent(cogscm)))
        return vardecl_vars == antecedent_vars
    else:
        return True

# TODO: optimize using comprehension
def get_free_variables_of_atoms(atoms):
    """Get the set of all free variables in all atoms.

    """

    variables = set()
    for atom in atoms:
        variables.update(set(get_free_variables(atom)))
    return variables


def get_times(timed_atoms):
    """Given a list of timestamped clauses, return a set of all times.

    """

    if timed_atoms == []:
        return set()
    return set.union(set([get_time(timed_atoms[0])]), get_times(timed_atoms[1:]))


def get_events(timed_atoms):
    """Given a list of timestamped clauses, return a list of all events.

    """

    return [get_event(ta) for ta in timed_atoms]


def get_latest_time(timed_clauses):
    """Given a list of timestamped clauses, return the latest timestamp.

    """

    if timed_clauses == []:
        return ZLink()
    return nat_max(get_time(timed_clauses[0]), get_latest_time(timed_clauses[1:]))


def get_latest_clauses(timed_clauses):
    """Given a list of timestamped clauses, return the latest clauses.

    For instance if the timestamped clauses are

    [AtTime(A, T), AtTime(B, S(T)), AtTime(C, S(S(T))), AtTime(D, S(S(T)))]

    return

    [AtTime(C, S(S(T))), AtTime(D, S(S(T)))]

    """

    lt = get_latest_time(timed_clauses)
    return [tc for tc in timed_clauses if get_time(tc) == lt]


def get_early_clauses(timed_clauses):
    """Return all clauses that are not the latest.

    """

    lcs = set(get_latest_clauses(timed_clauses))
    return list(set(timed_clauses).difference(lcs))


def get_total_lag(atom):
    """Return the total lag between earliest and lastest subatoms of atom.

    For instance if the atom is

    PredictiveImplicationScope <tv>
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


def get_t0_execution(cogscm):
    """Extract the action of a cognitive schematic.

    Given a cognitive schematic of that formats

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

    extract
        Execution
          <action>
          <input> [optional]
          <output> [optional]

    """

    cnjs = get_t0_clauses(get_antecedent(cogscm))
    execution = next(x for x in cnjs if is_execution(x))
    return execution


def get_context_actual_truth(atomspace, cogscm, i):
    """Calculate tv of the context of cognitive schematic cogscm at time i.

    Given a cognitive schematic of that format

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

    calculate the truth value of <context> at time i, where the
    variables have been properly instantiated.

    For now the calculation is crisp, either a context is completely
    true (stv 1 1) or completely false (stv 0 1).

    """

    # Build and run a query to check if the context is true
    vardecl = get_vardecl(cogscm)
    present_clauses, virtual_clauses = get_context(cogscm)
    stamped_present_clauses = [timestamp(pc, i) for pc in present_clauses]
    body = AndLink(PresentLink(*stamped_present_clauses),
                   IsClosedLink(*stamped_present_clauses),
                   IsTrueLink(*stamped_present_clauses),
                   *virtual_clauses)
    query = SatisfactionLink(vardecl, body)
    tv = execute_atom(atomspace, query)
    return tv


def get_event(timed_atom):
    """Return the event in a clause that is a timestamped event

    For instance if clause is

    AtTime
      <event>
      <time>

    return <event>

    """

    return timed_atom.out[0]


def get_time(timed_atom):
    """Given (AtTime A T) return T.

    """

    return timed_atom.out[1]


def timestamp(atom, i, tv=None, nat=True):
    """Timestamp a given atom.  Optionally set its TV

    AtTimeLink <tv>               # if tv is provided
      <atom>
      TimeNode <str(i)>

    if nat is True it uses a Natural instead of TimeNode (see to_nat).

    """

    time = to_nat(i) if nat else TimeNode(str(i))
    return AtTimeLink(atom, time, tv=tv)


def nat_max(n, m):

    """Return the max between two naturals (including if they wrap variables)."""

    if is_Z(n) or is_variable(n):
        return m
    if is_Z(m) or is_variable(m):
        return n
    return SLink(nat_max(n.out[0], m.out[0]))


def to_nat(i):
    """Convert i to a Natural.

    For instance if i = 3, then it returns

    S S S Z

    """

    return ZLink() if i == 0 else SLink(to_nat(i - 1))


def lag_to_nat(i, T):
    """Given an int i and T, return as many SLinks wrapping around T.

    For instance if i=3 and T=VariableNode("$T") return

    S(S(S(T)))

    """

    return T if i == 0 else SLink(lag_to_nat(i - 1, T))


def to_int(n):

    """Convert n to an Int.

    For instance if n = S S S Z, then it returns 3.

    """

    if is_Z(n):
        return 0
    if is_S(n):
        return 1 + to_int(n.out[0])
