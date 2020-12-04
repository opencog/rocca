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
from opencog.logger import Logger, log, create_logger

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

    # agent_log.fine("w8d_cogscm_to_str(w8d_cogscm={}, indent={})".format(w8d_cogscm, indent))

    weight = w8d_cogscm[0]
    cogscm = w8d_cogscm[1]
    tv = get_cogscm_tv(cogscm)
    idstr = atom_to_idstr(cogscm)
    s = "(weight={}, tv={}, id={})".format(weight, tv, idstr)
    return s


def w8d_cogscms_to_str(w8d_cogscms, indent=""):
    """Pretty print the given list of weighted cogscms"""

    w8d_cogscms_sorted = sorted(w8d_cogscms, key=lambda x: x[0], reverse=True)

    s = ""
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


# TODO: use join
def act_pblts_to_str(act_pblts, indent=""):
    """Pretty print a list of pairs (action, probability)."""

    s = ""
    for act_pblt in act_pblts:
        s += indent + act_pblt_to_str(act_pblt) + "\n"
    return s


def act_w8d_cogscm_to_str(act_w8d_cogscm, indent=""):
    """Pretty print a pair (action, (weight, cogscm))."""

    # agent_log.fine("act_w8d_cogscm_to_str(act_w8d_cogscm={}, indent={})".format(act_w8d_cogscm, indent))

    action = act_w8d_cogscm[0]
    w8d_cogscm = act_w8d_cogscm[1]
    s = indent + action_to_str(action) + ": " + w8d_cogscm_to_str(w8d_cogscm)
    return s


# TODO: use join
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

    # agent_log.fine("thompson_sample(mxmdl={}, prior_a={}, prior_b={})".format(mxmdl_to_str(mxmdl), prior_a, prior_b))

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

    return <context>.

    Since the context can be multiple clauses, virtual and
    non-virtual, it outputs a pair of two lists

    (present-clauses, virtual-clauses)

    """

    # Grab all clauses pertaining to context
    clauses = get_cogscm_antecedants(cogscm)
    no_exec_clauses = [x for x in clauses if x.type != types.ExecutionLink]

    # Split them into present and virtual clauses
    present_clauses = [x for x in no_exec_clauses if not is_virtual(x)]
    virtual_clauses = [x for x in no_exec_clauses if is_virtual(x)]

    # Return pair of present and virtual clauses
    return (present_clauses, virtual_clauses)


def is_scope(atom):
    """Return True iff the atom is a scope link."""

    return is_a(atom.type, types.ScopeLink)


def is_predictive_implication(atom):
    """Return True iff the atom is a predictive implication link."""

    return is_a(atom.type, types.PredictiveImplicationLink)


def is_predictive_implication_scope(atom):
    """Return True iff the atom is a predictive implication scope link."""

    return is_a(atom.type, types.PredictiveImplicationScopeLink)


def is_and(atom):
    """Return True iff the atom is an and link."""

    return is_a(atom.type, types.AndLink)


def get_cogscm_antecedants(cogscm):
    """Return the list of antecedants of a cognitive schema.

    For instance is the cognitive schematics is represented by

    PredictiveImplicationScope <tv>
      <vardecl>
      <expiry>
      And (or SimultaneousAnd?)
        <context-or-action-1>
        ...
        <context-or-action-n>
      <goal>

    it returns [<context-or-action-1>, ..., <context-or-action-1>]

    """

    ante = cogscm.out[2] if is_scope(cogscm) else cogscm.out[1]
    return ante.out if is_and(ante) else [ante]


def get_action(cogscm):
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

    cnjs = get_cogscm_antecedants(cogscm)
    execution = next(x for x in cnjs if x.type == types.ExecutionLink)
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


def timestamp(atom, i, tv=None, nat=True):
    """Timestamp a given atom.  Optionally set its TV

    AtTimeLink <tv>               # if tv is provided
      <atom>
      TimeNode <str(i)>

    if nat is True it uses a Natural instead of TimeNode (see to_nat).

    """

    time = to_nat(i) if nat else TimeNode(str(i))
    return AtTimeLink(atom, time, tv=tv)


def to_nat(i):
    """Convert i to a Natural.

    For instance if i = 3, then it returns

    S S S Z

    """

    return ZLink() if i == 0 else SLink(to_nat(i - 1))
