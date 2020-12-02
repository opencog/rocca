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
from opencog.logger import Logger, log

#############
# Constants #
#############

TRUE_TV = TruthValue(1, 1)
DEFAULT_TV = TruthValue(1, 0)

#############
# Functions #
#############

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


def w8d_cogscm_to_str(w8d_cogscm, indent=""):
    """Pretty print the given list of weighted cogscm"""

    weight = w8d_cogscm[0]
    cogscm = w8d_cogscm[1]
    tv = get_cogscm_tv(cogscm)
    idstr = cogscm.id_string() if cogscm else "None"
    s = "(weight={}, tv={}, id={})".format(weight, tv, idstr)
    return s


def w8d_cogscms_to_str(w8d_cogscms, indent=""):
    """Pretty print the given list of weighted cogscms"""

    w8d_cogscms_sorted = sorted(w8d_cogscms, key=lambda x: x[0], reverse=True)

    s = ""
    for w8d_cogscm in w8d_cogscms_sorted:
        s += indent + w8d_cogscm_to_str(w8d_cogscm, indent + "  ") + "\n"
    return s


def mxmdl_to_str(mxmdl, indent=""):
    """Pretty print the given mixture model of cogscms"""

    s = ""
    for act_w8d_cogscms in mxmdl.listitems():
        s += "\n" + indent + str(act_w8d_cogscms[0]) + "\n"
        s += w8d_cogscms_to_str(act_w8d_cogscms[1], indent + "  ")
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

    # 1. For each action select its TV according its weight
    actcogscms = [(action, weighted_sampling(w8d_cogscms))
                  for (action, w8d_cogscms) in mxmdl.listitems()]

    # 2. For each action select its first order probability given its tv
    actps = [(action, tv_rv(get_cogscm_tv(cogscm), prior_a, prior_b))
             for (action, cogscm) in actcogscms]

    # Return an action with highest probability of success (TODO: take
    # case of ties)
    return max(actps, key=lambda actp: actp[1])


def get_cogscm_tv(cogscm):
    """Return the Truth Value of a cogscm or the default if it is None"""

    return cogscm.tv if cogscm else DEFAULT_TV


def weighted_sampling(weighted_list):
    """Given list of pairs (weight, element) weight-randomly select an element.

    """

    w8s = [weight for (weight, _) in weighted_list]
    elements = [element for (_, element) in weighted_list]
    return random.choices(elements, weights=w8s)[0]


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

    """

    return cogscm.out[0]


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


def is_scope(x):
    """Return True iff the atom is a scope link."""

    return is_a(x.type, types.ScopeLink)


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

    ante = cogscm.out[2]
    return ante.out if is_a(ante.type, types.AndLink) else [ante]


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
                   *[IsClosedLink(spc) for spc in stamped_present_clauses],
                   *[high_strength_virtual_clause(spc)
                     for spc in stamped_present_clauses],
                   *[high_confidence_virtual_clause(spc)
                     for spc in stamped_present_clauses],
                   *virtual_clauses)
    query = SatisfactionLink(vardecl, body)
    tv = execute_atom(atomspace, query)
    return tv


def high_strength_virtual_clause(a):
    """Make a virtual clause checking that a has a high strength."""
    almost_one = NumberNode("0.99")
    return GreaterThanLink(StrengthOfLink(a), almost_one)

def high_confidence_virtual_clause(a):
    """Make a virtual clause checking that a has a high confidence."""
    almost_one = NumberNode("0.99")
    return GreaterThanLink(ConfidenceOfLink(a), almost_one)

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
