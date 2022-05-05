# Python
from pytest import approx

# OpenCog
from opencog.type_constructors import (
    EvaluationLink,
    ListLink,
    PredicateNode,
    ConceptNode,
    TruthValue,
    VariableSet,
    AndLink,
    ExecutionLink,
    SchemaNode,
    NumberNode,
    GreaterThanLink,
)
from opencog.pln import (
    SLink,
    ZLink,
    BackPredictiveImplicationScopeLink,
    BackSequentialAndLink,
)
from opencog.atomspace import AtomSpace, createTruthValue
from opencog.utilities import set_default_atomspace

# ROCCA
from rocca.agents import MixtureModel
from rocca.agents.utils import agent_log

# Set main atomspace
atomspace = AtomSpace()
set_default_atomspace(atomspace)

# Set loggers
agent_log.set_level("fine")
agent_log.use_stdout()


def setup():
    """Setup, to be called before each test."""

    None


def teardown():
    """Teardown, to be called after each test."""

    atomspace.clear()


def test_weight():
    """Test MixtureModel.weight()."""

    # Manually call setup for now (surely pytest can do better)
    setup()

    # Construct OpencogAgent with appropriate parameters
    action_names = ["go_to_key", "go_to_house", "go_to_diamonds"]
    action_space = {ExecutionLink(SchemaNode(a)) for a in action_names}
    mm = MixtureModel(action_space, prior_a=1.0, prior_b=1.0)
    mm.complexity_penalty = 0.5
    mm.compressiveness = 0.1

    # 1. hold(self, key) ∧ do(go_to_house) ⩘ do(go_to_diamonds) ↝ Reward(1)
    cogscm_1 = BackPredictiveImplicationScopeLink(
        VariableSet(),
        SLink(ZLink()),
        BackSequentialAndLink(
            SLink(ZLink()),
            AndLink(
                EvaluationLink(
                    PredicateNode("hold"),
                    ListLink(ConceptNode("self"), ConceptNode("key")),
                ),
                ExecutionLink(SchemaNode("go_to_house")),
            ),
            ExecutionLink(SchemaNode("go_to_diamonds")),
        ),
        EvaluationLink(PredicateNode("Reward"), NumberNode("1")),
        tv=createTruthValue(1.0, 0.00780669),
    )

    # 2. hold(self, key) ∧ do(go_to_house) ⩘ do(go_to_house) ⩘ do(go_to_diamonds) ↝ Reward(1)
    cogscm_2 = BackPredictiveImplicationScopeLink(
        VariableSet(),
        SLink(ZLink()),
        BackSequentialAndLink(
            SLink(ZLink()),
            BackSequentialAndLink(
                SLink(ZLink()),
                AndLink(
                    EvaluationLink(
                        PredicateNode("hold"),
                        ListLink(ConceptNode("self"), ConceptNode("key")),
                    ),
                    ExecutionLink(SchemaNode("go_to_house")),
                ),
                ExecutionLink(SchemaNode("go_to_house")),
            ),
            ExecutionLink(SchemaNode("go_to_diamonds")),
        ),
        EvaluationLink(PredicateNode("Reward"), NumberNode("1")),
        tv=createTruthValue(1.0, 0.00402985),
    )

    # Needs to be called before testing MixtureModel.prior_estimate
    mm.infer_data_set_size([cogscm_1, cogscm_2], 50)

    # Check that cogscm_1 complexity (a proxy for its syntactic
    # length) is lower than cogscm_2 complexity
    assert mm.complexity(cogscm_1) < mm.complexity(cogscm_2)

    # Check that the Kolmogorov complexity estimate of the unexplained
    # data of cogscm_1 is lower than that of cogscm_2, because
    # cogscm_1 has a higher confidence (thus cover more data points).
    uds_1 = mm.unexplained_data_size(cogscm_1)
    uds_2 = mm.unexplained_data_size(cogscm_2)
    assert mm.kolmogorov_estimate(uds_1) < mm.kolmogorov_estimate(uds_2)

    # Check that cogscm_1 has a higher prior estimate than cogscm_2
    # since cogscm_1 is both simpler and has higher confidence than
    # cogscm_2
    assert mm.prior_estimate(cogscm_2) < mm.prior_estimate(cogscm_1)

    # Check that cogscm_1 has a lower Beta factor than cogscm_2 because
    # cogscm_1 has a higher confidence than cogscm_2 and same strength
    assert mm.beta_factor(cogscm_1) < mm.beta_factor(cogscm_2)

    # Check that cogscm_1 has a higher weight than cogscm_2 since
    # cogscm_1 is both simpler and has higher predictive power (higher
    # confidence for same strength).  Note that this test can only
    # pass if the compressiveness parameter is set sufficiently lower
    # and the complexity_penalty is set sufficiently high, as to take
    # the higher confidence sufficiently into account and counter act
    # the beta factor.
    assert mm.weight(cogscm_1) > mm.weight(cogscm_2)

    # Manually call teardown for now (surely pytest can do better)
    teardown()
