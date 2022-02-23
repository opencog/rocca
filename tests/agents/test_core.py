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

# Set main atomspace
atomspace = AtomSpace()
set_default_atomspace(atomspace)

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

    # 1. hold(self, key) ∧ do(go_to_house) ≺ do(go_to_diamonds) ↝ Reward(1)
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

    # 2. hold(self, key) ∧ do(go_to_house) ≺ do(go_to_house) ≺ do(go_to_diamonds) ↝ Reward(1)
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

    # Check that cogscm_1 has a higher prior estimate than cogscm_2
    # since cogscm_1 is simpler than cogscm_2
    assert mm.prior_estimate(cogscm_2) < mm.prior_estimate(cogscm_1)

    # Check that cogscm_1 has a higher Beta factor than cogscm_2 since
    # cogscm_1 has a higher confidence than cogscm_2
    assert mm.beta_factor(cogscm_2) < mm.beta_factor(cogscm_1)

    # Check that cogscm_1 has a higher weight than cogscm_2 since
    # cogscm_1 has both higher prior estimate and beta factor than
    # cogscm_2
    assert mm.weight(cogscm_2) < mm.weight(cogscm_1)

    # Manually call teardown for now (surely pytest can do better)
    teardown()
