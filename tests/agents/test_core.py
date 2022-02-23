# Python
from pytest import approx

# OpenCog
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


def test_prior_estimate():
    """Test OpencogAgent.prior_estimate().

    Make sure that more complex cognitive schematics have lower prior
    estimates than simpler ones.

    """

    # Manually call setup for now (surely pytest can do better)
    setup()

    # Construct OpencogAgent with appropriate parameters
    oc_agent = OpencogAgent()

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
    )


    # Manually call teardown for now (surely pytest can do better)
    teardown()
