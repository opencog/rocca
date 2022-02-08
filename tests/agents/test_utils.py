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
    VariableNode,
    TypeNode,
    TypedVariableLink,
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
from rocca.agents.utils import (
    shannon_entropy,
    differential_entropy,
    get_uniq_atoms,
    to_human_readable_str,
)

# Set main atomspace
atomspace = AtomSpace()
set_default_atomspace(atomspace)

# TODO: for now setup and teardown are manually called for each test.
# Surely pytest can do better.


def setup():
    """Setup, to be called before each test."""

    None


def teardown():
    """Teardown, to be called after each test."""

    atomspace.clear()


def test_shannon_entropy():
    """Test Shannon entropy."""

    setup()

    # A is almost sure, thus has minimum Shannon entropy
    A = ConceptNode("A", tv=createTruthValue(1, 1))
    A_se = shannon_entropy(A)
    assert A_se == approx(0)

    # B is nearly unsure, thus has low Shannon entropy
    B = ConceptNode("B", tv=createTruthValue(0, 0.1))
    B_se = shannon_entropy(B)
    assert B_se == approx(0, abs=0.1)

    # C is true half of the time, thus has maximum Shannon entropy
    C = ConceptNode("C", tv=createTruthValue(0.5, 1))
    C_se = shannon_entropy(C)
    assert C_se == approx(1)

    # D is unknown, thus has maximum Shannon entropy
    D = ConceptNode("D", tv=createTruthValue(1, 0))
    D_se = shannon_entropy(D)
    assert D_se == approx(1)

    # E is highly probable, thus has mid range Shannon entropy
    E = ConceptNode("E", tv=createTruthValue(0.9, 0.5))
    E_se = shannon_entropy(E)
    assert E_se == approx(0.5, abs=0.1)

    # F is higly improbable and has a moderate confidence, thus has
    # Shannon entropy below 0.1
    F = ConceptNode("F", tv=createTruthValue(0, 1e-1))
    F_se = shannon_entropy(F)
    assert F_se < 0.1

    # G is higly improbable and has low confidence, thus has Shannon
    # entropy above 0.1
    G = ConceptNode("G", tv=createTruthValue(0, 1e-2))
    G_se = shannon_entropy(G)
    assert 0.1 < G_se

    # H is higly improbable and has very low confidence, thus has
    # Shannon entropy above 0.9
    H = ConceptNode("H", tv=createTruthValue(0, 1e-3))
    H_se = shannon_entropy(H)
    assert 0.9 < H_se

    teardown()


def test_differential_entropy():
    """Test differential entropy."""

    setup()

    # A is almost sure, thus has minimum differential entropy
    A = ConceptNode("A", tv=createTruthValue(1, 1))
    A_de = differential_entropy(A)
    assert A_de == -float("inf")

    # B is nearly unsure, thus has low differential entropy
    B = ConceptNode("B", tv=createTruthValue(0, 0.1))
    B_de = differential_entropy(B)
    assert B_de == approx(-3.5, abs=0.1)

    # C is true half of the time, thus has maximum differential entropy
    C = ConceptNode("C", tv=createTruthValue(0.5, 1))
    C_de = differential_entropy(C)
    assert C_de == approx(0)

    # D is unknown, thus has maximum differential entropy
    D = ConceptNode("D", tv=createTruthValue(1, 0))
    D_de = differential_entropy(D)
    assert D_de == approx(0)

    # E is slightly more probable than average, thus has high
    # differential entropy
    E = ConceptNode("E", tv=createTruthValue(0.55, 0.03))
    E_de = differential_entropy(E)
    assert E_de == approx(-1.0, abs=0.1)

    # F is slightly more probable than average, but with high
    # confidence, thus has lower differential entropy than E.
    F = ConceptNode("F", tv=createTruthValue(0.55, 0.99))
    F_de = differential_entropy(F)
    assert F_de == approx(-5.0, abs=0.1)

    # Atoms with (stv 0.9 0.9) or (stv 0.1 0.9) below the (currently)
    # default threshold
    G = ConceptNode("G", tv=createTruthValue(0.99, 1e-3))
    G_de = differential_entropy(G)
    H = ConceptNode("H", tv=createTruthValue(0.01, 1e-3))
    H_de = differential_entropy(H)
    assert G_de == approx(H_de)
    assert G_de < -1e-1

    # Atoms with (stv 0.5 0.9) or (stv 0.9 0.01) below the (currently)
    # default threshold
    I = ConceptNode("I", tv=createTruthValue(0.5, 1e-3))
    I_de = differential_entropy(I)
    J = ConceptNode("J", tv=createTruthValue(0.99, 1e-4))
    J_de = differential_entropy(J)
    assert -1e-1 < I_de
    assert -1e-1 < J_de

    teardown()


def test_get_uniq_atoms():
    P = PredicateNode("P")
    A = ConceptNode("A")
    B = ConceptNode("B")
    AB = ListLink(A, B)
    AA = ListLink(A, A)
    PAB = EvaluationLink(P, AB)
    PAA = EvaluationLink(P, AA)

    # Test all uniq atoms of PAB
    assert get_uniq_atoms(PAB) == {P, A, B, AB, PAB}

    # Test all uniq atoms of PAA
    assert get_uniq_atoms(PAA) == {P, A, AA, PAA}


def test_to_human_readable_str():
    # 1. outside(self, house) ∧ do(go_to_key) ↝ hold(self, key)
    cogscm_1 = BackPredictiveImplicationScopeLink(
        VariableSet(),
        SLink(ZLink()),
        AndLink(
            EvaluationLink(
                PredicateNode("outside"),
                ListLink(ConceptNode("self"), ConceptNode("house")),
            ),
            ExecutionLink(SchemaNode("go_to_key")),
        ),
        EvaluationLink(
            PredicateNode("hold"), ListLink(ConceptNode("self"), ConceptNode("key"))
        ),
    )

    cogscm_hrs_1 = to_human_readable_str(cogscm_1)
    expected_1 = "outside(self, house) ∧ do(go_to_key) ↝ hold(self, key)"

    assert cogscm_hrs_1 == expected_1

    # 2. AgentPosition(RightSquare) ∧ do(Eat) ↝ Reward(1)
    # Check that whitespace is removed in names
    cogscm_2 = BackPredictiveImplicationScopeLink(
        VariableSet(),
        SLink(ZLink()),
        AndLink(
            ExecutionLink(SchemaNode("Eat")),
            EvaluationLink(
                PredicateNode("Agent Position"), ConceptNode("Right Square")
            ),
        ),
        EvaluationLink(PredicateNode("Reward"), NumberNode("1")),
    )

    cogscm_hrs_2 = to_human_readable_str(cogscm_2)
    expected_2 = "AgentPosition(RightSquare) ∧ do(Eat) ↝ Reward(1)"

    assert cogscm_hrs_2 == expected_2

    # 3. -0.01 > $angle ∧ PoleAngle($angle) ∧ do(GoLeft) ↝ Reward(1)
    # Test with variable and GreaterThanLink
    cogscm_3 = BackPredictiveImplicationScopeLink(
        TypedVariableLink(VariableNode("$angle"), TypeNode("NumberNode")),
        SLink(ZLink()),
        AndLink(
            GreaterThanLink(NumberNode("-0.01"), VariableNode("$angle")),
            ExecutionLink(SchemaNode("Go Left")),
            EvaluationLink(PredicateNode("Pole Angle"), VariableNode("$angle")),
        ),
        EvaluationLink(PredicateNode("Reward"), NumberNode("1")),
    )

    cogscm_hrs_3 = to_human_readable_str(cogscm_3)

    # Two expected results due to AndLink commutativity (in spite of
    # actions being moved to the right)
    expected_3a = "-0.01 > $angle ∧ PoleAngle($angle) ∧ do(GoLeft) ↝ Reward(1)"
    expected_3b = "PoleAngle($angle) ∧ -0.01 > $angle ∧ do(GoLeft) ↝ Reward(1)"

    assert cogscm_hrs_3 == expected_3a or cogscm_hrs_3 == expected_3b
