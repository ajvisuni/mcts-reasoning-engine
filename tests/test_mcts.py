import math
from src.mcts import MCTSNode, MCTSEngine


class MockModelInterface:
    def expand_candidates(self, state: str, k: int = 3):
        if state == "ROOT":
            return [
                ("STEP_A_TRAP", "STATE_TRAP", 0.75, False),
                ("STEP_B_GOOD", "STATE_GOOD", 0.25, False),
            ]
        if state == "STATE_TRAP":
            return [("TRAP_END", "DEAD", 1.0, False)]
        if state == "STATE_GOOD":
            return [("SOLVE", "GOAL <EOS>", 1.0, True)]
        return []

    def evaluate_prm(self, states: list[str]) -> list[float]:
        scores = {
            "STATE_TRAP": 0.04,
            "STATE_GOOD": 0.94,
            "DEAD": 0.01,
            "GOAL <EOS>": 1.00,
        }
        return [scores.get(s, 0.5) for s in states]


def test_puct_formula_exactness():
    parent = MCTSNode(state="ROOT", prior=1.0)
    parent.visit_count = 10
    child = MCTSNode(state="STATE_A", parent=parent, prior=0.5)
    child.visit_count = 2
    child.value_sum = 1.5

    c_puct = 1.5
    expected_q = 1.5 / 2
    expected_u = c_puct * 0.5 * (math.sqrt(10) / (1 + 2))
    assert math.isclose(child.puct_score(c_puct), expected_q + expected_u, rel_tol=1e-6)


def test_backpropagation_invariants():
    root = MCTSNode(state="ROOT", prior=1.0)
    c1 = MCTSNode(state="C1", parent=root, prior=0.5)
    c2 = MCTSNode(state="C2", parent=c1, prior=0.5)

    engine = MCTSEngine(model=MockModelInterface(), c_puct=1.4)
    engine.backpropagate(c2, value=0.8)

    assert c2.visit_count == 1 and math.isclose(c2.value_sum, 0.8)
    assert c1.visit_count == 1 and math.isclose(c1.value_sum, 0.8)
    assert root.visit_count == 1 and math.isclose(root.value_sum, 0.8)


def test_mcts_escapes_high_prior_trap():
    engine = MCTSEngine(model=MockModelInterface(), c_puct=1.4)
    solved, trace, root = engine.search(root_state="ROOT", max_simulations=6)

    assert solved is True
    assert trace == ["STEP_B_GOOD", "SOLVE"]
    assert root.children["STEP_B_GOOD"].visit_count >= root.children["STEP_A_TRAP"].visit_count