import math
from src.mcts import MCTSNode, MCTSEngine
from src.baselines import run_greedy, run_best_of_n_prm
from src.env import format_root_state, parse_latest_state, generate_random_valid_moves, verify_and_apply_step


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


class UniformCountdownInterface:
    def expand_candidates(self, state: str, k: int = 4):
        target, nums, is_term = parse_latest_state(state)
        if is_term or len(nums) <= 1:
            return []
        moves = generate_random_valid_moves(nums, target)[:k]
        out = []
        prior = 1.0 / max(1, len(moves))
        for step_s, _ in moves:
            ok, nxt_s, _, solved = verify_and_apply_step(state, step_s)
            if ok:
                out.append((step_s, nxt_s, prior, solved))
        return out

    def evaluate_prm(self, states: list[str]) -> list[float]:
        return [0.5 for _ in states]


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


def test_mcts_handles_empty_and_duplicate_candidates():
    class EmptyInterface:
        def expand_candidates(self, state: str, k: int = 4):
            return []

        def evaluate_prm(self, states: list[str]) -> list[float]:
            return []

    solved, trace, root = MCTSEngine(model=EmptyInterface()).search("ROOT", max_simulations=4)
    assert solved is False
    assert trace == []
    assert root.is_terminal is True

    class DupInterface:
        def expand_candidates(self, state: str, k: int = 4):
            return [
                ("SAME", "S1 <EOS>", 0.5, True),
                ("SAME", "S2 <EOS>", 0.5, False),
            ]

        def evaluate_prm(self, states: list[str]) -> list[float]:
            return [1.0, 0.0]

    solved2, trace2, root2 = MCTSEngine(model=DupInterface()).search("ROOT", max_simulations=2)
    assert solved2 is True
    assert trace2 == ["SAME"]
    assert len(root2.children) == 1


def test_mcts_on_real_countdown_env_and_mermaid_escaping():
    root_s = format_root_state([12, 2], 24)
    engine = MCTSEngine(model=UniformCountdownInterface(), branch_factor=4)
    solved, trace, root = engine.search(root_s, max_simulations=6)
    assert solved is True
    assert len(trace) == 1

    mermaid = MCTSEngine.to_mermaid(root)
    assert "graph TD" in mermaid
    assert "&lt;EOS&gt;" in mermaid


def test_baselines_greedy_and_best_of_n():
    mock = MockModelInterface()
    g_ok, g_trace = run_greedy(mock, "ROOT")
    assert g_ok is False
    assert g_trace[0] == "STEP_A_TRAP"

    bon_ok, bon_trace = run_best_of_n_prm(mock, "ROOT", n_trajectories=8, seed=42)
    assert bon_ok is True
    assert bon_trace == ["STEP_B_GOOD", "SOLVE"]