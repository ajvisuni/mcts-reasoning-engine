import random
from typing import List, Tuple
from src.mcts import ModelInterface


def run_greedy(model: ModelInterface, root_state: str, max_steps: int = 3) -> Tuple[bool, List[str]]:
    curr = root_state
    trace = []
    for _ in range(max_steps):
        cands = model.expand_candidates(curr, k=1)
        if not cands:
            return False, trace
        action_s, next_s, _, is_solved = cands[0]
        trace.append(action_s)
        curr = next_s
        if is_solved:
            return True, trace
        if curr.strip().endswith("<EOS>"):
            break
    return False, trace


def run_best_of_n_prm(
    model: ModelInterface,
    root_state: str,
    n_trajectories: int = 4,
    max_steps: int = 3,
    seed: int = 42,
) -> Tuple[bool, List[str]]:
    best_trace: List[str] = []
    best_score = -1.0
    rng = random.Random(seed)
    n_runs = max(1, n_trajectories)

    for idx in range(n_runs):
        curr = root_state
        trace = []
        visited = []
        solved = False

        for _ in range(max_steps):
            cands = model.expand_candidates(curr, k=1 if idx == 0 else 4)
            if not cands:
                break
            if idx == 0:
                chosen = cands[0]
            else:
                weights = [max(c[2], 1e-4) for c in cands]
                chosen = rng.choices(cands, weights=weights, k=1)[0]

            action_s, next_s, _, is_solved = chosen
            trace.append(action_s)
            curr = next_s
            visited.append(curr)
            if is_solved:
                solved = True
                break
            if curr.strip().endswith("<EOS>"):
                break

        if solved:
            return True, trace

        if n_runs == 1:
            return False, trace

        if visited:
            traj_score = min(model.evaluate_prm(visited))
            if traj_score > best_score:
                best_score = traj_score
                best_trace = trace

    return False, best_trace