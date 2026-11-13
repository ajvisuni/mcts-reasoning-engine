from src.env import solve_countdown, is_state_solvable, format_root_state, verify_and_apply_step


def test_exact_oracle_solves_known_instance():
    nums = [8, 7, 3, 2]
    target = 24
    trace = solve_countdown(nums, target)
    assert trace is not None
    assert len(trace) == 3

    state = format_root_state(nums, target)
    for step in trace:
        valid, state, is_term, is_solved = verify_and_apply_step(state, step)
        assert valid is True
    assert is_term is True
    assert is_solved is True


def test_oracle_detects_dead_end():
    assert is_state_solvable([4, 4], target=24) is False
    assert is_state_solvable([12, 2], target=24) is True


def test_step_verifier_rejects_hallucinated_math():
    root = format_root_state([8, 7, 3, 2], 24)
    valid, _, _, _ = verify_and_apply_step(root, "7-3=5 | [8,5,2] ->")
    assert valid is False

    valid2, _, _, _ = verify_and_apply_step(root, "9*3=27 | [27,8,7,2] ->")
    assert valid2 is False