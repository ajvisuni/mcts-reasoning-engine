import pytest
from src.env import (
    solve_countdown,
    is_state_solvable,
    format_root_state,
    parse_latest_state,
    verify_and_apply_step,
    generate_dataset,
)


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
    assert is_state_solvable([2, 12], target=24) is True


def test_step_verifier_rejects_hallucinated_math():
    root = format_root_state([8, 7, 3, 2], 24)
    valid, _, _, _ = verify_and_apply_step(root, "7-3=5 | [8,5,2] ->")
    assert valid is False

    valid2, _, _, _ = verify_and_apply_step(root, "9*3=27 | [27,8,7,2] ->")
    assert valid2 is False


def test_step_verifier_arithmetic_and_format_edge_cases():
    root = format_root_state([8, 7, 3, 2], 24)
    # Non-positive subtraction and non-exact division
    assert verify_and_apply_step(root, "3-7=-4 | [8,2,-4] ->")[0] is False
    assert verify_and_apply_step(root, "7-7=0 | [8,3,2,0] ->")[0] is False
    assert verify_and_apply_step(root, "7/2=3 | [8,3,3] ->")[0] is False
    # Malformed equation or trailing junk
    assert verify_and_apply_step(root, "8+7*2=22 | [22,3] ->")[0] is False
    assert verify_and_apply_step(root, "8+7=15 | [15,3,2] extra ->")[0] is False
    # Reject reusing a single number twice
    assert verify_and_apply_step(root, "8+8=16 | [16,7,3] ->")[0] is False

    # Allow using repeated numbers when two copies exist
    dup_root = format_root_state([6, 6, 2, 1], 24)
    assert verify_and_apply_step(dup_root, "6+6=12 | [12,2,1] ->")[0] is True


def test_parse_latest_state_and_terminal_rejection():
    with pytest.raises(ValueError):
        parse_latest_state("TARGET:24 | [] ->")
    with pytest.raises(ValueError):
        parse_latest_state("BAD_HEADER | [8,7] ->")

    term_state = "TARGET:24 | [24] <EOS>"
    ok, _, is_term, is_solved = verify_and_apply_step(term_state, "24+0=24 | [24] <EOS>")
    assert ok is False and is_term is True and is_solved is False


def test_dataset_determinism_and_uniqueness():
    ds_a = generate_dataset(num_problems=20, seed=123)
    ds_b = generate_dataset(num_problems=20, seed=123)
    assert [r["root_state"] for r in ds_a] == [r["root_state"] for r in ds_b]
    keys = {(tuple(sorted(r["nums"])), r["target"]) for r in ds_a}
    assert len(keys) == 20