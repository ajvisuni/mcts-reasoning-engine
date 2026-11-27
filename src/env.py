import random
import re
from typing import List, Optional, Tuple

_EQ_RE = re.compile(r"^(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)$")
_ARR_RE = re.compile(r"^\[\s*\d+(?:\s*,\s*\d+)*\s*\]$")


def solve_countdown(nums: List[int], target: int) -> Optional[List[str]]:
    if len(nums) == 1:
        return [] if nums[0] == target else None

    nums_sorted = sorted(nums, reverse=True)
    n = len(nums_sorted)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = nums_sorted[i], nums_sorted[j]
            rem = [nums_sorted[k] for k in range(n) if k != i and k != j]

            ops = [("+", a + b), ("*", a * b)]
            if a - b > 0:
                ops.append(("-", a - b))
            if b > 0 and a % b == 0:
                ops.append(("/", a // b))

            for op, res in ops:
                nxt = sorted(rem + [res], reverse=True)
                arr_s = "[" + ",".join(map(str, nxt)) + "]"
                if len(nxt) == 1 and nxt[0] == target:
                    return [f"{a}{op}{b}={res} | {arr_s} <EOS>"]
                if len(nxt) > 1:
                    sub = solve_countdown(nxt, target)
                    if sub is not None:
                        return [f"{a}{op}{b}={res} | {arr_s} ->"] + sub
    return None


def is_state_solvable(nums: List[int], target: int) -> bool:
    return solve_countdown(nums, target) is not None


def format_root_state(nums: List[int], target: int) -> str:
    arr_s = "[" + ",".join(map(str, sorted(nums, reverse=True))) + "]"
    return f"TARGET:{target} | {arr_s} ->"


def parse_latest_state(state_str: str) -> Tuple[int, List[int], bool]:
    s = state_str.strip()
    is_term = s.endswith("<EOS>")
    header = s.split("|")[0].strip()
    if "TARGET:" not in header:
        raise ValueError(f"Missing target: {state_str}")
    target = int(header.split("TARGET:", 1)[1].strip())

    open_idx = s.rfind("[")
    close_idx = s.rfind("]")
    if open_idx == -1 or close_idx == -1 or close_idx < open_idx:
        raise ValueError(f"Bad state: {state_str}")

    inside = s[open_idx + 1 : close_idx].strip()
    if not inside:
        raise ValueError(f"Empty number list: {state_str}")
    nums = [int(x.strip()) for x in inside.split(",")]
    return target, nums, is_term


def verify_and_apply_step(state_str: str, step_str: str) -> Tuple[bool, str, bool, bool]:
    try:
        target, cur_nums, was_term = parse_latest_state(state_str)
        if was_term:
            return False, state_str, True, False

        step = step_str.strip()
        if step.endswith("<EOS>"):
            is_eos = True
            body = step[:-5].strip()
        elif step.endswith("->"):
            is_eos = False
            body = step[:-2].strip()
        else:
            return False, state_str, False, False

        parts = body.split("|")
        if len(parts) != 2:
            return False, state_str, False, False

        eq_part = parts[0].strip()
        rhs_part = parts[1].strip()

        eq_match = _EQ_RE.match(eq_part)
        if not eq_match or not _ARR_RE.match(rhs_part):
            return False, state_str, False, False

        a = int(eq_match.group(1))
        op_used = eq_match.group(2)
        b = int(eq_match.group(3))
        res = int(eq_match.group(4))

        avail = list(cur_nums)
        if a not in avail:
            return False, state_str, False, False
        avail.remove(a)
        if b not in avail:
            return False, state_str, False, False
        avail.remove(b)

        if op_used == "+" and a + b != res:
            return False, state_str, False, False
        if op_used == "-" and (a - b != res or res <= 0):
            return False, state_str, False, False
        if op_used == "*" and a * b != res:
            return False, state_str, False, False
        if op_used == "/" and (b == 0 or a % b != 0 or a // b != res):
            return False, state_str, False, False

        expected = sorted(avail + [res], reverse=True)
        stated = [int(x.strip()) for x in rhs_part[1:-1].split(",")]

        if sorted(stated, reverse=True) != expected:
            return False, state_str, False, False

        is_term = len(expected) == 1 or is_eos
        is_solved = len(expected) == 1 and expected[0] == target and is_eos
        return True, f"{state_str.strip()} {step}", is_term, is_solved
    except Exception:
        return False, state_str, False, False


def generate_random_valid_moves(nums: List[int], target: int) -> List[Tuple[str, List[int]]]:
    nums_sorted = sorted(nums, reverse=True)
    n = len(nums_sorted)
    moves = []
    seen = set()

    for i in range(n):
        for j in range(i + 1, n):
            a, b = nums_sorted[i], nums_sorted[j]
            rem = [nums_sorted[k] for k in range(n) if k != i and k != j]
            ops = [("+", a + b), ("*", a * b)]
            if a - b > 0:
                ops.append(("-", a - b))
            if b > 0 and a % b == 0:
                ops.append(("/", a // b))

            for op, res in ops:
                nxt = sorted(rem + [res], reverse=True)
                arr_s = "[" + ",".join(map(str, nxt)) + "]"
                suffix = "<EOS>" if len(nxt) == 1 else "->"
                step_s = f"{a}{op}{b}={res} | {arr_s} {suffix}"
                if step_s not in seen:
                    seen.add(step_s)
                    moves.append((step_s, nxt))
    return moves


def generate_dataset(num_problems: int = 4000, seed: int = 42) -> List[dict]:
    rng = random.Random(seed)
    seen = set()
    dataset = []
    pool = list(range(1, 13)) + [15, 20, 25]

    while len(dataset) < num_problems:
        nums = sorted([rng.choice(pool) for _ in range(4)], reverse=True)
        target = rng.randint(12, 85)
        key = (tuple(nums), target)
        if key in seen or target in nums:
            continue

        sol = solve_countdown(nums, target)
        if sol is None:
            continue

        seen.add(key)
        root_s = format_root_state(nums, target)

        curr = root_s
        pos = []
        for s in sol:
            curr = f"{curr} {s}"
            pos.append((curr, 1.0))

        neg = []
        root_moves = generate_random_valid_moves(nums, target)
        rng.shuffle(root_moves)
        for move_s, nxt_nums in root_moves:
            if not is_state_solvable(nxt_nums, target):
                bad_state = f"{root_s} {move_s}"
                neg.append((bad_state, 0.0))
                sub_moves = generate_random_valid_moves(nxt_nums, target)
                if sub_moves:
                    sub_m, _ = rng.choice(sub_moves)
                    neg.append((f"{bad_state} {sub_m}", 0.0))
                break

        if len(pos) >= 2:
            d1_state = pos[0][0]
            _, d1_nums, _ = parse_latest_state(d1_state)
            d2_moves = generate_random_valid_moves(d1_nums, target)
            rng.shuffle(d2_moves)
            for move_s, nxt_nums in d2_moves:
                if not is_state_solvable(nxt_nums, target):
                    neg.append((f"{d1_state} {move_s}", 0.0))
                    break

        dataset.append({
            "nums": nums,
            "target": target,
            "root_state": root_s,
            "full_solution_text": f"{root_s} " + " ".join(sol),
            "prm_examples": pos + neg,
        })

    return dataset