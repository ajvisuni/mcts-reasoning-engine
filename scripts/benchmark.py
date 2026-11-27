import argparse
from pathlib import Path
import re
import sys
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import CountdownTokenizer
from src.model import DualHeadReasoningTransformer, TorchPolicyPRMInterface
from src.mcts import MCTSEngine, ModelInterface
from src.baselines import run_greedy, run_best_of_n_prm
from src.train import get_device

_ROOT_RE = re.compile(r"TARGET:\s*(-?\d+)\s*\|\s*\[([^\]]+)\]")


class ComputeCounter:
    def __init__(self, inner: ModelInterface):
        self.inner = inner
        self.expand_calls = 0
        self.prm_calls = 0

    def reset(self) -> None:
        self.expand_calls = 0
        self.prm_calls = 0

    @property
    def total_calls(self) -> int:
        return self.expand_calls + self.prm_calls

    def expand_candidates(self, state: str, k: int = 4) -> List[Tuple[str, str, float, bool]]:
        self.expand_calls += 1
        return self.inner.expand_candidates(state, k=k)

    def evaluate_prm(self, states: List[str]) -> List[float]:
        self.prm_calls += 1
        return self.inner.evaluate_prm(states)


def canonical_problem_key(root_state: str):
    match = _ROOT_RE.search(root_state)
    if not match:
        return root_state.strip()
    target = int(match.group(1))
    nums = tuple(sorted(int(x.strip()) for x in match.group(2).split(",") if x.strip()))
    return (target, nums)


def deduplicate_problems(problems: List[dict]) -> List[dict]:
    seen = set()
    unique = []
    for prob in problems:
        key = canonical_problem_key(prob["root_state"])
        if key not in seen:
            seen.add(key)
            unique.append(prob)
    return unique


def run_scaling_sweep(
    checkpoint_path: str = str(PROJECT_ROOT / "data" / "checkpoint.pt"),
    num_test: int = 25,
):
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    device = get_device()
    tokenizer = CountdownTokenizer()

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    model = DualHeadReasoningTransformer(
        vocab_size=tokenizer.vocab_size, d_model=192, n_layers=4, n_heads=4, max_len=128
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    problems = deduplicate_problems(ckpt["test_problems"])[:num_test]
    if not problems:
        raise ValueError("No test problems found in checkpoint.")

    print(f"Loaded checkpoint ({n_params:,} params) on {device}")
    print(f"Evaluating on {len(problems)} unique test problems")

    base_interface = TorchPolicyPRMInterface(model, tokenizer, device, temperature=0.85)
    counter = ComputeCounter(base_interface)

    bon_counts = [1, 2, 4, 8, 16]
    mcts_sims = [4, 8, 16, 32]

    greedy_hits = 0
    greedy_expand = 0

    bon_hits = {n: 0 for n in bon_counts}
    bon_expand = {n: 0 for n in bon_counts}
    bon_prm = {n: 0 for n in bon_counts}

    mcts_hits = {s: 0 for s in mcts_sims}
    mcts_expand = {s: 0 for s in mcts_sims}
    mcts_prm = {s: 0 for s in mcts_sims}

    saved_mermaid = None
    saved_header = None

    for idx, prob in enumerate(problems):
        root_s = prob["root_state"]
        print(f"[{idx+1}/{len(problems)}] {root_s}")

        counter.reset()
        g_ok, _ = run_greedy(counter, root_s)
        greedy_hits += int(g_ok)
        greedy_expand += counter.expand_calls

        for n in bon_counts:
            counter.reset()
            bon_ok, _ = run_best_of_n_prm(counter, root_s, n_trajectories=n)
            bon_hits[n] += int(bon_ok)
            bon_expand[n] += counter.expand_calls
            bon_prm[n] += counter.prm_calls

        for s in mcts_sims:
            counter.reset()
            engine = MCTSEngine(model=counter, c_puct=1.4, branch_factor=4)
            m_ok, _, root_node = engine.search(root_state=root_s, max_simulations=s)
            mcts_hits[s] += int(m_ok)
            mcts_expand[s] += counter.expand_calls
            mcts_prm[s] += counter.prm_calls

            if not g_ok and m_ok and saved_mermaid is None:
                saved_mermaid = MCTSEngine.to_mermaid(root_node)
                saved_header = root_s

    total = len(problems)
    greedy_pct = 100.0 * greedy_hits / total
    greedy_calls = greedy_expand / total

    bon_x = [(bon_expand[n] + bon_prm[n]) / total for n in bon_counts]
    bon_pcts = [100.0 * bon_hits[n] / total for n in bon_counts]

    mcts_x = [(mcts_expand[s] + mcts_prm[s]) / total for s in mcts_sims]
    mcts_pcts = [100.0 * mcts_hits[s] / total for s in mcts_sims]

    print(f"\nGreedy: {greedy_hits}/{total} ({greedy_pct:.1f}%) | Mean calls: {greedy_calls:.1f}")

    header = f"{'Setting':<10} | {'Expand':<8} | {'PRM':<8} | {'Total':<8} | {'Solved':<10} | {'Pass@1 (%)':<10}"
    print(f"\nBest-of-N + PRM\n{header}\n" + "-" * len(header))
    for i, n in enumerate(bon_counts):
        solved = f"{bon_hits[n]}/{total}"
        print(
            f"N={n:<8} | {bon_expand[n]/total:<8.1f} | {bon_prm[n]/total:<8.1f} | "
            f"{bon_x[i]:<8.1f} | {solved:<10} | {bon_pcts[i]:<10.1f}"
        )

    print(f"\nMCTS + PRM\n{header}\n" + "-" * len(header))
    for i, s in enumerate(mcts_sims):
        solved = f"{mcts_hits[s]}/{total}"
        print(
            f"S={s:<8} | {mcts_expand[s]/total:<8.1f} | {mcts_prm[s]/total:<8.1f} | "
            f"{mcts_x[i]:<8.1f} | {solved:<10} | {mcts_pcts[i]:<10.1f}"
        )

    assets_dir = PROJECT_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    plot_path = assets_dir / "scaling_law.png"

    plt.figure(figsize=(8.5, 5.2), dpi=200)
    plt.plot(mcts_x, mcts_pcts, marker="o", linewidth=2.6, color="#2563eb", label="MCTS + PRM (PUCT)")
    plt.plot(bon_x, bon_pcts, marker="s", linewidth=2.0, linestyle="--", color="#f59e0b", label="Best-of-N + PRM")
    plt.scatter([greedy_calls], [greedy_pct], s=70, zorder=5, color="#374151", label=f"Greedy ({greedy_pct:.0f}%)")

    plt.xscale("log", base=2)
    ticks = [2, 4, 8, 16, 32, 64]
    plt.xticks(ticks, [str(t) for t in ticks])
    plt.xlabel("Mean Model Forward Calls per Problem (Log2 Scale)", fontsize=11, fontweight="bold")
    plt.ylabel(f"Countdown Pass@1 Accuracy (% over N={total})", fontsize=11, fontweight="bold")
    plt.title("Test-Time Compute Scaling: PRM-Guided MCTS vs. Best-of-N", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    if saved_mermaid:
        ex_dir = PROJECT_ROOT / "examples"
        ex_dir.mkdir(parents=True, exist_ok=True)
        with open(ex_dir / "backtrack_trace.md", "w", encoding="utf-8") as f:
            f.write(f"# MCTS Backtracking Trace\n\n**Problem:** `{saved_header}`\n\n```mermaid\n{saved_mermaid}\n```\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "data" / "checkpoint.pt"))
    parser.add_argument("--num-test", type=int, default=25)
    args = parser.parse_args()
    run_scaling_sweep(checkpoint_path=args.checkpoint, num_test=args.num_test)