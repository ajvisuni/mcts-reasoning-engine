import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.tokenizer import CountdownTokenizer
from src.model import DualHeadReasoningTransformer, TorchPolicyPRMInterface
from src.mcts import MCTSEngine
from src.baselines import run_greedy, run_best_of_n_prm
from src.train import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_scaling_sweep(
    checkpoint_path: str = str(PROJECT_ROOT / "data" / "checkpoint.pt"),
    num_test: int = 25,
):
    device = get_device()
    tokenizer = CountdownTokenizer()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DualHeadReasoningTransformer(
        vocab_size=tokenizer.vocab_size, d_model=192, n_layers=4, n_heads=4, max_len=128
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    problems = ckpt["test_problems"][:num_test]
    interface = TorchPolicyPRMInterface(model, tokenizer, device, temperature=0.85)

    budgets = [1, 4, 8, 16, 32]
    greedy_hits = 0
    bon_hits = {b: 0 for b in budgets}
    mcts_hits = {b: 0 for b in budgets}

    saved_mermaid = None
    saved_header = None

    for idx, prob in enumerate(problems):
        root_s = prob["root_state"]
        print(f"[{idx+1}/{len(problems)}] {root_s}")

        g_ok, _ = run_greedy(interface, root_s)
        if g_ok:
            greedy_hits += 1

        for b in budgets:
            if b == 1:
                bon_hits[b] += int(g_ok)
                mcts_hits[b] += int(g_ok)
                continue

            bon_ok, _ = run_best_of_n_prm(interface, root_s, n_trajectories=max(1, b // 3))
            bon_hits[b] += int(bon_ok)

            engine = MCTSEngine(model=interface, c_puct=1.4, branch_factor=4)
            m_ok, _, root_node = engine.search(root_state=root_s, max_simulations=b // 2)
            mcts_hits[b] += int(m_ok)

            if not g_ok and m_ok and saved_mermaid is None:
                saved_mermaid = MCTSEngine.to_mermaid(root_node)
                saved_header = root_s

    total = max(1, len(problems))
    greedy_pct = 100.0 * greedy_hits / total
    bon_pcts = [100.0 * bon_hits[b] / total for b in budgets]
    mcts_pcts = [100.0 * mcts_hits[b] / total for b in budgets]

    print(f"\n{'Budget (C)':<12} | {'Greedy':<10} | {'Best-of-N + PRM':<18} | {'MCTS + PRM':<18}")
    print("-" * 64)
    for i, b in enumerate(budgets):
        print(f"{b:<12} | {greedy_pct:<10.1f} | {bon_pcts[i]:<18.1f} | {mcts_pcts[i]:<18.1f}")

    assets_dir = PROJECT_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    plot_path = assets_dir / "scaling_law.png"

    plt.figure(figsize=(8.5, 5.2), dpi=200)
    x_pos = np.log2(budgets)

    plt.plot(x_pos, mcts_pcts, marker="o", linewidth=2.6, color="#2563eb", label="System-2 MCTS + PRM (PUCT)")
    plt.plot(x_pos, bon_pcts, marker="s", linewidth=2.0, linestyle="--", color="#f59e0b", label="Best-of-N + PRM Reranking")
    plt.axhline(greedy_pct, linestyle=":", linewidth=1.8, color="#6b7280", label=f"System-1 Greedy ({greedy_pct:.1f}%)")

    plt.xticks(x_pos, [f"{b}" for b in budgets])
    plt.xlabel("Test-Time Compute Budget (Step Expansions - Log2 Scale)", fontsize=11, fontweight="bold")
    plt.ylabel("Countdown Pass@1 Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Test-Time Compute Scaling: PRM-Guided MCTS vs. Parallel Sampling", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_path)

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