# MCTS Reasoning Engine

[![Tests](https://github.com/ajvisuni/mcts-reasoning-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/ajvisuni/mcts-reasoning-engine/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ajvisuni/mcts-reasoning-engine)](https://github.com/ajvisuni/mcts-reasoning-engine/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Test-Time Compute Scaling with Process Reward Models (PRM) and PUCT Search**

![Test-Time Compute Scaling](assets/scaling_law.png)

I built this project to test how much inference-time search improves multi-step reasoning on a small language model without changing its weights.

Standard autoregressive models generate tokens left-to-right: if they make a bad arithmetic move on step one, they can't backtrack. Here, I paired a 1.7M-parameter dual-head causal Transformer with a PUCT Monte Carlo Tree Search engine on the **Countdown Numbers Game** (combining 4 integers with `+`, `-`, `*`, `/` to hit a target). Instead of committing to a single greedy rollout, the engine uses extra compute at inference time to explore intermediate states, score partial paths with a Process Reward Model (PRM), and back out of dead ends.

---

## Benchmark Results

All three inference strategies were evaluated using the same 1.7M-parameter checkpoint ([`v0.1.0`](https://github.com/ajvisuni/mcts-reasoning-engine/releases/tag/v0.1.0)) across compute budgets `C ∈ {1, 4, 8, 16, 32}` (measured in step expansions) on 25 held-out test instances (`seed=42`):

| Compute Budget (`C`) | Greedy (Single Pass) | Best-of-N + PRM | MCTS + PRM | Gain over Greedy |
| :--- | :---: | :---: | :---: | :---: |
| **C = 1** | 20% (5/25) | 20% (5/25) | 20% (5/25) | 0 pp (0/25) |
| **C = 4** | 20% (5/25) | 20% (5/25) | 4% (1/25) | -16 pp (-4/25) |
| **C = 8** | 20% (5/25) | 24% (6/25) | 20% (5/25) | 0 pp (0/25) |
| **C = 16** | 20% (5/25) | 32% (8/25) | 28% (7/25) | +8 pp (+2/25) |
| **C = 32** | 20% (5/25) | 36% (9/25) | **48% (12/25)** | **+28 pp (+7/25)** |

### What stood out in the runs
* **The dip at `C = 4`:** A 4-number Countdown puzzle takes 3 arithmetic steps (depth 3) to reach `<EOS>`. Because both the policy and PUCT selection are deterministic for a fixed puzzle, this drop isn't random seed variance. At `C = 4`, the exploration term spreads visits across competing depth-1 and depth-2 branches before any single path reaches depth 3.
* **The crossover at `C = 32`:** Once the search budget clears depth 3, MCTS stops spending expansions on low-PRM branches and focuses on viable subtrees. At `C = 32`, it hits **48% Pass@1 (12/25)**, beating Best-of-N + PRM (**36%, 9/25**) and more than doubling single-pass Greedy decoding (**20%, 5/25**).

---

## Example Trace: Escaping a Dead End

On `TARGET:24 | [8,7,3,2]`, greedy decoding picks the highest-probability first move (`8+3=11`) and gets stuck at `[29]`. MCTS checks that branch first, sees the leaf PRM score drop to `0.00`, and backtracks to `8+7=15` to solve the puzzle:

**Greedy (`C = 1`):**
```text
8+3=11 | [11,7,2] -> 11*2=22 | [22,7] -> 22+7=29 | [29] <EOS>  [FAILED]
```

**MCTS + PRM (`C = 16`, top two branches shown):**
```text
ROOT: TARGET:24 | [8,7,3,2] ->  [Visits N=9]
├── 8+3=11 | [11,7,2] ->    | Prior P=0.26 | PRM V=0.99 | Q=0.50 (N=2)  [EXPLORED]
│   └── 7*2=14 | [14,11] -> | Prior P=0.25 | PRM V=0.99 | Q=0.50 (N=2)
│       └── 14+11=25 | [25] <EOS> | Prior P=0.54 | PRM V=0.00 | Q=0.00  [DEAD-END]
└── 8+7=15 | [15,3,2] ->    | Prior P=0.26 | PRM V=0.22 | Q=0.61 (N=2)
    └── 15-3=12 | [12,2] -> | Prior P=0.26 | PRM V=0.23 | Q=0.61 (N=2)
        └── 12*2=24 | [24] <EOS> | Prior P=0.43 | PRM V=1.00 | Q=1.00  [SOLVED]
```

There's also a full Mermaid search graph exported from `TARGET:65 | [20,7,5,1] ->` in [`examples/backtrack_trace.md`](examples/backtrack_trace.md).

---

## How It Works

```text
State s_t (e.g. TARGET:24 | [8,7,3,2] ->)
   │
   ▼
Dual-Head Causal Transformer (4 Layers, d_model=192, ~1.7M Params)
   ├── Policy Head π(a | s_t)  ──► Candidate Step Priors P(s, a)
   └── Process Reward Head v(s_t) ──► Step Solvability Score ∈ [0, 1]
   │
   ▼
PUCT Monte Carlo Tree Search (Select ──► Expand ──► Backpropagate)
   │
   ▼
Exact Symbolic Step Verifier ──► Verified Solution (or Backtrack on Dead End)
```

1. **Environment & Exact Oracle (`src/env.py`):** States are plain text strings. During dataset generation, a recursive DFS solver (`solve_countdown`) labels intermediate states as solvable (`1.0`) or dead ends (`0.0`) so the PRM learns to spot traps early.
2. **Dual-Head Transformer (`src/model.py`):** A 4-layer causal Transformer (`vocab_size=28`, `d_model=192`) trained jointly on next-token cross-entropy and step-boundary binary cross-entropy (`prm_head`). At inference time, candidate next steps are scored in one batched forward pass to get length-normalized priors `P(s, a)`.
3. **PUCT Tree Search (`src/mcts.py`):** Pure-Python MCTS kept separate from PyTorch. At each node, it picks the child action `a` maximizing:

```text
PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

where `Q(s, a) = W(s, a) / N(s, a)` is the mean backed-up PRM score and `P(s, a)` is the policy prior.

---

## Project Structure

```text
mcts-reasoning-engine/
├── .github/workflows/ci.yml
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── app.py                     # Gradio UI for comparing Greedy vs. MCTS
├── assets/
│   └── scaling_law.png
├── examples/
│   └── backtrack_trace.md
├── scripts/
│   └── benchmark.py           # Compute sweep and plot generation
├── src/
│   ├── env.py                 # Countdown generator, step verifier, DFS oracle
│   ├── tokenizer.py           # 28-token vocabulary
│   ├── model.py               # Dual-head Transformer and batched inference
│   ├── mcts.py                # Pure-Python PUCT search and tree exporters
│   ├── baselines.py           # Greedy and Best-of-N PRM baselines
│   └── train.py               # Joint Policy + PRM training script
└── tests/
    ├── test_env.py
    ├── test_model.py
    └── test_mcts.py
```

---

## Running It Locally

Clone the repo, set up a virtual environment (`.venv\Scripts\activate` on Windows), install dependencies, and run the tests:

```bash
git clone https://github.com/ajvisuni/mcts-reasoning-engine.git
cd mcts-reasoning-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -v
```

To skip training, download `checkpoint.pt` from the [`v0.1.0` release](https://github.com/ajvisuni/mcts-reasoning-engine/releases/tag/v0.1.0) and put it at `data/checkpoint.pt`. If you'd rather train the 1.7M model from scratch:

```bash
python -m src.train --problems 4000 --epochs 10 --batch-size 128
```

Once the checkpoint is in place, run the scaling benchmark or start the Gradio viewer at `http://127.0.0.1:7860`:

```bash
python scripts/benchmark.py --num-test 25
python app.py
```

---

## Contributing & License

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for code standards and test requirements before opening a PR. Released under the [MIT License](LICENSE).
