# Contributing to MCTS Reasoning Engine

Thanks for checking out **`mcts-reasoning-engine`**! I built this repo to explore how a small, 1.7M-parameter Transformer can solve multi-step Countdown arithmetic problems by searching and backtracking at test time instead of just guessing left-to-right.

If you want to test out new tree-search ideas, speed up node expansions, or push the model on harder puzzles, I'd love to see your PRs.

---

## Getting It Running

Setting up the repo locally takes about a minute:

```bash
git clone https://github.com/ajvisuni/mcts-reasoning-engine.git
cd mcts-reasoning-engine
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -v
```

---

## Ground Rules (How This Repo Stays Clean)

I wrote this project from scratch so anyone reading the code can trace the exact math without digging through ten layers of wrapper classes. If you open a PR, please stick to these four rules:

1. **Keep `src/mcts.py` and `src/env.py` Pure Python:**
   These two files should only use the standard library (`math`, `random`, `typing`). Don't import `torch`, `numpy`, or `gradio` inside them. Any neural network calls belong behind the `ModelInterface` in `src/model.py`.

2. **Don't Break the Causal Mask (`src/model.py`):**
   When the Process Reward head (`prm_head`) scores a step at token `t`, it must never peek at future tokens `> t`. The test `test_causal_mask_zero_future_leakage` in `tests/test_model.py` checks this down to `1e-6`, and it needs to stay green.

3. **Compare Apples to Apples (`scripts/benchmark.py`):**
   If you add a new search baseline or tweak MCTS, measure compute in **step expansions (`C`)**, not wall-clock seconds. That way, parallel sampling (`Best-of-N`) and tree search (`MCTS`) are always judged on the exact same forward-pass budget.

4. **No Heavy Wrappers:**
   Please don't pull in external MCTS libraries, Gym environments, or high-level trainer frameworks. Keeping the PyTorch loop and PUCT math out in the open is the whole point of the repo.

---

## Sending a Pull Request

1. **Branch off `main`:** Give your branch a clear name like `feat/kv-cache` or `fix/puct-tiebreak`.

2. **Write clean, comment-free Python:** Use Python 3.10+ type hints and clear variable names so the code explains itself. Skip docstrings and obvious comments—only leave a short note if a tensor reshape isn't obvious at first glance.

3. **Keep junk out of Git:** Never commit model weights (`data/checkpoint.pt` or any `*.pt` file), virtual environments (`.venv/`), IDE folders (`.idea/`, `.vscode/`), or `__pycache__/`.

4. **Keep tests fast:** Add unit tests in `tests/` using `MockModelInterface` (like in `tests/test_mcts.py`) so `pytest -v` finishes in a few seconds on a regular CPU.

5. **Show the numbers:** If you change how PUCT picks nodes, how rollouts work, or how the PRM trains, run `python scripts/benchmark.py --num-test 25` and drop the new `Pass@1` table (`C = 1, 4, 8, 16, 32`) into your PR description so we can see if it actually beats the 48.0% baseline.

---

## Cool Things to Build Next

If you're looking for something fun to tackle that fits right into the current code:

* **Fix the `C = 4` cold-start dip:** Right now in `src/mcts.py`, `expand_and_evaluate` picks `max(node.children.values(), key=lambda c: c.prm_value + 0.2 * c.prior)` on the first visit, which makes `C = 4` dip below Greedy (`4.0%` vs `20.0%`) before climbing to `48.0%` at `C = 32`. Making the very first simulation follow the highest policy prior (`c.prior`) should keep the MCTS curve strictly above Greedy across every budget.

* **KV-Cache Prefix Sharing:** In `TorchPolicyPRMInterface.expand_candidates` (`src/model.py`), sibling moves all share the exact same parent prompt tokens. Caching the Transformer's key-value states for the parent prefix instead of recomputing the whole sequence would make tree expansion much faster.

* **Self-Play Loop (Expert Iteration):** Write a script that takes the hard problems MCTS solves at `C = 32` (where Greedy failed) and feeds those winning trajectories back into `src/train.py` to make the base model smarter.

* **5-Number Countdown:** Bump `src/env.py` up from 4 starting numbers to 5 and see how well the 4-layer Transformer's PRM holds up when the search tree goes 4 steps deep instead of 3.

---

Happy Building — **AJX** 😉