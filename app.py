from pathlib import Path
import threading
import gradio as gr
import torch

from src.env import format_root_state, solve_countdown
from src.tokenizer import CountdownTokenizer
from src.model import DualHeadReasoningTransformer, TorchPolicyPRMInterface
from src.mcts import MCTSEngine
from src.baselines import run_greedy
from src.train import get_device

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "checkpoint.pt"

_INTERFACE = None
_LOCK = threading.Lock()


def load_interface() -> TorchPolicyPRMInterface:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint at {CHECKPOINT_PATH}. "
            "Download checkpoint.pt from the release or run `python -m src.train`."
        )

    device = get_device()
    tokenizer = CountdownTokenizer()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = DualHeadReasoningTransformer(
        vocab_size=tokenizer.vocab_size, d_model=192, n_layers=4, n_heads=4, max_len=128
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return TorchPolicyPRMInterface(model, tokenizer, device, temperature=0.85)


def get_interface() -> TorchPolicyPRMInterface:
    global _INTERFACE
    if _INTERFACE is None:
        with _LOCK:
            if _INTERFACE is None:
                _INTERFACE = load_interface()
    return _INTERFACE


def solve_interactive(nums_str: str, target_val: int, simulations: int, c_puct: float, branch_factor: int):
    try:
        nums = [int(x.strip()) for x in nums_str.split(",") if x.strip()]
        if len(nums) != 4 or any(n <= 0 or n > 999 for n in nums):
            return "Enter 4 positive integers between 1 and 999.", "", "", ""
        target = int(target_val)
        if target <= 0 or target > 9999:
            return "Target must be an integer between 1 and 9999.", "", "", ""

        sims = max(1, min(64, int(simulations)))
        cpuct = max(0.1, min(5.0, float(c_puct)))
        bf = max(1, min(8, int(branch_factor)))
    except Exception as e:
        return f"Invalid input: {e}", "", "", ""

    try:
        interface = get_interface()
    except Exception as e:
        return str(e), "", "", ""

    try:
        root_s = format_root_state(nums, target)
        oracle = solve_countdown(nums, target)
        oracle_s = " -> ".join(oracle) if oracle is not None else "No exact path exists"

        g_ok, g_trace = run_greedy(interface, root_s)
        g_steps = "\n".join(g_trace) if g_trace else "(None)"
        g_out = (
            f"{'SOLVED' if g_ok else 'FAILED'}\n"
            f"Root: {root_s}\n"
            f"Steps:\n{g_steps}\n\n"
            f"Oracle:\n{oracle_s}"
        )

        engine = MCTSEngine(model=interface, c_puct=cpuct, branch_factor=bf)
        m_ok, m_trace, root_node = engine.search(root_state=root_s, max_simulations=sims)
        m_steps = "\n".join(m_trace) if m_trace else "(None)"

        m_out = (
            f"{'SOLVED' if m_ok else 'UNSOLVED'}\n"
            f"Simulations: {sims} | c_puct: {cpuct} | Branch Factor: {bf}\n"
            f"Trajectory:\n{m_steps}"
        )

        ascii_tree = MCTSEngine.to_ascii_tree(root_node)
        mermaid_code = f"```mermaid\n{MCTSEngine.to_mermaid(root_node)}\n```"
        return g_out, m_out, ascii_tree, mermaid_code
    except Exception as e:
        return f"Search error: {e}", "", "", ""


with gr.Blocks(title="MCTS Reasoning Engine") as demo:
    gr.Markdown("# `mcts-reasoning-engine`: System-2 Test-Time Compute Explorer")

    with gr.Row():
        with gr.Column(scale=1):
            nums_in = gr.Textbox(label="4 Starting Numbers", value="8, 7, 3, 2")
            target_in = gr.Number(label="Target Number", value=24, precision=0)
            sim_sl = gr.Slider(minimum=1, maximum=64, value=16, step=1, label="Simulations")
            cpuct_sl = gr.Slider(minimum=0.2, maximum=3.0, value=1.4, step=0.1, label="c_puct")
            bf_sl = gr.Slider(minimum=2, maximum=6, value=4, step=1, label="Branch Factor (k)")
            run_btn = gr.Button("Run System-1 vs. System-2 Search", variant="primary")

            gr.Examples(
                examples=[
                    ["8, 7, 3, 2", 24, 16, 1.4, 4],
                    ["10, 8, 5, 2", 36, 16, 1.4, 4],
                    ["15, 9, 4, 3", 33, 24, 1.4, 4],
                    ["20, 6, 5, 2", 48, 24, 1.4, 4],
                ],
                inputs=[nums_in, target_in, sim_sl, cpuct_sl, bf_sl],
            )

        with gr.Column(scale=2):
            with gr.Row():
                g_box = gr.Textbox(label="System-1 Greedy", lines=5)
                m_box = gr.Textbox(label="System-2 MCTS + PRM", lines=5)

            tree_box = gr.Code(label="MCTS Search Tree", language=None, lines=14)
            mermaid_box = gr.Markdown(label="Search Graph")

    run_btn.click(
        fn=solve_interactive,
        inputs=[nums_in, target_in, sim_sl, cpuct_sl, bf_sl],
        outputs=[g_box, m_box, tree_box, mermaid_box],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)