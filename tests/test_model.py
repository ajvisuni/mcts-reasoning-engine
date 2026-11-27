import math
import torch
from src.tokenizer import CountdownTokenizer
from src.model import DualHeadReasoningTransformer, TorchPolicyPRMInterface
from src.env import format_root_state, generate_dataset
from src.train import CountdownReasoningDataset


def test_causal_mask_zero_future_leakage():
    tokenizer = CountdownTokenizer()
    model = DualHeadReasoningTransformer(
        vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=2, max_len=64, dropout=0.0
    )
    model.eval()

    seq_a = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 11]], dtype=torch.long)
    seq_b = torch.tensor([[1, 5, 6, 7, 19, 20, 21, 22]], dtype=torch.long)

    with torch.no_grad():
        logits_a, prm_a = model(seq_a)
        logits_b, prm_b = model(seq_b)

    assert torch.allclose(logits_a[:, :4, :], logits_b[:, :4, :], atol=1e-6)
    assert torch.allclose(prm_a[:, :4], prm_b[:, :4], atol=1e-6)


def test_dual_head_gradient_flow():
    tokenizer = CountdownTokenizer()
    model = DualHeadReasoningTransformer(
        vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=2, max_len=64
    )
    x = torch.randint(0, tokenizer.vocab_size, (4, 16))
    logits, prm_logits = model(x)
    loss = logits.mean() + prm_logits.mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing grad: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN grad: {name}"


def test_interface_priors_and_terminal_state():
    tokenizer = CountdownTokenizer()
    model = DualHeadReasoningTransformer(
        vocab_size=tokenizer.vocab_size, d_model=32, n_layers=1, n_heads=2, max_len=128
    )
    iface = TorchPolicyPRMInterface(model, tokenizer, torch.device("cpu"))

    root = format_root_state([8, 7, 3, 2], 24)
    cands = iface.expand_candidates(root, k=4)
    assert len(cands) == 4
    assert math.isclose(sum(c[2] for c in cands), 1.0, rel_tol=1e-5)

    scores = iface.evaluate_prm([c[1] for c in cands])
    assert len(scores) == 4
    assert all(0.0 <= s <= 1.0 for s in scores)

    assert iface.expand_candidates("TARGET:24 | [24] <EOS>", k=4) == []


def test_dataset_padding_and_prm_positions():
    tokenizer = CountdownTokenizer()
    records = generate_dataset(num_problems=5, seed=7)
    ds = CountdownReasoningDataset(records, tokenizer, max_len=128)
    seq, prm_pos, prm_label, _ = ds[0]
    assert seq.shape == (128,)
    assert 0 <= int(prm_pos.item()) < 128
    assert seq[int(prm_pos.item())].item() != tokenizer.pad_id
    assert prm_label.item() in (0.0, 1.0)