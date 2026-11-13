import torch
from src.tokenizer import CountdownTokenizer
from src.model import DualHeadReasoningTransformer


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