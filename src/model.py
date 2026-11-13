import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tokenizer import CountdownTokenizer
from src.env import verify_and_apply_step, parse_latest_state, generate_random_valid_moves


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int = 256, dropout: float = 0.05):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.dropout(F.softmax(att, dim=-1))

        out = (att @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int = 256, dropout: float = 0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DualHeadReasoningTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, max_len, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.prm_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.forward_hidden(idx)
        return self.lm_head(h), self.prm_head(h).squeeze(-1)

    @torch.no_grad()
    def score_states_prm(
        self, states: List[str], tokenizer: CountdownTokenizer, device: torch.device
    ) -> List[float]:
        if not states:
            return []
        self.eval()
        encoded = [tokenizer.encode(s, add_bos=True)[: self.max_len] for s in states]
        lengths = [len(seq) for seq in encoded]
        max_t = max(lengths)

        batch = torch.full((len(states), max_t), tokenizer.pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(encoded):
            batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        _, prm_logits = self.forward(batch)
        probs = torch.sigmoid(prm_logits)
        return [float(probs[i, lengths[i] - 1].item()) for i in range(len(states))]


class TorchPolicyPRMInterface:
    def __init__(
        self,
        model: DualHeadReasoningTransformer,
        tokenizer: CountdownTokenizer,
        device: torch.device,
        temperature: float = 0.85,
        max_step_tokens: int = 28,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.max_step_tokens = max_step_tokens

    @torch.no_grad()
    def expand_candidates(self, state: str, k: int = 4) -> List[Tuple[str, str, float, bool]]:
        self.model.eval()
        try:
            target, cur_nums, is_term = parse_latest_state(state)
        except Exception:
            return []

        if is_term or len(cur_nums) <= 1:
            return []

        valid_moves = generate_random_valid_moves(cur_nums, target)
        if not valid_moves:
            return []

        prefix_ids = self.tokenizer.encode(state, add_bos=True)
        prefix_len = len(prefix_ids)

        seqs = []
        meta = []
        for step_str, _ in valid_moves:
            ok, next_state, _, is_solved = verify_and_apply_step(state, step_str)
            if not ok:
                continue
            full_ids = self.tokenizer.encode(next_state, add_bos=True)[: self.model.max_len]
            if len(full_ids) > prefix_len:
                seqs.append(full_ids)
                meta.append((step_str, next_state, is_solved))

        if not seqs:
            return []

        max_t = max(len(s) for s in seqs)
        batch = torch.full((len(seqs), max_t), self.tokenizer.pad_id, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            batch[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)

        logits, _ = self.model(batch)
        log_probs = F.log_softmax(logits[:, :-1, :] / max(self.temperature, 1e-4), dim=-1)
        token_lps = log_probs.gather(2, batch[:, 1:].unsqueeze(-1)).squeeze(-1)

        scored = []
        for i, (step_str, next_state, is_solved) in enumerate(meta):
            seq_len = len(seqs[i])
            step_lp = token_lps[i, prefix_len - 1 : seq_len - 1]
            mean_lp = float(step_lp.mean().item()) if step_lp.numel() > 0 else -5.0
            scored.append([step_str, next_state, math.exp(mean_lp), is_solved])

        scored.sort(key=lambda x: x[2], reverse=True)
        top_k = scored[:k]

        total_p = sum(r[2] for r in top_k)
        if total_p > 0:
            for r in top_k:
                r[2] /= total_p

        return [(r[0], r[1], r[2], r[3]) for r in top_k]

    @torch.no_grad()
    def evaluate_prm(self, states: List[str]) -> List[float]:
        return self.model.score_states_prm(states, self.tokenizer, self.device)