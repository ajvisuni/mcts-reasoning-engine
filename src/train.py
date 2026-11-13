import argparse
import os
import random
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.env import generate_dataset
from src.tokenizer import CountdownTokenizer
from src.model import DualHeadReasoningTransformer


class CountdownReasoningDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer: CountdownTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: List[Tuple[List[int], int, float, bool]] = []

        for rec in records:
            full_ids = tokenizer.encode(rec["full_solution_text"], add_bos=True)[:max_len]
            self.samples.append((full_ids, len(full_ids) - 1, 1.0, True))

            for state_text, label in rec["prm_examples"]:
                s_ids = tokenizer.encode(state_text, add_bos=True)[:max_len]
                self.samples.append((s_ids, len(s_ids) - 1, float(label), False))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, prm_pos, prm_label, is_policy = self.samples[idx]
        padded = seq + [self.tokenizer.pad_id] * (self.max_len - len(seq))
        return (
            torch.tensor(padded[: self.max_len], dtype=torch.long),
            torch.tensor(min(prm_pos, self.max_len - 1), dtype=torch.long),
            torch.tensor(prm_label, dtype=torch.float32),
            torch.tensor(1.0 if is_policy else 0.0, dtype=torch.float32),
        )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    num_problems: int = 4000,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 4e-4,
    save_path: str = "data/checkpoint.pt",
):
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = CountdownTokenizer()
    raw_data = generate_dataset(num_problems=num_problems, seed=42)
    random.Random(42).shuffle(raw_data)

    split = int(0.9 * len(raw_data))
    train_ds = CountdownReasoningDataset(raw_data[:split], tokenizer)
    val_ds = CountdownReasoningDataset(raw_data[split:], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DualHeadReasoningTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=192,
        n_layers=4,
        n_heads=4,
        max_len=128,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        lm_sum = 0.0
        prm_sum = 0.0

        for seq, prm_pos, prm_label, is_policy in train_loader:
            seq = seq.to(device)
            prm_pos = prm_pos.to(device)
            prm_label = prm_label.to(device)
            is_policy = is_policy.to(device)

            logits, prm_logits = model(seq)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = seq[:, 1:].contiguous()
            token_loss = F.cross_entropy(
                shift_logits.view(-1, tokenizer.vocab_size),
                shift_targets.view(-1),
                ignore_index=tokenizer.pad_id,
                reduction="none",
            ).view(seq.size(0), -1)
            seq_lm_loss = token_loss.mean(dim=-1)
            lm_loss = (seq_lm_loss * is_policy).sum() / (is_policy.sum() + 1e-6)

            b_idx = torch.arange(seq.size(0), device=device)
            pred_prm = prm_logits[b_idx, prm_pos]
            prm_loss = F.binary_cross_entropy_with_logits(pred_prm, prm_label)

            loss = lm_loss + prm_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            lm_sum += lm_loss.item()
            prm_sum += prm_loss.item()

        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seq, prm_pos, prm_label, _ in val_loader:
                seq, prm_pos, prm_label = seq.to(device), prm_pos.to(device), prm_label.to(device)
                _, prm_logits = model(seq)
                preds = (torch.sigmoid(prm_logits[torch.arange(seq.size(0), device=device), prm_pos]) > 0.5).float()
                correct += int((preds == prm_label).sum().item())
                total += seq.size(0)

        acc = 100.0 * correct / max(1, total)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"LM Loss: {lm_sum/len(train_loader):.4f} | "
            f"PRM Loss: {prm_sum/len(train_loader):.4f} | "
            f"Val PRM Acc: {acc:.2f}%"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "test_problems": raw_data[split:][:200],
        },
        save_path,
    )
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train_model(num_problems=args.problems, epochs=args.epochs, batch_size=args.batch_size)