from typing import List


class CountdownTokenizer:
    SPECIAL = ["<PAD>", "<BOS>", "<EOS>"]
    ATOMIC = ["TARGET:", "->"]
    CHARS = list("0123456789+-*/=|[],: ")

    def __init__(self):
        self.itos = self.SPECIAL + self.ATOMIC + self.CHARS
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi["<PAD>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.arrow_id = self.stoi["->"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        ids = [self.bos_id] if add_bos else []
        i = 0
        n = len(text)
        while i < n:
            hit = False
            for tok in ["<EOS>", "<BOS>", "<PAD>", "TARGET:", "->"]:
                if text.startswith(tok, i):
                    ids.append(self.stoi[tok])
                    i += len(tok)
                    hit = True
                    break
            if not hit:
                ch = text[i]
                if ch in self.stoi:
                    ids.append(self.stoi[ch])
                i += 1
        return ids

    def decode(self, ids: List[int], skip_special: bool = False) -> str:
        out = []
        for idx in ids:
            tok = self.itos[idx]
            if skip_special and tok in ("<PAD>", "<BOS>"):
                continue
            out.append(tok)
        return "".join(out)