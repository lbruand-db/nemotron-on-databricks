"""nemotron-ocr-v1 recognizer token space and target encoding.

The recognizer emits 32 positions over 858 tokens: 0 = blank, 1 = EOS, 2 = rare, 3.. = `charset.txt`
(855 entries). With the default settings, NVIDIA's inference decodes **per position** (argmax; skip
blanks; stop at EOS; no CTC duplicate merging), and its training targets are `text + EOS` with a mask
(`RecognitionTargetEncoder`). So fine-tuning uses masked per-position cross-entropy over
[c_1 .. c_L, EOS], and positions after EOS are ignored.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

BLANK, EOS, RARE = 0, 1, 2
NUM_SPECIAL = 3
SEQ_LEN = 32
IGNORE = -100


class Charset:
    def __init__(self, chars: Sequence[str]) -> None:
        self.chars = list(chars)
        self.char_to_idx = {c: i + NUM_SPECIAL for i, c in enumerate(self.chars)}
        self.idx_to_char = {i + NUM_SPECIAL: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_file(cls, path: str | Path) -> Charset:
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    @property
    def num_tokens(self) -> int:
        return NUM_SPECIAL + len(self.chars)

    def oov_chars(self, text: str) -> list[str]:
        return [c for c in text if c not in self.char_to_idx]

    def trainable(self, text: str, seq_len: int = SEQ_LEN) -> bool:
        """Text fits (L + EOS <= seq_len) and every character is in the charset."""
        return 0 < len(text) < seq_len and not self.oov_chars(text)

    def encode(self, texts: Sequence[str], seq_len: int = SEQ_LEN) -> np.ndarray:
        """[N, seq_len] int64 targets: char ids, then EOS, then IGNORE."""
        out = np.full((len(texts), seq_len), IGNORE, dtype=np.int64)
        for i, text in enumerate(texts):
            if not self.trainable(text, seq_len):
                raise ValueError(f"Untrainable text {text!r}")
            ids = [self.char_to_idx[c] for c in text]
            out[i, : len(ids)] = ids
            out[i, len(ids)] = EOS
        return out

    def decode(self, ids: Sequence[int]) -> str:
        """NVIDIA's `decode_sequence` (remove_duplicates=False): skip blank, stop at EOS, rare -> '^'."""
        text = []
        for t in ids:
            t = int(t)
            if t == BLANK:
                continue
            if t == EOS:
                break
            text.append("^" if t == RARE else self.idx_to_char.get(t, "^"))
        return "".join(text)
