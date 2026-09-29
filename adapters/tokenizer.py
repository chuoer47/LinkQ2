"""Tokenizer adapter: HF tokenizers -> qslab engine interface."""
# The engine core only sees encode()/decode()/vocab_size, never the HF object.
from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer


class QwenTokenizerAdapter:
    def __init__(self, model_path: Path | str):
        self._tok = AutoTokenizer.from_pretrained(str(model_path))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    def apply_chat(self, messages: list[dict[str, str]]) -> str:
        return self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
