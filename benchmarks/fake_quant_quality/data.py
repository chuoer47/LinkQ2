"""Prepare deterministic calibration and PPL token streams."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from benchmarks.fake_quant_quality.config import BenchmarkConfig


def _digest(ids: list[int]) -> str:
    return hashlib.sha256(torch.tensor(ids, dtype=torch.int32).numpy().tobytes()).hexdigest()


def prepare(config: BenchmarkConfig, force: bool = False) -> tuple[Path, Path]:
    cache = config.dataset_cache
    cache.mkdir(parents=True, exist_ok=True)
    calib_file = cache / "calibration.pt"
    eval_file = cache / "wikitext_test.pt"
    if calib_file.exists() and eval_file.exists() and not force:
        return calib_file, eval_file

    tokenizer = AutoTokenizer.from_pretrained(str(config.model), use_fast=True)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    calibration: list[list[int]] = []
    buf: list[int] = []
    for row in ds:
        buf.extend(tokenizer(row["text"], add_special_tokens=False).input_ids)
        while len(buf) >= config.calibration_seq_len:
            calibration.append(buf[:config.calibration_seq_len])
            buf = buf[config.calibration_seq_len:]
            if len(calibration) == config.calibration_samples:
                break
        if len(calibration) == config.calibration_samples:
            break
    if len(calibration) != config.calibration_samples:
        raise RuntimeError(f"C4 yielded only {len(calibration)} calibration sequences")

    eval_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join(text for text in eval_ds["text"] if text.strip())
    eval_ids = tokenizer(eval_text, add_special_tokens=False).input_ids
    if len(eval_ids) < config.max_length:
        raise RuntimeError("WikiText-2 token stream is shorter than the evaluation window")

    torch.save({"dataset": config.calibration_dataset,
                "token_ids": calibration}, calib_file)
    torch.save({"dataset": config.evaluation_dataset,
                "token_ids": eval_ids}, eval_file)
    manifest = {
        "tokenizer": str(config.model),
        "calibration_dataset": config.calibration_dataset,
        "calibration_sequences": len(calibration),
        "calibration_sequence_length": config.calibration_seq_len,
        "calibration_token_sha256": _digest([x for row in calibration for x in row]),
        "evaluation_dataset": config.evaluation_dataset,
        "evaluation_tokens": len(eval_ids),
        "evaluation_token_sha256": _digest(eval_ids),
    }
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return calib_file, eval_file
