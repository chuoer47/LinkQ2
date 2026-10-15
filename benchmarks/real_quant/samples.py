"""Select and persist a small, identical token corpus for every configuration."""
import random

from benchmarks.real_quant.common import digest_json


def select_samples(token_ids, count=8, length=512, seed=20261013):
    if count < 1 or length < 2 or len(token_ids) // length < count:
        raise ValueError("not enough tokens for the requested non-overlapping samples")
    blocks = sorted(random.Random(seed).sample(range(len(token_ids) // length), count))
    samples = [{"index": i, "start": block * length, "length": length,
                "token_ids": [int(t) for t in token_ids[block * length:(block + 1) * length]]}
               for i, block in enumerate(blocks)]
    return {"selection": "seeded non-overlapping fixed-length blocks without padding",
            "seed": seed, "sample_count": count, "sample_length": length,
            "input_tokens": count * length, "scored_tokens": count * (length - 1),
            "token_sha256": digest_json([s["token_ids"] for s in samples]), "samples": samples}
