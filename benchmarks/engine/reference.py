"""Independent HF references: original FP16 or checkpoint-dequantized W4."""
import argparse
from pathlib import Path

import torch

from adapters.model_builder import build_hf_causal_lm, build_quantized_causal_lm
from transformers import AutoTokenizer


PROMPTS = [
    'The capital of France is',
    'Water freezes at a temperature of',
    'The sequence is 2, 4, 6, 8, and the next number is',
]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--quantized', action='store_true')
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (build_quantized_causal_lm(args.model, fake=True) if args.quantized
             else build_hf_causal_lm(args.model))
    samples = []
    for text in PROMPTS:
        ids = tokenizer.encode(text)
        x = torch.tensor([ids], device='cuda')
        cache, logits, tokens = None, [], []
        for _ in range(args.tokens):
            output = model(input_ids=x, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            last = output.logits[:, -1].float()
            token = int(last.argmax(-1).item())
            logits.append(last[0].half().cpu())
            tokens.append(token)
            x = torch.tensor([[token]], device='cuda')
        samples.append({'text': text, 'prompt_ids': ids, 'token_ids': tokens,
                        'logits': torch.stack(logits)})
    torch.save({'model': args.model, 'quantized': args.quantized, 'samples': samples}, args.out)


if __name__ == '__main__':
    main()
