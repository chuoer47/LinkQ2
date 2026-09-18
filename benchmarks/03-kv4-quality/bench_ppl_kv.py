"""How much does the KV4 scheme cost in PPL? Measured on HF directly.

The runtime is not needed to answer this: the scheme is a property of how K
and V are quantized, not of where they are stored.

  q' = q * lambda,  k' = k / lambda        (SmoothAttention; the pairing
                                            constraint makes it commute with
                                            RoPE)
  k' -> int4 with the calibrated STATIC per-channel scale
  v  -> int4 per token, grouped along head_dim

The interception point matters: K and Q must be quantized AFTER RoPE, and
V after its projection (V has no norm or RoPE in Qwen3). K is caught by
wrapping `apply_rotary_pos_emb`, which is exactly the post-RoPE boundary; a
hook on k_proj would quantize pre-norm, pre-RoPE values and measure something
else entirely.

Standard sliding-window PPL (docs/03: window 1024, stride 512, WikiText-2
raw). Modes: fp16 | kv4 | k4only | v4only.
"""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

WINDOW = 1024
STRIDE = 512
V_GROUP = 64
N_DOCS = int(os.environ.get("N_DOCS", "10"))


def load_texts():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return [t for t in ds["text"] if len(t.strip()) > 200][:N_DOCS]


def q_k_grouped(k, scale):
    """k [B,H,T,D] fp16, scale [H,D] fp32 -> int4 round-trip."""
    kf = k.float()
    q = torch.clamp(torch.floor(kf / scale[None, :, None, :] + 0.5), -8, 7)
    return (q * scale[None, :, None, :])


def q_v_grouped(v, group=V_GROUP):
    """v [B,H,T,D] -> int4 per token, grouped along D."""
    D = v.shape[-1]
    vf = v.float().reshape(*v.shape[:-1], D // group, group)
    s = (vf.abs().amax(-1, keepdim=True) / 7).clamp_min(1e-4)
    q = torch.clamp(torch.floor(vf / s + 0.5), -8, 7)
    return (q * s).reshape(*v.shape[:-1], D)


class Scheme:
    """Per-layer lambda/kscale, indexed by call order (layers run in order)."""

    def __init__(self, lam, kscale, mode):
        self.lam = lam            # [L, H_kv, D] fp16
        self.kscale = kscale      # [L, H_kv, D] fp32
        self.mode = mode
        self.layer = 0
        self.n_layers = lam.shape[0]

    def next_layer(self):
        li = self.layer % self.n_layers
        self.layer += 1
        return li


@torch.inference_mode()
def ppl(model, tok, texts):
    nll, n = 0.0, 0
    for text in texts:
        ids = tok.encode(text)
        if len(ids) < 2:
            continue
        prev_end = 0
        for begin in range(0, len(ids) - 1, STRIDE):
            end = min(begin + WINDOW, len(ids))
            x = torch.tensor([ids[begin:end]], device="cuda")
            logits = model(input_ids=x, use_cache=False).logits
            count = min(end - prev_end, x.shape[1] - 1)
            lp = torch.log_softmax(logits[0, -count - 1:-1, :].float(), dim=-1)
            tgt = x[0, -count:]
            nll -= lp.gather(-1, tgt[:, None]).sum().item()
            n += count
            prev_end = end
    return math.exp(nll / max(n, 1)), n


def install(model, scheme):
    """Patch RoPE (post-RoPE q/k) and v_proj (V)."""
    import transformers.models.qwen3.modeling_qwen3 as M

    orig_rope = M.apply_rotary_pos_emb
    v_hooks = []
    n_rep = model.config.num_attention_heads // model.config.num_key_value_heads

    def rope_with_kv(q, k, cos, sin, unsqueeze_dim=1):
        q, k = orig_rope(q, k, cos, sin, unsqueeze_dim)
        if scheme.mode == "fp16":
            return q, k
        li = scheme.next_layer()
        L = scheme.lam[li].float()[None, :, None, :]      # [1,H_kv,1,D]
        S = scheme.kscale[li]                             # [H_kv,D] fp32
        if scheme.mode == "v4only":
            # V-only ablation: no smoothing either, or q and k would be
            # scaled inconsistently and the logits would be destroyed
            return q, k
        # SmoothAttention is a pair: q *= lambda, k /= lambda. Applying only
        # one half changes every logit, so it goes on exactly when K is
        # quantized.
        Lq = scheme.lam[li].float().repeat_interleave(n_rep, 0)[None, :, None, :]
        q = q.float() * Lq
        k = (k.float() / L)
        if scheme.mode in ("kv4", "k4only"):
            k = q_k_grouped(k, S)
        return q.to(torch.float16), k.to(torch.float16)

    M.apply_rotary_pos_emb = rope_with_kv

    if scheme.mode in ("kv4", "v4only"):
        def v_hook(mod, inp, out):
            B, T, _ = out.shape
            hd = mod.out_features // model.config.num_key_value_heads
            v = out.view(B, T, -1, hd)
            return q_v_grouped(v).reshape(out.shape).to(out.dtype)

        for layer in model.model.layers:
            v_hooks.append(layer.self_attn.v_proj.register_forward_hook(v_hook))

    return orig_rope, v_hooks


if __name__ == "__main__":
    from qslab.models.loader import load_reference_model
    from adapters.tokenizer import QwenTokenizerAdapter

    MODEL = os.environ["MODEL"]
    MODE = os.environ.get("MODE", "fp16")
    tok = QwenTokenizerAdapter(MODEL)
    texts = load_texts()
    model = load_reference_model(MODEL)

    orig_rope, v_hooks = None, []
    if MODE != "fp16":
        ck = torch.load(os.environ["CALIB"], weights_only=False, map_location="cuda")
        scheme = Scheme(ck["lambda"].cuda(), ck["kscale"].float().cuda(), MODE)
        orig_rope, v_hooks = install(model, scheme)

    val, n = ppl(model, tok, texts)
    if orig_rope is not None:
        import transformers.models.qwen3.modeling_qwen3 as M
        M.apply_rotary_pos_emb = orig_rope
    for h in v_hooks:
        h.remove()
    print(f"RESULT {Path(MODEL).name} {MODE}: PPL {val:.4f}  ({n} tokens)")
