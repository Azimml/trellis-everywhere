"""Perplexity evaluation harness.

Measures wikitext-2 perplexity with a sliding window — the standard metric for
comparing weight-quantization schemes. Same code path runs a 135M model on a
4 GB laptop and an 8B model on the DGX; only `--model` and `--max-samples`
change. This is the ruler the Phase-1 quality gate is measured against.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, device: str = "cuda", dtype=torch.bfloat16):
    # Default bf16: modern LLMs (Qwen3, Llama-3) are TRAINED in bf16. Loading in
    # fp16 narrows dynamic range and can overflow attention softmax to Inf/NaN
    # once weights are perturbed by quantization — which is exactly what broke
    # the 8B eval while leaving weights and per-layer output MSE healthy.
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()
    return model, tok


def get_wikitext2_test(tokenizer) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt").input_ids


@torch.no_grad()
def eval_ppl(model, input_ids: torch.Tensor, seqlen: int = 2048,
             device: str = "cuda", max_samples: int | None = None) -> float:
    """Sliding-window perplexity (non-overlapping windows of `seqlen`)."""
    n = input_ids.numel() // seqlen
    if max_samples is not None:
        n = min(n, max_samples)
    if n == 0:
        raise ValueError("not enough tokens for even one window; lower --seqlen")
    nlls = []
    for i in range(n):
        batch = input_ids[:, i * seqlen:(i + 1) * seqlen].to(device)
        out = model(batch, labels=batch)
        # HF loss = mean CE over (seqlen-1) shifted tokens; scale back to a sum
        nlls.append(out.loss.float() * (seqlen - 1))
    total_nll = torch.stack(nlls).sum()
    total_tokens = n * (seqlen - 1)
    return torch.exp(total_nll / total_tokens).item()
