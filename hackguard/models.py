"""Loading models with Hugging Face transformers, and reaching their transformer blocks.

A decoder language model is a stack of identical "blocks". Each block reads the
*residual stream* (one vector per token, e.g. 3,584 numbers for Qwen 7B), adds its own
contribution, and passes it on. Probing means reading that stream after some block;
steering means changing it. Both use PyTorch "forward hooks": small functions that
PyTorch calls with a block's output every time that block runs.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def pick_device() -> str:
    """CUDA on the EC2 GPU, MPS on an Apple Silicon Mac, otherwise the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    """Number format for the weights.

    bfloat16 on the A10G (16-bit, so a 7B model takes ~15 GB of the 24 GB). float16 on the
    Mac GPU. float32 on the CPU, where 16-bit math is slow or unsupported.
    """
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)


def hf_token() -> str | None:
    """The Hugging Face token from .env, needed only for gated models such as Llama."""
    return os.environ.get("HF_TOKEN") or None


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token())
    if tokenizer.pad_token is None:
        # Some models (Llama) have no padding token; reuse end-of-text for batching.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_id: str, device: str | None = None) -> tuple[Any, PreTrainedTokenizerBase, str]:
    """Load a model in evaluation mode. Returns (model, tokenizer, device)."""
    device = device or pick_device()
    # transformers has incomplete type hints, so we treat the model as Any.
    model: Any = AutoModelForCausalLM.from_pretrained(model_id, dtype=pick_dtype(device), token=hf_token())
    model.to(device)
    model.eval()  # turn off training-only behavior such as dropout
    return model, load_tokenizer(model_id), device


def decoder_blocks(model: Any) -> Any:
    """The list of transformer blocks. Qwen2 and Llama both keep them at model.model.layers."""
    return model.model.layers


def num_blocks(model: Any) -> int:
    return int(model.config.num_hidden_layers)


def block_output_tensor(output: Any) -> torch.Tensor:
    """A block's output is either a tensor or a tuple whose first item is the tensor,
    depending on the transformers version. This returns the tensor in both cases."""
    tensor: torch.Tensor = output[0] if isinstance(output, tuple) else output
    return tensor


def chat_text(tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    """Render a conversation with the model's own chat template, as plain text."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    assert isinstance(text, str)
    return text
