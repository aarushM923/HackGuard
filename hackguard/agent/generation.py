"""Text generation backends. Each takes a batch of prompts and returns one reply per prompt.

- HFGenerator uses plain Hugging Face transformers. It runs anywhere (including the Mac)
  and supports steering hooks, but is slow for thousands of episodes.
- VLLMGenerator uses vLLM, which batches requests very efficiently on an NVIDIA GPU.
  It does not run on a Mac, so the laptop test uses HFGenerator.

Both receive prompts that are already rendered with the model's chat template, so the
exact text the model sees is the same in both backends and in activation extraction later.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from hackguard.agent.protocol import STOP_STRINGS
from hackguard.models import block_output_tensor, decoder_blocks, load_model, load_tokenizer

TOP_P = 0.95  # sample only from the most likely tokens that cover 95% of the probability


class Generator(Protocol):
    tokenizer: Any

    def generate(self, prompts: list[str], seeds: list[int]) -> list[str]: ...


def cut_at_stop(text: str) -> str:
    """Keep text up to and including the first stop string (the end of the first action)."""
    ends = [text.find(s) + len(s) for s in STOP_STRINGS if s in text]
    return text[: min(ends)] if ends else text


@dataclass
class Steering:
    """Push the residual stream away from the "cheating direction" while generating.

    After block `layer` runs, we subtract strength * direction from every token's vector.
    direction is a unit vector (length 1), so `strength` is in the same units as the
    activations themselves.
    """

    layer: int  # 1-based block number, as in config.probe_layers
    direction: torch.Tensor  # shape [hidden_size], unit length
    strength: float


@contextlib.contextmanager
def steering_hook(model: Any, steering: Steering | None) -> Iterator[None]:
    """Attach the steering hook for the duration of a `with` block, then remove it."""
    if steering is None or steering.strength == 0:
        yield
        return
    block = decoder_blocks(model)[steering.layer - 1]  # block L is index L-1 in the list
    shift = (steering.strength * steering.direction).to(next(model.parameters()).device)

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        hidden = block_output_tensor(output)
        steered = hidden - shift.to(hidden.dtype)
        # Returning a value from a forward hook replaces the block's output.
        return (steered, *output[1:]) if isinstance(output, tuple) else steered

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class HFGenerator:
    """Batched sampling with transformers' model.generate."""

    model: Any
    tokenizer: Any
    device: str

    def __init__(self, model_id: str, max_new_tokens: int, temperature: float,
                 batch_size: int, steering: Steering | None = None, device: str | None = None,
                 loaded: tuple[Any, Any, str] | None = None) -> None:
        # loaded: an already loaded (model, tokenizer, device), to avoid loading twice.
        self.model, self.tokenizer, self.device = loaded or load_model(model_id, device)
        # Pad on the left so every prompt in a batch ends at the same position and new
        # tokens are appended right after each prompt's real end.
        self.tokenizer.padding_side = "left"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.steering = steering

    @torch.no_grad()  # no gradients needed: we only run the model forward
    def generate(self, prompts: list[str], seeds: list[int]) -> list[str]:
        replies: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            # transformers samples a whole batch from one random generator, so we seed it
            # from the batch's first episode; results are reproducible for a fixed batching.
            torch.manual_seed(seeds[i])
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with steering_hook(self.model, self.steering):
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=TOP_P,
                    stop_strings=STOP_STRINGS,
                    tokenizer=self.tokenizer,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = out[:, inputs["input_ids"].shape[1] :]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            # In a batch, generation continues until every sequence stops, so trim each one.
            replies.extend(cut_at_stop(t) for t in texts)
        return replies


class VLLMGenerator:
    """Fast batched sampling with vLLM (NVIDIA GPU only)."""

    def __init__(self, model_id: str, max_new_tokens: int, temperature: float, max_context_tokens: int) -> None:
        from vllm import LLM  # imported here because vLLM is only installed on the GPU machine

        self.tokenizer = load_tokenizer(model_id)
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            max_model_len=max_context_tokens,
            gpu_memory_utilization=0.90,  # leave a little GPU memory free for safety
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompts: list[str], seeds: list[int]) -> list[str]:
        from vllm import SamplingParams

        params = [
            SamplingParams(
                temperature=self.temperature,
                top_p=TOP_P,
                max_tokens=self.max_new_tokens,
                seed=seed,  # vLLM supports a separate seed per request
                stop=STOP_STRINGS,
                include_stop_str_in_output=True,
            )
            for seed in seeds
        ]
        outputs = self.llm.generate(prompts, params)
        return [cut_at_stop(o.outputs[0].text) for o in outputs]


def make_generator(backend: str, model_id: str, max_new_tokens: int, temperature: float,
                   batch_size: int, max_context_tokens: int, steering: Steering | None = None) -> Generator:
    if backend == "vllm":
        if steering is not None:
            raise ValueError("steering needs the 'hf' backend (vLLM does not expose forward hooks)")
        return VLLMGenerator(model_id, max_new_tokens, temperature, max_context_tokens)
    if backend == "hf":
        return HFGenerator(model_id, max_new_tokens, temperature, batch_size, steering)
    raise ValueError(f"unknown backend {backend!r}")
