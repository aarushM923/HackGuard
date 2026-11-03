"""Re-run saved trajectories through a model and save residual stream activations.

"Teacher forcing": we feed the model the complete conversation in one forward pass, with
no new generation. Because the model is causal (each token only sees earlier tokens), the
activation at token t is exactly what the model had computed when it was about to write
token t+1 during the original episode. So activations just before the hack show what the
model "was thinking" right before it cheated.

Output in runs/<name>/activations/<model>/:
    shard_00000.safetensors   tensor "acts" with shape [samples, layers, hidden_size], float16
    samples.csv               one row per sample: which trajectory/token it is, its label,
                              and which shard and row its activations are in
    meta.json                 model name, which layers, hidden size
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.numpy import load_file
from safetensors.torch import save_file

from hackguard.activations.sampling import USABLE_LABELS, Sample, hack_rank, sample_positions
from hackguard.activations.tokens import TokenizedTrajectory, tokenize_trajectory
from hackguard.config import model_slug, probe_layers
from hackguard.models import block_output_tensor, decoder_blocks, load_model, num_blocks
from hackguard.trajectories.schema import Label, Trajectory

SHARD_SIZE = 4096  # samples per file


@contextlib.contextmanager
def capture_blocks(model: Any, layers: list[int]) -> Iterator[dict[int, torch.Tensor]]:
    """Record the output of the chosen blocks during the next forward pass.

    Yields a dict {layer: tensor of shape [batch, tokens, hidden_size]} that fills in when
    the model runs. Hooks are removed afterwards so they never leak into later passes.
    """
    store: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        def hook(module: Any, inputs: Any, output: Any, layer: int = layer) -> None:
            store[layer] = block_output_tensor(output).detach()
        handles.append(decoder_blocks(model)[layer - 1].register_forward_hook(hook))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def forward_capture(model: Any, device: str, input_ids: list[int], layers: list[int]) -> dict[int, torch.Tensor]:
    """One forward pass; returns {layer: [tokens, hidden_size]} activations on the device."""
    ids = torch.tensor([input_ids], device=device)
    with capture_blocks(model, layers) as store:
        # model.model is the stack of blocks without the final "language model head". The head
        # would compute a score for every vocabulary word at every token (~2.4 GB for 8,000
        # tokens with Qwen's 152,000-word vocabulary) that we never use.
        # use_cache=False: one pass and nothing generated, so no need to keep a cache.
        model.model(input_ids=ids, use_cache=False)
    return {layer: store[layer][0] for layer in layers}


def reference_hack_ranks(items: list[tuple[Trajectory, Label, TokenizedTrajectory]]) -> dict[str, list[int]]:
    """How far into the episode (in agent tokens) each hack happened, by variant and overall."""
    ranks: dict[str, list[int]] = {"all": []}
    for traj, label, tt in items:
        if label.label != "hack":
            continue
        rank = hack_rank(traj, label, tt, tt.assistant_tokens())
        if rank:
            ranks["all"].append(rank)
            ranks.setdefault(traj.variant, []).append(rank)
    return ranks


class ShardWriter:
    """Collects activation rows and writes them to numbered shard files."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: list[np.ndarray] = []
        self.rows: list[dict[str, Any]] = []
        self.shard = 0
        self.total = 0

    def add(self, acts: np.ndarray, samples: list[dict[str, Any]]) -> None:
        for i, sample in enumerate(samples):
            self.buffer.append(acts[i])
            self.rows.append({**sample, "shard": self.shard, "row": len(self.buffer) - 1})
            if len(self.buffer) >= SHARD_SIZE:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        tensor = torch.from_numpy(np.stack(self.buffer)).to(torch.float16)
        save_file({"acts": tensor}, str(self.out_dir / f"shard_{self.shard:05d}.safetensors"))
        self.total += len(self.buffer)
        self.buffer = []
        self.shard += 1


def extract(
    trajectories: list[Trajectory],
    labels: dict[str, Label],
    model_id: str,
    out_root: Path,
    window: int,
    max_tokens: int,
    seed: int = 0,
    positions_from: Path | None = None,
    model: Any = None,
    tokenizer: Any = None,
    device: str | None = None,
) -> Path:
    """Save activations for every usable trajectory. Returns the output folder.

    positions_from: a samples.csv from another model's extraction. If given, we save this
    model's activations at the same (message, character) positions instead of choosing
    new ones. The transfer experiment needs exactly matched positions in both models.
    """
    if model is None:
        model, tokenizer, device = load_model(model_id)
    assert device is not None
    layers = probe_layers(num_blocks(model))
    out_dir = out_root / model_slug(model_id)
    writer = ShardWriter(out_dir)

    usable = [t for t in trajectories if labels[t.trajectory_id].label in USABLE_LABELS]
    print(f"Tokenizing {len(usable)} usable trajectories for {model_id}...")
    items = [(t, labels[t.trajectory_id], tokenize_trajectory(tokenizer, t.messages, max_tokens)) for t in usable]

    if positions_from is None:
        rng = random.Random(seed)
        refs = reference_hack_ranks(items)
        plan = {t.trajectory_id: [asdict(s) for s in sample_positions(t, lab, tt, window, refs, rng)] for t, lab, tt in items}
    else:
        plan = remap_positions(pd.read_csv(positions_from), {t.trajectory_id: tt for t, _, tt in items})

    start = time.time()
    for n, (traj, _, tt) in enumerate(items):
        samples = plan.get(traj.trajectory_id, [])
        if not samples:
            continue
        acts = forward_capture(model, device, tt.input_ids, layers)
        index = torch.tensor([s["token_index"] for s in samples], device=device)
        # Stack to shape [samples, layers, hidden_size] and move to the CPU as float16.
        stacked = torch.stack([acts[layer].index_select(0, index) for layer in layers], dim=1)
        writer.add(stacked.float().cpu().numpy().astype(np.float16), samples)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(items)} trajectories ({(n + 1) / (time.time() - start):.1f} per second)")
    writer.flush()

    pd.DataFrame(writer.rows).to_csv(out_dir / "samples.csv", index=False)
    meta = {"model": model_id, "layers": layers, "num_blocks": num_blocks(model),
            "hidden_size": int(model.config.hidden_size),
            "samples": writer.total, "window": window}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {writer.total} samples x {len(layers)} layers to {out_dir}")
    return out_dir


def remap_positions(samples: pd.DataFrame, tokenized: dict[str, TokenizedTrajectory]) -> dict[str, list[dict[str, Any]]]:
    """Find, in this model's tokens, the positions another model's samples point to."""
    plan: dict[str, list[dict[str, Any]]] = {}
    for row in samples.to_dict("records"):
        tt = tokenized.get(row["trajectory_id"])
        if tt is None or row["message_index"] < 0:
            continue
        token = tt.last_token_ending_by(int(row["message_index"]), int(row["char_end"]))
        sample = {k: row[k] for k in Sample.__dataclass_fields__}
        sample["token_index"] = token
        sample["hack_types"] = "" if pd.isna(sample["hack_types"]) else sample["hack_types"]
        plan.setdefault(row["trajectory_id"], []).append(sample)
    return plan


def load_meta(act_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((act_dir / "meta.json").read_text(encoding="utf-8"))
    return meta


def load_samples(act_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(act_dir / "samples.csv")
    df["hack_types"] = df["hack_types"].fillna("")
    return df


def load_layer(act_dir: Path, layer: int) -> np.ndarray:
    """All saved activations for one layer, as float32 [samples, hidden_size], in samples.csv order."""
    meta = load_meta(act_dir)
    li = meta["layers"].index(layer)
    samples = load_samples(act_dir)
    out = np.zeros((len(samples), meta["hidden_size"]), dtype=np.float32)
    for shard, group in samples.groupby("shard"):
        acts = load_file(str(act_dir / f"shard_{int(shard):05d}.safetensors"))["acts"]
        out[group.index.to_numpy()] = acts[group["row"].to_numpy(), li, :].astype(np.float32)
    return out
