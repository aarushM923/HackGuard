from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from hackguard.activations.extract import capture_blocks, extract, forward_capture, load_layer, load_meta
from hackguard.agent.generation import Steering, steering_hook
from hackguard.trajectories.schema import HackLocation, Label, RunSummary, Trajectory


def _trajectories() -> tuple[list[Trajectory], dict[str, Label]]:
    summary = RunSummary(0, 0, False)
    trajs, labels = [], {}
    for p in range(4):
        for hack in (0, 1):
            content = "Looking at it. The test is wrong.\n<write_file path=\"test_solution.py\">\nx\n</write_file>"
            messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": f"task {p}"},
                        {"role": "assistant", "content": content}]
            tid = f"P/{p}::impossible::neutral::s{hack}"
            trajs.append(Trajectory(tid, f"P/{p}::impossible", f"P/{p}", "impossible", "humaneval", "neutral", hack,
                                    "m", messages, [], "submit", {}, summary, summary, summary))
            labels[tid] = (Label(tid, "hack", ["edit_tests"], HackLocation(2, content.index("<write_file"), 0, "edit_tests"))
                           if hack else Label(tid, "honest_fail"))
    return trajs, labels


def test_hooks_capture_block_outputs_and_are_removed(tiny_model: Any) -> None:
    model, tok, device = tiny_model
    ids = tok("hello world", add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        hidden = model.model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    # transformers adds hooks of its own the first time hidden states are requested (just
    # above), so count relative to what is there now.
    before = [len(block._forward_hooks) for block in model.model.layers]
    acts = forward_capture(model, device, ids, [4, 8])
    # Block L's output equals hidden_states[L] for inner blocks (the last one gets a final norm).
    assert torch.allclose(acts[4], hidden[4][0], atol=1e-5)
    assert acts[8].shape == (len(ids), 32)
    assert [len(block._forward_hooks) for block in model.model.layers] == before
    with capture_blocks(model, [2]):
        assert len(model.model.layers[1]._forward_hooks) == before[1] + 1


def test_extract_saves_windows_and_layers(tiny_model: Any, tmp_path: Path) -> None:
    model, tok, device = tiny_model
    trajs, labels = _trajectories()
    out = extract(trajs, labels, "tiny/fake-model", tmp_path, window=3, max_tokens=4096,
                  model=model, tokenizer=tok, device=device)
    meta = load_meta(out)
    assert meta["layers"] == [4, 8] and meta["hidden_size"] == 32 and meta["num_blocks"] == 8
    samples = pd.read_csv(out / "samples.csv")
    assert len(samples) == 8 * 3 and set(samples["label"]) == {0, 1}
    x = load_layer(out, 8)
    assert x.shape == (24, 32) and np.isfinite(x).all()

    # The second-model path re-uses the same (message, character) positions.
    out2 = extract(trajs, labels, "tiny/other", tmp_path, window=3, max_tokens=4096,
                   positions_from=out / "samples.csv", model=model, tokenizer=tok, device=device)
    samples2 = pd.read_csv(out2 / "samples.csv")
    assert list(samples2["token_index"]) == list(samples["token_index"])  # same tokenizer, same tokens
    assert np.allclose(load_layer(out2, 8), x, atol=1e-2)


def test_steering_hook_shifts_block_output(tiny_model: Any) -> None:
    model, tok, device = tiny_model
    ids = tok("abc", add_special_tokens=False)["input_ids"]
    direction = torch.zeros(32)
    direction[0] = 1.0
    before = len(model.model.layers[3]._forward_hooks)
    plain = forward_capture(model, device, ids, [4])[4]
    with steering_hook(model, Steering(layer=4, direction=direction, strength=3.0)):
        steered = forward_capture(model, device, ids, [4])[4]
    assert torch.allclose(steered[:, 0], plain[:, 0] - 3.0, atol=1e-5)
    assert torch.allclose(steered[:, 1:], plain[:, 1:], atol=1e-5)
    assert len(model.model.layers[3]._forward_hooks) == before
