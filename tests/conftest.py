"""Shared test fixtures: small fake dataset rows in the same shape as HumanEval+ and MBPP+.

pytest automatically loads this file, and any test can ask for a fixture by naming it as a
function argument (e.g. `def test_x(humaneval_row): ...`).
"""

from __future__ import annotations

from typing import Any

import pytest

from hackguard.tasks.loaders import problem_from_humaneval, problem_from_mbpp
from hackguard.tasks.schema import Problem

# The comparison helper, shortened from the real EvalPlus one.
ASSERTION_HELPER = '''
import numpy as np

def assertion(out, exp, atol):
    if atol != 0 and isinstance(exp, float):
        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        assert out == exp
'''

# Twelve cases for add(a, b). The first three play the role of the original benchmark tests.
ADD_INPUTS = [[1, 2], [0, 0], [-1, 1], [5, 5], [100, 200], [-50, -50], [7, 8], [3, 4],
              [10, -3], [999, 1], [2, 2], [123456, 654321]]
ADD_RESULTS = [a + b for a, b in ADD_INPUTS]


def humaneval_test_code(inputs: list[Any], results: list[Any], atol: float = 0) -> str:
    """Test code laid out like HumanEval+: lists and loop inside def check(candidate)."""
    return f"""{ASSERTION_HELPER}

def check(candidate):
    inputs = {inputs!r}
    results = {results!r}
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assertion(candidate(*inp), exp, {atol!r})
"""


def mbpp_test_code(func: str, inputs: list[Any], results: list[Any], atol: float = 0) -> str:
    """Test code laid out like MBPP+: lists and loop at the top level."""
    return f"""{ASSERTION_HELPER}

inputs = {inputs!r}
results = {results!r}
for i, (inp, exp) in enumerate(zip(inputs, results)):
    assertion({func}(*inp), exp, {atol!r})
"""


@pytest.fixture
def humaneval_row() -> dict[str, Any]:
    return {
        "task_id": "HumanEval/900",
        "entry_point": "add",
        "prompt": 'def add(a: int, b: int) -> int:\n    """Add two numbers.\n    >>> add(1, 2)\n    3\n    """\n',
        "canonical_solution": "    return a + b\n",
        "test": humaneval_test_code(ADD_INPUTS, ADD_RESULTS),
    }


@pytest.fixture
def mbpp_row() -> dict[str, Any]:
    inputs = [[i, i + 1] for i in range(12)]
    return {
        "task_id": 901,
        "prompt": "Write a function to multiply two numbers.",
        "code": "import math\n\ndef mul(x, y):\n  return x * y\n",
        "test_imports": ["import math"],
        "test": mbpp_test_code("mul", inputs, [a * b for a, b in inputs]),
    }


@pytest.fixture
def humaneval_problem(humaneval_row: dict[str, Any]) -> Problem:
    return problem_from_humaneval(humaneval_row)


# ---------------------------------------------------------------- Step 2+ helpers


@pytest.fixture(scope="session")
def qwen_tokenizer() -> Any:
    """The real tokenizer of the tiny Qwen model (downloaded once, then cached)."""
    from hackguard.config import TINY_MODEL
    from hackguard.models import load_tokenizer
    return load_tokenizer(TINY_MODEL)


class ScriptedGenerator:
    """A fake model: replies come from a function of (prompt, turn), so tests are exact.

    script(prompt_text, turn) -> reply. The turn number counts calls per prompt batch.
    """

    def __init__(self, tokenizer: Any, script: Any) -> None:
        self.tokenizer = tokenizer
        self.script = script
        self.turn = 0

    def generate(self, prompts: list[str], seeds: list[int]) -> list[str]:
        replies = [self.script(p, self.turn) for p in prompts]
        self.turn += 1
        return replies


@pytest.fixture(scope="session")
def tiny_model(qwen_tokenizer: Any) -> tuple[Any, Any, str]:
    """A very small randomly initialized Qwen2-shaped model (8 blocks, 32 numbers per token).

    Its outputs are meaningless, but it has the same structure as the real models, so it
    tests hooks, extraction and steering in a second without downloading anything.
    """
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(qwen_tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=8192)
    model: Any = Qwen2ForCausalLM(config)  # type: ignore[no-untyped-call]
    model.eval()
    return model, qwen_tokenizer, "cpu"


@pytest.fixture
def mbpp_problem(mbpp_row: dict[str, Any]) -> Problem:
    return problem_from_mbpp(mbpp_row)
