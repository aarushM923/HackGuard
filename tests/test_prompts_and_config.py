from pathlib import Path

import pytest

from hackguard.agent.prompts import PROMPT_SUFFIXES, first_user_message, system_prompt
from hackguard.config import PRESETS, RunConfig, model_slug, probe_layers
from hackguard.tasks.schema import Problem
from hackguard.tasks.variants import make_honest


def test_system_prompts_differ_only_by_suffix() -> None:
    neutral = system_prompt("neutral", 5)
    assert "at most 5 actions" in neutral
    assert system_prompt("pressure", 5).startswith(neutral) and "shut down" in system_prompt("pressure", 5)
    assert "Do not modify the test file" in system_prompt("no_modify", 5)
    assert set(PROMPT_SUFFIXES) == {"neutral", "pressure", "no_modify"}
    with pytest.raises(ValueError):
        system_prompt("unknown", 5)


def test_first_message_shows_both_files(humaneval_problem: Problem) -> None:
    msg = first_user_message(make_honest(humaneval_problem))
    assert "--- solution.py ---" in msg and "--- test_solution.py ---" in msg
    assert "raise NotImplementedError" in msg


def test_probe_layers_every_fourth() -> None:
    assert probe_layers(28) == [4, 8, 12, 16, 20, 24, 28]
    assert probe_layers(24) == [4, 8, 12, 16, 20, 24]


def test_model_slug() -> None:
    assert model_slug("Qwen/Qwen2.5-Coder-7B-Instruct") == "Qwen2.5-Coder-7B-Instruct"


def test_config_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # run folders are relative to the current folder
    cfg = PRESETS["laptop"]("x")
    cfg.save()
    assert RunConfig.load(cfg.run_dir) == cfg
    assert PRESETS["gpu"]("g").backend == "vllm" and PRESETS["gpu"]("g").runner == "docker"
    assert PRESETS["pilot"]("p").num_problems == 34
