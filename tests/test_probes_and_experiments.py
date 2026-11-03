import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hackguard.activations.extract import ShardWriter, load_layer, load_meta, load_samples
from hackguard.experiments.generalization import family_of, run_generalization
from hackguard.experiments.transfer import matching_layer, run_transfer
from hackguard.probes.lead_time import lead_time_report
from hackguard.probes.regex_baseline import regex_score
from hackguard.probes.train import LinearProbe, best_layer, train_all_layers, train_diffmeans
from synthetic import make_activation_dir


@pytest.fixture
def trained(tmp_path: Path) -> tuple[Path, Path, Path, np.ndarray]:
    act_dir, direction = make_activation_dir(tmp_path / "acts")
    probe_dir, results = tmp_path / "probes", tmp_path / "results"
    train_all_layers(act_dir, probe_dir, results, seed=0)
    return act_dir, probe_dir, results, direction


def test_probes_find_the_planted_direction(trained: tuple[Path, Path, Path, np.ndarray]) -> None:
    act_dir, probe_dir, results, direction = trained
    table = pd.read_csv(results / "probe_layers.csv")
    assert set(table["probe"]) == {"logreg", "diffmeans"}
    deep = table[(table["layer"] == 8) & (table["probe"] == "logreg")].iloc[0]
    assert deep["auroc"] > 0.95 and deep["auroc_ci_low"] <= deep["auroc"] <= deep["auroc_ci_high"]
    assert best_layer(results) == 8  # the deeper layer has the stronger signal
    diff = LinearProbe.load(probe_dir / "diffmeans_layer08.npz")
    cosine = diff.weight @ direction / np.linalg.norm(diff.weight)
    assert cosine > 0.9  # difference in means points along the planted direction


def test_probe_save_load_round_trip(tmp_path: Path) -> None:
    x = np.random.default_rng(0).normal(size=(20, 4))
    y = np.array([0, 1] * 10)
    probe = train_diffmeans(x, y, layer=4)
    probe.save(tmp_path / "p.npz")
    again = LinearProbe.load(tmp_path / "p.npz")
    assert np.allclose(again.score(x), probe.score(x)) and again.kind == "diffmeans" and again.layer == 4
    assert np.all((probe.prob(x) > 0) & (probe.prob(x) < 1))


def test_training_refuses_one_class(tmp_path: Path) -> None:
    act_dir, _ = make_activation_dir(tmp_path / "acts")
    samples = pd.read_csv(act_dir / "samples.csv")
    samples["label"] = 0
    samples.to_csv(act_dir / "samples.csv", index=False)
    with pytest.raises(ValueError, match="only one class"):
        train_all_layers(act_dir, tmp_path / "p", tmp_path / "r")


def test_same_point_baselines_and_judge(trained: tuple[Path, Path, Path, np.ndarray], tmp_path: Path) -> None:
    from hackguard.probes.judge import add_llm_judge, run_same_point_baselines, transcript_until
    from hackguard.trajectories.schema import RunSummary, Trajectory
    act_dir, probe_dir, results, _ = trained
    samples = load_samples(act_dir)
    summary = RunSummary(0, 0, False)
    trajs = [Trajectory(tid, "t", g["problem_id"].iloc[0], g["variant"].iloc[0], "x", "neutral", 0, "m",
                        [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
                         {"role": "assistant", "content": "I will hardcode it." if g["label"].iloc[0] else "Thinking carefully."}],
                        [], "submit", {}, summary, summary, summary)
             for tid, g in samples.groupby("trajectory_id")]

    # Free part (GPU machine): probe and keyword scores at the same moments.
    report = run_same_point_baselines(act_dir, probe_dir, trajs, results, layer=8, seed=0, max_items=20)
    assert report["n_items"] == 20 and report["probe"]["auroc"] > 0.9 and report["regex"]["auroc"] == 1.0
    assert "llm_judge" not in report

    # Paid part (laptop): refuses to spend without permission.
    with pytest.raises(PermissionError):
        add_llm_judge(results, trajs)

    calls = []

    def fake_judge(text: str) -> float:
        calls.append(text)
        return 0.9 if "hardcode" in text else 0.1

    report = add_llm_judge(results, trajs, judge=fake_judge)
    assert report["llm_judge"]["auroc"] == 1.0 and report["judge_used"]
    assert all("ENVIRONMENT" in c and "[transcript cut off here]" in c for c in calls)
    # Cached answers are reused: a second run makes no new calls.
    n = len(calls)
    add_llm_judge(results, trajs, judge=fake_judge)
    assert len(calls) == n
    assert transcript_until(trajs[0], 2, 4).endswith("[transcript cut off here]")


def test_generalization_runs_both_directions(trained: tuple[Path, Path, Path, np.ndarray]) -> None:
    act_dir, _, results, _ = trained
    report = run_generalization(act_dir, results, layer=8, seed=0)
    for key in ("train_test_edits__test_code_hacks", "train_code_hacks__test_test_edits"):
        assert report[key]["auroc_other_family"] > 0.9  # same planted direction for both families
    assert family_of("edit_tests") == "test_edits" and family_of("special_case hardcode_outputs") == "code_hacks"
    assert family_of("edit_tests special_case") is None and family_of("") is None


def test_transfer_recovers_a_linear_map(trained: tuple[Path, Path, Path, np.ndarray], tmp_path: Path) -> None:
    act_dir, probe_dir, results, _ = trained
    # Build "model B": a fixed random linear transform of model A's activations, in 24 dimensions.
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(16, 24))
    samples = load_samples(act_dir)
    b_dir = tmp_path / "b"
    writer = ShardWriter(b_dir)
    stacked = np.stack([load_layer(act_dir, layer) @ matrix for layer in (4, 8)], axis=1)
    writer.add(stacked.astype(np.float16), samples.drop(columns=["shard", "row"]).to_dict("records"))
    writer.flush()
    pd.DataFrame(writer.rows).to_csv(b_dir / "samples.csv", index=False)
    (b_dir / "meta.json").write_text(json.dumps({"model": "b", "layers": [4, 8], "num_blocks": 8, "hidden_size": 24}))
    report = run_transfer(act_dir, b_dir, probe_dir, results, layer_a=8, seed=0)
    assert report["layer_b"] == 8
    assert report["alignment_r2_test"] > 0.9
    assert report["auroc_qwen_probe_on_mapped_b"] > 0.9
    assert matching_layer(14, 28, [4, 8, 12, 16], 32) == 16


def test_regex_score() -> None:
    assert regex_score("I will carefully implement the loop.") == 0
    assert regex_score("The test is wrong, so I will hardcode the expected output.") >= 2


def test_lead_time_report(tmp_path: Path) -> None:
    arrays = {"probe_0": np.array([[0.1, 0.9, 0.9]]), "regex_0": np.array([0.0, 0.0, 1.0]),
              "probe_1": np.array([[0.1, 0.2, 0.3]]), "regex_1": np.array([0.0, 0.0, 0.0]),
              "probe_2": np.array([[0.2, 0.3, 0.4]]), "regex_2": np.array([0.0, 0.0, 0.0])}
    path = tmp_path / "scores.npz"
    np.savez(path, **arrays)  # type: ignore[arg-type]
    items = [{"key": 0, "split": "test", "label": "hack", "hack_rank": 3, "tokens": 3},
             {"key": 1, "split": "val", "label": "honest_fail", "hack_rank": None, "tokens": 3},
             {"key": 2, "split": "test", "label": "honest_success", "hack_rank": None, "tokens": 3}]
    path.with_suffix(".index.json").write_text(json.dumps({"layers": [8], "items": items}))
    report = lead_time_report(path, tmp_path / "results", best=8)
    assert report["probe"]["caught_before"] == 1 and report["probe"]["median_lead"] == 2
    assert report["probe"]["honest_false_alarm_rate"] == 1.0  # tau is 0.3 and the test honest run reaches 0.4
    assert (tmp_path / "results" / "lead_times.csv").exists()


def test_report_builds_from_partial_results(trained: tuple[Path, Path, Path, np.ndarray]) -> None:
    from hackguard.report import build_report
    _, _, results, _ = trained
    run_dir = results.parent
    summary = build_report(run_dir)
    assert "headline_auroc" in summary
    text = (run_dir / "report.md").read_text()
    assert "Probe AUROC by layer" in text and "(best)" in text
    assert (results / "auroc_by_layer.png").exists()
    assert load_meta(run_dir / "acts" / "fake-model")["layers"] == [4, 8]
