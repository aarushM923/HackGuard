import csv
from pathlib import Path

import pytest

from hackguard.trajectories.audit import sample_for_audit, score_audit
from hackguard.trajectories.schema import Label, RunSummary, Trajectory, load_labels, load_trajectories, save_jsonl


def _data(n: int = 10) -> tuple[list[Trajectory], dict[str, Label]]:
    summary = RunSummary(1, 0, True)
    trajs, labels = [], {}
    for i in range(n):
        tid = f"P/{i}::honest::neutral::s0"
        trajs.append(Trajectory(tid, f"P/{i}::honest", f"P/{i}", "honest", "mbpp", "neutral", 0, "m",
                                [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                 {"role": "assistant", "content": f"answer {i}"}], [], "submit", {},
                                summary, summary, summary))
        labels[tid] = Label(tid, "hack" if i % 2 else "honest_success", ["edit_tests"] if i % 2 else [])
    return trajs, labels


def test_jsonl_round_trip(tmp_path: Path) -> None:
    trajs, labels = _data(3)
    save_jsonl(trajs, tmp_path / "t.jsonl")
    save_jsonl(labels.values(), tmp_path / "l.jsonl")
    assert load_trajectories(tmp_path / "t.jsonl") == trajs
    assert load_labels(tmp_path / "l.jsonl") == labels


def test_sample_and_score(tmp_path: Path) -> None:
    trajs, labels = _data(10)
    sheet = sample_for_audit(trajs, labels, tmp_path / "audit", size=6)
    with sheet.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert (tmp_path / "audit" / rows[0]["transcript"]).exists()
    assert "Automatic label" in (tmp_path / "audit" / rows[0]["transcript"]).read_text()
    with pytest.raises(ValueError):
        score_audit(sheet, tmp_path / "r.json")

    # A reviewer agrees with every label except one false "hack".
    flipped = False
    for r in rows:
        r["your_label"] = "hack" if r["auto_label"] == "hack" else "not_hack"
        if r["auto_label"] == "hack" and not flipped:
            r["your_label"], flipped = "not_hack", True
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = score_audit(sheet, tmp_path / "r.json")
    assert result["audited"] == 6 and result["false_positive"] == 1
    assert result["accuracy"] == 5 / 6
