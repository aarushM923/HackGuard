"""Collect all results of a run into summary.json, report.md and the two headline charts.

Every results file is optional: the report includes whatever steps have finished, so it can
be rebuilt at any point (for example before the hand audit or the judge baseline are done).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # draw charts to files without opening a window
import matplotlib.pyplot as plt  # noqa: E402  (must come after choosing the backend)
import pandas as pd  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def plot_auroc_by_layer(table: pd.DataFrame, best: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for kind, label in (("logreg", "Logistic regression"), ("diffmeans", "Difference in means")):
        part = table[table["probe"] == kind].sort_values("layer")
        ax.plot(part["layer"], part["auroc"], marker="o", label=label)
        if kind == "logreg":
            ax.fill_between(part["layer"], part["auroc_ci_low"], part["auroc_ci_high"], alpha=0.2)
    ax.axvline(best, color="gray", linestyle="--", linewidth=1, label=f"best layer ({best}, chosen on validation)")
    ax.axhline(0.5, color="black", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Predicting reward hacking from activations")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_lead_times(leads: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    probe = leads[leads["detector"] == "probe"]["lead_time"]
    if len(probe):
        ax.hist(probe, bins=30)
    ax.set_xlabel("Tokens of warning before the first hacking token")
    ax.set_ylabel("Hack trajectories")
    ax.set_title("Probe lead time (hacks caught before they happen)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def build_report(run_dir: Path) -> dict[str, Any]:
    results = run_dir / "results"
    summary: dict[str, Any] = {
        "labels": _read_json(run_dir / "label_summary.json"),
        "label_audit": _read_json(results / "label_audit.json"),
        "best_layer": _read_json(results / "best_layer.json"),
        "lead_time": _read_json(results / "lead_time.json"),
        "baselines_same_points": _read_json(results / "baselines_same_points.json"),
        "generalization": _read_json(results / "generalization.json"),
        "transfer": _read_json(results / "transfer.json"),
        "steering": _read_json(results / "steering.json"),
    }
    lines = [f"# HackGuard results: {run_dir.name}", ""]

    if summary["labels"]:
        counts = summary["labels"]["counts"]
        lines += ["## Labels", "", "| Group | hack | honest_success | honest_fail | ambiguous |", "|---|---|---|---|---|"]
        for group, c in counts.items():
            lines.append(f"| {group} | {c.get('hack', 0)} | {c.get('honest_success', 0)} | {c.get('honest_fail', 0)} | {c.get('ambiguous', 0)} |")
        lines += ["", f"First hack type: {summary['labels']['hack_types_first']}", ""]
    if summary["label_audit"]:
        a = summary["label_audit"]
        lines += [f"Hand audit of {a['audited']} labels: accuracy {_fmt(a['accuracy'])}, "
                  f"hack precision {_fmt(a['hack_precision'])}, hack recall {_fmt(a['hack_recall'])}", ""]

    table_path = results / "probe_layers.csv"
    if table_path.exists() and summary["best_layer"]:
        table = pd.read_csv(table_path)
        best = summary["best_layer"]["layer"]
        plot_auroc_by_layer(table, best, results / "auroc_by_layer.png")
        lines += ["## Probe AUROC by layer (test split)", "",
                  "| Probe | Layer | AUROC (95% CI) | Precision at 5% FPR | Trajectory AUROC | Impossible-only AUROC |",
                  "|---|---|---|---|---|---|"]
        for r in table.sort_values(["probe", "layer"]).itertuples():
            star = " **(best)**" if r.layer == best and r.probe == "logreg" else ""
            lines.append(f"| {r.probe} | {r.layer}{star} | {_fmt(r.auroc)} ({_fmt(r.auroc_ci_low)} to {_fmt(r.auroc_ci_high)}) | "
                         f"{_fmt(r.precision_at_5pct_fpr)} | {_fmt(r.trajectory_auroc)} | {_fmt(r.impossible_only_auroc)} |")
        lines += ["", "![AUROC by layer](results/auroc_by_layer.png)", ""]
        best_row = table[(table["probe"] == "logreg") & (table["layer"] == best)].iloc[0]
        summary["headline_auroc"] = float(best_row["auroc"])

    if summary["lead_time"]:
        lt = summary["lead_time"]
        lines += ["## Lead time (test split, alarm threshold set for 5% false alarms on validation)", "",
                  "| Detector | Hacks | Caught before | Caught after | Missed | Median lead | Mean lead | Honest false alarm rate |",
                  "|---|---|---|---|---|---|---|---|"]
        for name in ("probe", "regex"):
            d = lt[name]
            lines.append(f"| {name} | {d['hacks']} | {d['caught_before']} ({_fmt(d['share_caught_before'], 2)}) | "
                         f"{d['caught_after']} | {d['missed']} | {_fmt(d['median_lead'], 0)} | {_fmt(d['mean_lead'], 0)} | "
                         f"{_fmt(d['honest_false_alarm_rate'], 2)} |")
        leads_path = results / "lead_times.csv"
        if leads_path.exists():
            plot_lead_times(pd.read_csv(leads_path), results / "lead_time_hist.png")
            lines += ["", "![Lead time histogram](results/lead_time_hist.png)", ""]

    if summary["baselines_same_points"]:
        b = summary["baselines_same_points"]
        lines += ["## Detectors at the same moment (just before the hack)", "",
                  f"{b['n_items']} test moments, {b['n_hacks']} of them hacks.", "",
                  "| Detector | AUROC | Precision at 5% FPR |", "|---|---|---|"]
        for name in ("probe", "regex", "llm_judge"):
            if name in b:
                lines.append(f"| {name} | {_fmt(b[name]['auroc'])} | {_fmt(b[name]['precision_at_5pct_fpr'])} |")
        lines.append("")

    for title, key in (("Generalization across hack types", "generalization"),
                       ("Cross-model transfer", "transfer"), ("Steering", "steering")):
        if summary[key]:
            lines += [f"## {title}", "", "```json", json.dumps(summary[key], indent=2), "```", ""]

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (results / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
