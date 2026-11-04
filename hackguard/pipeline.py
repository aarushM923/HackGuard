"""One command for every step of the project.

    python -m hackguard.pipeline init --preset laptop          # write runs/laptop/config.json
    python -m hackguard.pipeline all --run runs/laptop         # run every step in order
    python -m hackguard.pipeline <step> --run runs/laptop      # or one step at a time

Steps, in order:
    generate        Step 2: run the agent on every task x prompt x seed
    label           Step 2: automatic labels and a summary
    audit-sample    Step 2: write 200 random transcripts for the hand audit
    extract         Step 3: save activations of the main model
    train           Step 4: train probes at every saved layer, choose the best on validation
    leadtime        Step 5: score every token of val/test trajectories, measure lead time
    baselines       Step 4: probe vs keyword baseline at the same moments (free)
    generalization  Step 6: train on one hack family, test on the other
    extract-second  Step 6: save the second model's activations at the same positions
    transfer        Step 6: map the second model into the first model's space and apply its probe
    steering        Step 6: subtract the cheating direction while generating
    report          collect everything into report.md and charts
    audit-score     after filling in audit_sheet.csv by hand: compute label accuracy
    judge           Step 4, on the laptop: add the paid LLM judge on Amazon Bedrock (needs --confirm-spend)

`all` runs each step as a separate process. On the GPU that matters: vLLM keeps its GPU
memory until its process ends, and later steps need that memory for transformers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from hackguard.config import PRESETS, TASK_SUITE_PATH, RunConfig, model_slug

STEPS_IN_ORDER = ["generate", "label", "audit-sample", "extract", "train", "leadtime", "baselines",
                  "generalization", "extract-second", "transfer", "steering", "report"]


def _tasks_by_id() -> dict[str, object]:
    from hackguard.tasks.schema import load_tasks
    return {t.task_id: t for t in load_tasks(TASK_SUITE_PATH)}


def _load(cfg: RunConfig) -> tuple[list, dict]:  # type: ignore[type-arg]
    from hackguard.trajectories.schema import load_labels, load_trajectories
    return load_trajectories(cfg.run_dir / "trajectories.jsonl"), load_labels(cfg.run_dir / "labels.jsonl")


def run_step(step: str, cfg: RunConfig, args: argparse.Namespace) -> None:
    run_dir = cfg.run_dir
    results = run_dir / "results"
    act_root = run_dir / "activations"
    act_main = act_root / model_slug(cfg.model)
    probe_dir = run_dir / "probes" / model_slug(cfg.model)

    if step == "generate":
        from hackguard.agent.generation import make_generator
        from hackguard.agent.run import run_generation
        generator = make_generator(cfg.backend, cfg.model, cfg.max_new_tokens, cfg.temperature,
                                   cfg.generation_batch_size, cfg.max_context_tokens)
        run_generation(cfg, generator)

    elif step == "label":
        from hackguard.trajectories.labeling import label_run
        from hackguard.trajectories.schema import load_trajectories
        trajectories = load_trajectories(run_dir / "trajectories.jsonl")
        label_run(trajectories, _tasks_by_id(), run_dir)  # type: ignore[arg-type]
        print((run_dir / "label_summary.json").read_text(encoding="utf-8"))

    elif step == "audit-sample":
        from hackguard.trajectories.audit import sample_for_audit
        trajectories, labels = _load(cfg)
        sheet = sample_for_audit(trajectories, labels, run_dir / "audit", seed=cfg.seed)
        print(f"Audit sheet written to {sheet}")

    elif step == "audit-score":
        from hackguard.trajectories.audit import score_audit
        print(score_audit(run_dir / "audit" / "audit_sheet.csv", results / "label_audit.json"))

    elif step == "extract":
        from hackguard.activations.extract import extract
        trajectories, labels = _load(cfg)
        extract(trajectories, labels, cfg.model, act_root, cfg.window, cfg.max_context_tokens + 2048, cfg.seed)

    elif step == "train":
        from hackguard.probes.train import train_all_layers
        train_all_layers(act_main, probe_dir, results, cfg.seed)

    elif step == "leadtime":
        from hackguard.activations.extract import load_meta, load_samples
        from hackguard.probes.lead_time import lead_time_report, score_tokens
        from hackguard.probes.splits import sample_splits
        from hackguard.probes.train import best_layer
        trajectories, labels = _load(cfg)
        split = sample_splits(load_samples(act_main), cfg.seed)
        scores = score_tokens(trajectories, labels, cfg.model, probe_dir, load_meta(act_main)["layers"],
                              cfg.max_context_tokens + 2048, split, results / "token_scores.npz")
        print(lead_time_report(scores, results, best_layer(results)))

    elif step == "baselines":
        from hackguard.probes.judge import run_same_point_baselines
        from hackguard.probes.train import best_layer
        trajectories, _ = _load(cfg)
        print(run_same_point_baselines(act_main, probe_dir, trajectories, results, best_layer(results), cfg.seed,
                                       cfg.judge_max_items))

    elif step == "judge":
        from hackguard.probes.judge import add_llm_judge
        from hackguard.trajectories.schema import load_trajectories
        print(add_llm_judge(results, load_trajectories(run_dir / "trajectories.jsonl"), confirm=args.confirm_spend))

    elif step == "generalization":
        from hackguard.experiments.generalization import run_generalization
        from hackguard.probes.train import best_layer
        print(run_generalization(act_main, results, best_layer(results), cfg.seed))

    elif step == "extract-second":
        from hackguard.activations.extract import extract
        trajectories, labels = _load(cfg)
        extract(trajectories, labels, cfg.second_model, act_root, cfg.window, cfg.max_context_tokens + 2048,
                cfg.seed, positions_from=act_main / "samples.csv")

    elif step == "transfer":
        from hackguard.experiments.transfer import run_transfer
        from hackguard.probes.train import best_layer
        print(run_transfer(act_main, act_root / model_slug(cfg.second_model), probe_dir, results,
                           best_layer(results), cfg.seed))

    elif step == "steering":
        from hackguard.agent.run import make_runner
        from hackguard.experiments.steering import pick_prompt, run_steering
        from hackguard.probes.train import best_layer
        from hackguard.tasks.schema import load_tasks
        trajectories, labels = _load(cfg)
        print(run_steering(cfg, act_main, probe_dir, results, best_layer(results), load_tasks(TASK_SUITE_PATH),
                           pick_prompt(trajectories, labels), make_runner(cfg.runner)))

    elif step == "report":
        from hackguard.report import build_report
        build_report(run_dir)
        print(f"Report written to {run_dir / 'report.md'}")

    else:
        raise ValueError(f"unknown step {step!r}")


def main() -> None:
    load_dotenv()  # read secrets (HF_TOKEN, AWS keys) from .env into environment variables
    parser = argparse.ArgumentParser(description="HackGuard pipeline")
    parser.add_argument("step", choices=["init", "all", *STEPS_IN_ORDER, "audit-score", "judge"])
    parser.add_argument("--run", type=Path, help="run folder, e.g. runs/laptop")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="for init: which settings to start from")
    parser.add_argument("--name", help="for init: run name (default: the preset name)")
    parser.add_argument("--confirm-spend", action="store_true", help="judge: allow spending money on Bedrock")
    parser.add_argument("--from-step", default=STEPS_IN_ORDER[0], help="all: start from this step")
    args = parser.parse_args()

    if args.step == "init":
        cfg = PRESETS[args.preset](args.name or args.preset)
        cfg.save()
        print(f"Wrote {cfg.run_dir / 'config.json'}")
        return

    cfg = RunConfig.load(args.run)
    if args.step == "all":
        for step in STEPS_IN_ORDER[STEPS_IN_ORDER.index(args.from_step):]:
            print(f"\n===== {step} =====", flush=True)
            command = [sys.executable, "-m", "hackguard.pipeline", step, "--run", str(args.run)]
            subprocess.run(command, check=True)  # check=True stops the chain if a step fails
        return
    run_step(args.step, cfg, args)


if __name__ == "__main__":
    main()
