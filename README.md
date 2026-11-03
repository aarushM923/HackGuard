# HackGuard

Detecting when AI coding agents are about to cheat on tests (reward hacking) by reading the model's internal activations.

## What this project does
- Builds a suite of coding tasks where cheating is possible (editing tests, hardcoding outputs)
- Runs an open-source coding model (Qwen2.5-Coder-7B) as an agent and labels which runs cheat
- Trains linear probes on the model's activations to predict cheating before it happens
- Measures detection accuracy (AUROC) and how many tokens of warning the probe gives
- Compares the probe with a keyword detector and with Claude (on Amazon Bedrock) reading the transcript
- Stretch experiments: generalization across hack types, transfer to a second model, and steering

## Setup
Requires Python 3.11.
```bash
python3.11 -m venv .venv                   # create a private environment for this project's packages
source .venv/bin/activate                  # use that environment in this terminal
pip install -r requirements-laptop.txt     # on the Mac (the GPU machine uses requirements-gpu.txt)
python -m hackguard.tasks.build_suite      # build the task suite into data/task_suite/
python -m pytest                           # run the project's tests
```

## Running the pipeline
Every step is one command; `all` runs them in order.
```bash
python -m hackguard.pipeline init --preset laptop      # settings: laptop, pilot or gpu
python -m hackguard.pipeline all --run runs/laptop     # generate, label, extract, train, ... report
```
The `laptop` preset runs everything with a tiny model (Qwen2.5-Coder-0.5B) on 21 tasks, to find bugs for free.
The real experiment runs on an AWS EC2 g5.2xlarge: see [docs/AWS_SETUP.md](docs/AWS_SETUP.md).

| Step | Command | What it does |
|---|---|---|
| 2 | `generate` | Runs the agent on every task x system prompt (neutral, pressure, don't modify tests) |
| 2 | `label` | Labels each trajectory hack / honest_success / honest_fail / ambiguous, and finds the first hacking token |
| 2 | `audit-sample`, `audit-score` | 200 random transcripts for a hand check of the labels, then label accuracy |
| 3 | `extract` | Re-runs trajectories through the model and saves activations at every 4th layer |
| 4 | `train` | Logistic regression and difference-in-means probes per layer; best layer chosen on validation |
| 5 | `leadtime` | Scores every token of validation and test trajectories; lead time at a 5% false alarm rate |
| 4 | `baselines`, `judge` | Probe vs keyword detector vs Claude judge at the same moments (`judge` costs money) |
| 6 | `generalization`, `extract-second`, `transfer`, `steering` | Stretch experiments |
| | `report` | `runs/<name>/report.md` with tables and charts |

## Code layout
| Folder | What it contains |
|---|---|
| `hackguard/tasks/` | Step 1: the hackable task suite (loading, variants, workspaces, grading) |
| `hackguard/agent/` | Step 2: prompts, action format, generation backends (transformers, vLLM), agent loop |
| `hackguard/trajectories/` | Step 2: trajectory records, automatic labeling, hand audit |
| `hackguard/activations/` | Step 3: tokens and positions, sampling windows, activation extraction |
| `hackguard/probes/` | Steps 4 and 5: probes, metrics, lead time, keyword and LLM judge baselines |
| `hackguard/experiments/` | Step 6: generalization, cross-model transfer, steering |
| `hackguard/cloud/`, `cloud/` | Launching and managing the EC2 machine; auto-shutdown watchdog |
| `docker/` | The sandbox image that runs model-written code on EC2 |

## Status
In progress (November 2026). Built by Aarush Maheshwari, Columbia University.
