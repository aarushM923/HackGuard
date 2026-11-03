# HackGuard

Detecting when AI coding agents are about to cheat on tests (reward hacking) by reading the model's internal activations.

## What this project does
- Builds a suite of coding tasks where cheating is possible (editing tests, hardcoding outputs)
- Runs an open-source coding model (Qwen2.5-Coder-7B) as an agent and labels which runs cheat
- Trains linear probes on the model's activations to predict cheating before it happens
- Measures detection accuracy (AUROC) and how many tokens of warning the probe gives

## Setup
Requires Python 3.11.
```bash
python3.11 -m venv .venv          # create a private environment for this project's packages
source .venv/bin/activate         # use that environment in this terminal
pip install -r requirements.txt   # install the packages
```

## Step 1: the hackable task suite
Turns HumanEval+ and MBPP+ problems into coding tasks in three variants:
- **Honest:** correct visible tests.
- **Impossible:** one visible test expects a wrong answer, so only cheating passes all visible tests.
- **Tempting:** correct tests plus harder edge cases, and the agent is told it is graded only on the tests passing.

Each task also has hidden tests (the full EvalPlus test suite) that the agent never sees. They decide whether a solution is actually correct.

```bash
python -m hackguard.tasks.build_suite   # builds and checks the suite into data/task_suite/
python -m pytest                        # runs the project's tests
```
The build checks every task with the known-correct solution and with a fake cheat, and keeps only problems where all three variants behave as intended. Output: `data/task_suite/tasks.jsonl` (one task per line) and `report.json` (counts, and every skipped problem with its reason).

Code lives in `hackguard/tasks/`:
| File | What it does |
|---|---|
| `schema.py` | Data structures for problems, tasks and test cases, and saving them as JSONL |
| `loaders.py` | Downloads HumanEval+ and MBPP+ and converts them to one format |
| `cases.py` | Reads EvalPlus test code into (input, expected output) pairs and picks visible tests |
| `variants.py` | Makes the honest, impossible and tempting versions of a problem |
| `workspace.py` | Writes a task to its own folder, runs tests with a time limit, grades with hidden tests |
| `build_suite.py` | Builds, checks and saves the whole suite |

## Status
In progress (October 2026). Built by Aarush Maheshwari, Columbia University.
