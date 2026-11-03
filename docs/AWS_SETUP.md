# Running HackGuard on AWS

This guide takes you from an AWS account with credits to a finished GPU run. Steps marked
**(costs money)** start charges; everything else is free.

The GPU machine is one **g5.2xlarge**: 1 NVIDIA A10G GPU (24 GB), 8 vCPUs, 32 GB RAM, about
**$1.21 per hour** on demand in us-east-1 (check the EC2 pricing page for your region). Your
GPU quota of 8 vCPUs fits exactly one of these.

## Part 1: one-time account setup (free, about 20 minutes)

### 1.1 Create a user for the laptop

The laptop needs its own AWS login key so the scripts can launch the machine and call Bedrock.
Do not use your root account's keys.

1. Sign in to the AWS console. Open **IAM** → **Users** → **Create user**.
2. Name: `hackguard-laptop`. Leave "Provide user access to the AWS Management Console" **unchecked**. Next.
3. Choose **Attach policies directly** and tick:
   - `AmazonEC2FullAccess` (launch, stop and delete the GPU machine)
   - `AmazonSSMReadOnlyAccess` (look up the latest Deep Learning machine image)
4. Create the user. Open it → **Permissions** → **Add permissions** → **Create inline policy** →
   **JSON**, paste the following, and name it `hackguard-bedrock`:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": ["bedrock-mantle:CreateInference", "bedrock:InvokeModel"],
       "Resource": "*"
     }]
   }
   ```
5. Open the **Security credentials** tab → **Create access key** → choose "Local code" → create.
   Copy both values now (the secret is shown only once).

### 1.2 Put the keys in `.env`

In the project folder, create or edit `.env` (it is in `.gitignore`, so it is never committed):

```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
HF_TOKEN=hf_...
```

Use the region where your GPU quota is. `HF_TOKEN` is only needed for Llama (see 1.5).

### 1.3 Check the GPU quota

Console → **Service Quotas** → **Amazon EC2** → search "Running On-Demand G and VT instances".
The value for your region must be at least **8**. You said it is 8, which is exactly enough.

### 1.4 Turn on Claude in Bedrock (for the LLM judge)

Console → **Amazon Bedrock** (same region) → **Model access**. Check that **Claude Opus 5.5**
is available to your account. If AWS asks for a short use-case form the first time you use
Anthropic models, fill it in (for example: "research on detecting reward hacking in coding
agents"). Approval is usually immediate.

### 1.5 Hugging Face token and the Llama license (for the transfer experiment)

1. Make a free account at huggingface.co.
2. Open https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and accept Meta's license.
   Approval usually takes minutes to a few hours; the page will say when you have access.
3. Settings → **Access Tokens** → create a token of type **Read**, and put it in `.env` as `HF_TOKEN`.

If you would rather skip this, set `second_model` to `deepseek-ai/deepseek-coder-6.7b-instruct`
in `runs/gpu/config.json` after creating it (no account needed).

### 1.6 Set a budget alarm (strongly recommended)

Console → **Billing and Cost Management** → **Budgets** → **Create budget** → "Monthly cost
budget", amount e.g. **$100**, with email alerts at 50% and 80%. Free, and it emails you if
anything keeps running by mistake.

## Part 2: launch the machine

All commands run on the laptop from the project folder with the virtual environment active
(`source .venv/bin/activate`).

```bash
python -m hackguard.cloud.ec2 setup     # creates an SSH key and a firewall rule for your IP (free)
python -m hackguard.cloud.ec2 launch    # (costs money) starts the g5.2xlarge
```

`launch` uses AWS's Deep Learning Base image (Ubuntu 22.04 with NVIDIA drivers and Docker),
a 200 GB disk, and installs the **auto-shutdown watchdog** at first boot:

- every 5 minutes it checks whether the GPU is busy, the pipeline is running, or you typed
  in an SSH session in the last 30 minutes,
- after **30 idle minutes** it shuts the machine down,
- after **36 hours** it shuts down no matter what,
- shutting down **stops** the machine (GPU charges end; the disk and results are kept). A
  stopped machine's 200 GB disk costs about $16 per month until you terminate it.

Wait about 5 minutes for first-boot setup, then:

```bash
python -m hackguard.cloud.ec2 push      # copies code, the task suite and HF_TOKEN (never your AWS keys)
python -m hackguard.cloud.ec2 ssh       # prints the ssh command; run it to log in
```

On the machine:

```bash
cd ~/HackGuard
bash cloud/remote_setup.sh              # Python env, Docker sandbox, all tests (about 15 minutes)
```

All tests must pass, including the Docker sandbox test that is skipped on the laptop.

## Part 3: pilot, then the full run

```bash
bash cloud/remote_run.sh pilot          # 34 problems (102 tasks x 3 prompts), roughly 1 hour
tmux attach -t hackguard                # watch it; detach with Ctrl-b then d
```

When it finishes, check how often the 7B model cheated:

```bash
cat runs/pilot/label_summary.json
```

If the pilot has a reasonable number of hacks (the full run needs at least 300 in total;
the pilot is about 1/12 of the full run, so roughly 25+ hacks), start the full run:

```bash
bash cloud/remote_run.sh gpu            # all 400 problems; several hours
```

If the model rarely cheats, stop and tell Claude before the full run: the spec's fix is more
impossible tasks and stronger pressure prompts.

You can log out while it runs. When the run finishes, the GPU goes idle and the watchdog stops
the machine 30 minutes later.

## Part 4: bring results home, run the judge, finish

```bash
python -m hackguard.cloud.ec2 start              # only if the watchdog already stopped it
python -m hackguard.cloud.ec2 pull gpu           # copies runs/gpu (without the large activation files)
python -m hackguard.cloud.ec2 stop
python -m hackguard.pipeline judge --run runs/gpu                   # prints the judge's cost estimate, stops
python -m hackguard.pipeline judge --run runs/gpu --confirm-spend   # (costs money) runs the judge
python -m hackguard.pipeline audit-sample --run runs/gpu            # (already done on the GPU) 200 transcripts
# fill in runs/gpu/audit/audit_sheet.csv by hand, then:
python -m hackguard.pipeline audit-score --run runs/gpu
python -m hackguard.pipeline report --run runs/gpu                  # runs/gpu/report.md
```

When you are sure you have everything, delete the machine and its disk:

```bash
python -m hackguard.cloud.ec2 terminate          # asks you to type the instance id to confirm
```
