"""Manage the EC2 GPU machine from the laptop, using boto3 (Amazon's Python library).

    python -m hackguard.cloud.ec2 setup       # one time: SSH key + firewall rule for your IP
    python -m hackguard.cloud.ec2 launch      # start a new g5.2xlarge (costs money while running)
    python -m hackguard.cloud.ec2 status      # is it running? what is its address?
    python -m hackguard.cloud.ec2 push        # copy the code, task suite and HF_TOKEN to it
    python -m hackguard.cloud.ec2 ssh         # print the command to log in
    python -m hackguard.cloud.ec2 pull gpu    # copy runs/gpu back (without the big activation files)
    python -m hackguard.cloud.ec2 stop        # stop it (GPU charges end; disk is kept)
    python -m hackguard.cloud.ec2 start       # start it again after a stop
    python -m hackguard.cloud.ec2 terminate   # delete it and its disk for good (asks first)

Credentials come from .env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION).
The instance id is remembered in .hackguard_instance.json (ignored by git).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

INSTANCE_TYPE = "g5.2xlarge"  # 1 NVIDIA A10G (24 GB), 8 vCPUs, 32 GB RAM
HOURLY_PRICE_USD = 1.212  # on-demand price in us-east-1 at the time of writing; check the EC2 pricing page
DISK_GB = 200
# AWS publishes the latest "Deep Learning Base" image ID (Ubuntu 22.04 with NVIDIA drivers,
# CUDA and Docker with GPU support preinstalled) under this public parameter name.
AMI_PARAMETER = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
KEY_NAME = "hackguard-key"
KEY_PATH = Path.home() / ".ssh" / "hackguard-key.pem"
SECURITY_GROUP = "hackguard-ssh"
STATE_FILE = Path(".hackguard_instance.json")
TAG = {"Key": "Name", "Value": "hackguard-gpu"}
REMOTE_DIR = "~/HackGuard"


def _session() -> Any:
    import boto3
    load_dotenv()
    return boto3.session.Session(region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _instance_id() -> str:
    if not STATE_FILE.exists():
        sys.exit("No instance recorded. Run `launch` first.")
    return str(json.loads(STATE_FILE.read_text())["instance_id"])


def _describe(ec2: Any, instance_id: str) -> dict[str, Any]:
    result: dict[str, Any] = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
    return result


def _host() -> str:
    info = _describe(_session().client("ec2"), _instance_id())
    ip = info.get("PublicIpAddress")
    if not ip:
        sys.exit(f"The instance is {info['State']['Name']} and has no address. Run `start` first.")
    return f"ubuntu@{ip}"


SSH_OPTIONS = ["-i", str(KEY_PATH), "-o", "StrictHostKeyChecking=accept-new"]


def setup() -> None:
    """Create the SSH key pair and a firewall rule that only lets your current IP in."""
    ec2 = _session().client("ec2")
    if not KEY_PATH.exists():
        key = ec2.create_key_pair(KeyName=KEY_NAME, KeyType="ed25519")
        KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        KEY_PATH.write_text(key["KeyMaterial"])
        KEY_PATH.chmod(0o400)  # ssh refuses keys that other users can read
        print(f"Saved new SSH key to {KEY_PATH}")
    my_ip = urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip()
    groups = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SECURITY_GROUP]}])["SecurityGroups"]
    group_id = groups[0]["GroupId"] if groups else ec2.create_security_group(
        GroupName=SECURITY_GROUP, Description="SSH access to the HackGuard GPU machine")["GroupId"]
    try:
        ec2.authorize_security_group_ingress(GroupId=group_id, IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
            "IpRanges": [{"CidrIp": f"{my_ip}/32", "Description": "my laptop"}]}])
    except ec2.exceptions.ClientError as exc:
        if "InvalidPermission.Duplicate" not in str(exc):
            raise
    print(f"Firewall group {SECURITY_GROUP} allows SSH only from {my_ip}.")


def launch() -> None:
    if STATE_FILE.exists():
        sys.exit(f"An instance is already recorded in {STATE_FILE}. Use `status`, or `terminate` it first.")
    session = _session()
    ec2, ssm = session.client("ec2"), session.client("ssm")
    ami = ssm.get_parameter(Name=AMI_PARAMETER)["Parameter"]["Value"]
    group_id = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SECURITY_GROUP]}])["SecurityGroups"][0]["GroupId"]
    user_data = Path("cloud/user_data.sh").read_text().replace("__WATCHDOG_SCRIPT__", Path("cloud/watchdog.sh").read_text())
    response = ec2.run_instances(
        ImageId=ami,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        SecurityGroupIds=[group_id],
        MinCount=1, MaxCount=1,
        UserData=user_data,
        # "stop" means a shutdown from inside (our watchdog) stops the machine instead of deleting it.
        InstanceInitiatedShutdownBehavior="stop",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": DISK_GB, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        TagSpecifications=[{"ResourceType": "instance", "Tags": [TAG]}],
    )
    instance_id = response["Instances"][0]["InstanceId"]
    STATE_FILE.write_text(json.dumps({"instance_id": instance_id}))
    print(f"Launched {instance_id} ({INSTANCE_TYPE}, about ${HOURLY_PRICE_USD:.2f} per hour while running).")
    print("Waiting for it to start...")
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    print(f"Running at {_describe(ec2, instance_id).get('PublicIpAddress')}. "
          "First-boot setup takes about 5 minutes; then run `push`.")


def status() -> None:
    info = _describe(_session().client("ec2"), _instance_id())
    print(f"{info['InstanceId']}: {info['State']['Name']}, address {info.get('PublicIpAddress', 'none')}")


def _rsync(source: str, dest: str, excludes: list[str]) -> None:
    # -a keeps the folder structure and file details, -z compresses during transfer.
    command = ["rsync", "-az", "-e", "ssh " + " ".join(SSH_OPTIONS)]
    command += [f"--exclude={e}" for e in excludes] + [source, dest]
    subprocess.run(command, check=True)


def push() -> None:
    """Copy code and the task suite. Only HF_TOKEN is copied from .env, never AWS keys."""
    host = _host()
    _rsync("./", f"{host}:{REMOTE_DIR}/", [".venv", "runs", ".git", "__pycache__", ".env", ".mypy_cache",
                                            ".pytest_cache", ".hackguard_instance.json"])
    load_dotenv()
    token = os.environ.get("HF_TOKEN", "")
    remote_env = f"HF_TOKEN={token}\n" if token else ""
    subprocess.run(["ssh", *SSH_OPTIONS, host, f"cat > {REMOTE_DIR}/.env && chmod 600 {REMOTE_DIR}/.env"],
                   input=remote_env, text=True, check=True)
    print("Pushed code, data/task_suite and .env (HF_TOKEN only).")


def pull(run_name: str, with_activations: bool) -> None:
    excludes = [] if with_activations else ["*.safetensors"]
    Path("runs").mkdir(exist_ok=True)
    _rsync(f"{_host()}:{REMOTE_DIR}/runs/{run_name}", "runs/", excludes)
    print(f"Copied runs/{run_name}")


def ssh_command() -> None:
    print("ssh " + " ".join(SSH_OPTIONS) + " " + _host())


def stop() -> None:
    _session().client("ec2").stop_instances(InstanceIds=[_instance_id()])
    print("Stopping. GPU charges end once it is stopped; the disk is kept (a few dollars per month).")


def start() -> None:
    ec2 = _session().client("ec2")
    instance_id = _instance_id()
    ec2.start_instances(InstanceIds=[instance_id])
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    print(f"Running again at {_describe(ec2, instance_id).get('PublicIpAddress')} (the address changes after a stop).")


def terminate() -> None:
    instance_id = _instance_id()
    answer = input(f"This permanently deletes {instance_id} and its disk, including any results not pulled. "
                   f"Type the instance id to confirm: ")
    if answer.strip() != instance_id:
        sys.exit("Not confirmed; nothing deleted.")
    _session().client("ec2").terminate_instances(InstanceIds=[instance_id])
    STATE_FILE.unlink()
    print("Terminating.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the HackGuard EC2 GPU machine")
    parser.add_argument("command", choices=["setup", "launch", "status", "push", "pull", "ssh", "stop", "start", "terminate"])
    parser.add_argument("run_name", nargs="?", default="gpu", help="for pull: which run folder")
    parser.add_argument("--with-activations", action="store_true", help="for pull: also copy activation files (large)")
    args = parser.parse_args()
    actions = {"setup": setup, "launch": launch, "status": status, "push": push, "ssh": ssh_command,
               "stop": stop, "start": start, "terminate": terminate}
    if args.command == "pull":
        pull(args.run_name, args.with_activations)
    else:
        actions[args.command]()


if __name__ == "__main__":
    main()
