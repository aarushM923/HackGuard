import subprocess
from pathlib import Path

import pytest

SCRIPTS = ["cloud/watchdog.sh", "cloud/user_data.sh", "cloud/remote_setup.sh", "cloud/remote_run.sh"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_shell_scripts_have_valid_syntax(script: str) -> None:
    # `bash -n` reads a script and reports syntax errors without running anything.
    subprocess.run(["bash", "-n", script], check=True)


def test_user_data_embeds_the_watchdog() -> None:
    user_data = Path("cloud/user_data.sh").read_text().replace("__WATCHDOG_SCRIPT__", Path("cloud/watchdog.sh").read_text())
    assert "__WATCHDOG_SCRIPT__" not in user_data
    assert "IDLE_MINUTES=30" in user_data and "shutdown -h now" in user_data
    assert "systemctl enable --now hackguard-watchdog.timer" in user_data
    subprocess.run(["bash", "-n"], input=user_data, text=True, check=True)


def test_launch_settings() -> None:
    from hackguard.cloud import ec2
    assert ec2.INSTANCE_TYPE == "g5.2xlarge"
    source = Path("hackguard/cloud/ec2.py").read_text()
    assert 'InstanceInitiatedShutdownBehavior="stop"' in source  # watchdog shutdown stops, never deletes
    assert '".env"' in source  # .env (with AWS keys) is never copied to the machine
