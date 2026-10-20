"""
The unattended daily run, for the scheduled task on the always-on PC.

Steps, each in its own process:
  1. python -m pipeline.daily --refresh-only
       refresh the data and score last night's predictions
  2. python -m bot.discord_bot --webhook --results [--props --saves ...]
       predict today, write the prediction log, post to Discord
  3. commit the prediction log (locally; --push to push it as well)

Step 2 is the only writer of today's predictions.  Running ``pipeline.daily``
in full as well would predict twice, and the second write replaces the first
with a price taken a minute later.

Output is appended to ``logs/daily-YYYY-MM-DD.log``.  Nobody watches this PC,
so a failed step stops the run and posts a one-line notice to the same
webhook.  A failed refresh stops it before predicting: features that missed
last night's games are the one-game-stale snapshot this project has already
had to fix three times, and predictions built on them would go into the log.

While it runs it holds Windows awake.  The task's wake timer brings the PC
out of sleep, but a wake with nobody at the keyboard goes back to sleep
within minutes unless something asks it not to.

Usage:
    python -m pipeline.scheduled_run [--push] [--extra=--props --extra=--saves]
"""

import argparse
import ctypes
import logging
import os
import socket
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "logs"
HISTORY_FILE = "data/predictions/prediction_history.parquet"
STEP_TIMEOUT = 90 * 60  # seconds; a normal refresh takes a few minutes

# SetThreadExecutionState flags
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def steps(extras: Optional[list[str]] = None) -> list[tuple[str, list[str]]]:
    """(name, python arguments) for each step, in order."""
    return [
        ("refresh", ["-m", "pipeline.daily", "--refresh-only"]),
        ("predict and post",
         ["-m", "bot.discord_bot", "--webhook", "--results", *(extras or [])]),
    ]


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep until called again with False.  No-op elsewhere."""
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return
    flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0)
    windll.kernel32.SetThreadExecutionState(flags)


def run_step(args: list[str], log_path: Path) -> int:
    """Run ``python <args>`` from the repo root, appending output to the log."""
    env = dict(os.environ, PYTHONUTF8="1")
    with open(log_path, "ab") as log:
        try:
            return subprocess.run(
                [sys.executable, *args], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=STEP_TIMEOUT,
            ).returncode
        except subprocess.TimeoutExpired:
            log.write(f"\n*** timed out after {STEP_TIMEOUT // 60} minutes\n".encode())
            return -1


def commit_history(today: date, push: bool, log_path: Path) -> int:
    """Commit the prediction log alone, if it changed.  Returns an exit code."""
    def git(*args) -> int:
        with open(log_path, "ab") as log:
            return subprocess.run(
                ["git", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            ).returncode

    if git("add", "--", HISTORY_FILE):
        return 1
    # Exit 0 from --quiet means nothing staged: an empty slate, or every game
    # already under way.
    if git("diff", "--cached", "--quiet", "--", HISTORY_FILE) == 0:
        return 0
    # The pathspec keeps anything else in the working tree out of the commit.
    code = git("commit", "-m", f"Log predictions for {today}", "--", HISTORY_FILE)
    if code == 0 and push:
        code = git("push")
    return code


def notify_failure(step: str, code: int, log_path: Path) -> None:
    """One-line notice to the Discord webhook.  Best effort."""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    text = (
        f"NHLAnalyzer daily run failed at **{step}** (exit {code}) on "
        f"{socket.gethostname()}. See `logs/{log_path.name}`."
    )
    try:
        httpx.post(url, json={"content": text}, timeout=15).raise_for_status()
    except Exception as e:
        logger.error("Could not post the failure notice: %s", e)


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Unattended daily NHL run")
    parser.add_argument("--push", action="store_true",
                        help="Push the prediction log commit as well")
    parser.add_argument("--extra", action="append", default=[],
                        help="Extra flag for the Discord post, e.g. --extra=--props "
                             "(repeatable)")
    args = parser.parse_args(argv)

    today = date.today()
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"daily-{today}.log"
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    keep_awake(True)
    try:
        for name, step_args in steps(args.extra):
            logger.info("Step: %s", name)
            code = run_step(step_args, log_path)
            if code != 0:
                logger.error("Step %s failed with exit %d", name, code)
                notify_failure(name, code, log_path)
                return code
        logger.info("Step: commit the prediction log")
        code = commit_history(today, args.push, log_path)
        if code != 0:
            logger.error("Committing the prediction log failed with exit %d", code)
            notify_failure("commit the prediction log", code, log_path)
            return code
        logger.info("Done")
        return 0
    finally:
        keep_awake(False)


if __name__ == "__main__":
    sys.exit(main())
