"""The unattended daily run.

Nobody watches the PC it runs on, so the properties that matter are about
failure: a failed refresh must stop the run before anything is predicted or
logged, it must say so somewhere a person will see, and the PC must be let
go back to sleep however the run ends.
"""

import subprocess
from datetime import date

import pytest

from pipeline import scheduled_run as sr


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """main() with every side effect recorded instead of performed."""
    calls = {"steps": [], "awake": [], "notices": [], "commits": []}
    codes = {}

    def run_step(args, log_path):
        calls["steps"].append(args)
        return codes.get(args[1], 0)

    monkeypatch.setattr(sr, "LOG_DIR", tmp_path)
    monkeypatch.setattr(sr, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(sr, "run_step", run_step)
    monkeypatch.setattr(sr, "keep_awake", lambda on: calls["awake"].append(on))
    monkeypatch.setattr(
        sr, "notify_failure", lambda step, code, log: calls["notices"].append(step),
    )
    monkeypatch.setattr(
        sr, "commit_history",
        lambda today, push, log: calls["commits"].append(push) or codes.get("commit", 0),
    )
    return calls, codes


class TestSteps:
    def test_refresh_runs_first_and_does_not_predict(self):
        (name, args), _ = sr.steps()
        assert args == ["-m", "pipeline.daily", "--refresh-only"]

    def test_the_post_is_the_only_step_that_predicts(self):
        _, (name, args) = sr.steps()
        assert args[:2] == ["-m", "bot.discord_bot"]
        assert "--webhook" in args and "--results" in args

    def test_extras_go_to_the_post(self):
        _, (_, args) = sr.steps(["--props", "--saves"])
        assert args[-2:] == ["--props", "--saves"]


class TestMain:
    def test_a_clean_run_runs_everything_and_commits(self, harness):
        calls, _ = harness
        assert sr.main([]) == 0
        assert [a[1] for a in calls["steps"]] == ["pipeline.daily", "bot.discord_bot"]
        assert calls["commits"] == [False]
        assert calls["notices"] == []

    def test_a_failed_refresh_stops_before_predicting(self, harness):
        calls, codes = harness
        codes["pipeline.daily"] = 1
        assert sr.main([]) == 1
        assert [a[1] for a in calls["steps"]] == ["pipeline.daily"]
        assert calls["commits"] == []
        assert calls["notices"] == ["refresh"]

    def test_a_failed_post_is_reported_and_not_committed(self, harness):
        calls, codes = harness
        codes["bot.discord_bot"] = 1
        assert sr.main([]) == 1
        assert calls["commits"] == []
        assert calls["notices"] == ["predict and post"]

    def test_a_failed_commit_is_reported(self, harness):
        calls, codes = harness
        codes["commit"] = 128
        assert sr.main([]) == 128
        assert calls["notices"] == ["commit the prediction log"]

    @pytest.mark.parametrize("failing", [None, "pipeline.daily", "bot.discord_bot"])
    def test_the_pc_is_let_sleep_however_the_run_ends(self, harness, failing):
        calls, codes = harness
        if failing:
            codes[failing] = 1
        sr.main([])
        assert calls["awake"] == [True, False]

    def test_push_and_extras_are_passed_through(self, harness):
        calls, _ = harness
        sr.main(["--push", "--extra=--props"])
        assert calls["commits"] == [True]
        assert calls["steps"][1][-1] == "--props"


@pytest.fixture
def git(monkeypatch, tmp_path):
    """commit_history against a recorded git; set codes[subcommand] to fail one."""
    seen, codes = [], {}

    def fake_run(cmd, **kwargs):
        seen.append(cmd[1:])
        return subprocess.CompletedProcess(cmd, codes.get(cmd[1], 0))

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    return seen, codes, tmp_path / "run.log"


class TestCommitHistory:
    def test_only_the_prediction_log_is_committed(self, git):
        seen, codes, log = git
        codes["diff"] = 1  # something staged
        assert sr.commit_history(date(2026, 10, 21), push=False, log_path=log) == 0
        commit = next(c for c in seen if c[0] == "commit")
        assert commit[-2:] == ["--", sr.HISTORY_FILE]
        assert not any(c[0] == "push" for c in seen)

    def test_nothing_staged_means_no_commit(self, git):
        seen, codes, log = git
        codes["diff"] = 0
        assert sr.commit_history(date(2026, 10, 21), push=True, log_path=log) == 0
        assert [c[0] for c in seen] == ["add", "diff"]

    def test_push_only_when_asked(self, git):
        seen, codes, log = git
        codes["diff"] = 1
        sr.commit_history(date(2026, 10, 21), push=True, log_path=log)
        assert seen[-1] == ["push"]


class TestNotice:
    def test_no_webhook_means_no_post(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        posted = []
        monkeypatch.setattr(sr.httpx, "post", lambda *a, **k: posted.append(a))
        sr.notify_failure("refresh", 1, tmp_path / "daily.log")
        assert posted == []

    def test_a_dead_webhook_does_not_raise(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/hook")

        def boom(*a, **k):
            raise OSError("no network")

        monkeypatch.setattr(sr.httpx, "post", boom)
        sr.notify_failure("refresh", 1, tmp_path / "daily.log")
