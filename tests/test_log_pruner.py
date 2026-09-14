"""The pruner is the highest-leverage component, so it gets the most tests.

If it silently drops a failure, the agent triages the wrong thing -- cheap and
wrong. These tests pin both the savings and the fidelity.
"""

from agent.log_pruner import filter_diff, prune_ci_log, render_failures
from agent.tokens import count_tokens
from fixtures import SAMPLE_DIFF, make_ci_log


def test_finds_every_failure():
    failures = prune_ci_log(make_ci_log())
    names = [f.test for f in failures]
    assert names == ["test_refund_expired_charge", "test_webhook_signature_rejected"]


def test_captures_exception_type():
    failures = prune_ci_log(make_ci_log())
    assert "TypeError" in failures[0].exception


def test_culprit_frame_skips_site_packages():
    """httpx sits between our frames; the fix belongs in our code, not the lib."""
    frame = prune_ci_log(make_ci_log())[0].culprit_frame
    assert frame is not None
    assert "site-packages" not in frame.path
    assert frame.path.endswith("src/services/refund_policy.py")
    assert frame.line == 42


def test_pruning_cuts_at_least_95_percent():
    log = make_ci_log()
    before = count_tokens(log)
    after = count_tokens(render_failures(prune_ci_log(log)))
    assert after < before * 0.05, f"only cut to {after}/{before}"


def test_rendered_report_keeps_actionable_details():
    report = render_failures(prune_ci_log(make_ci_log()))
    for needle in ("refund_policy.py", "TypeError", "webhooks.py"):
        assert needle in report


def test_empty_log_is_safe():
    assert prune_ci_log("") == []
    assert "No test failures" in render_failures([])


def test_diff_filter_drops_lockfile_keeps_source():
    filtered = filter_diff(SAMPLE_DIFF)
    assert "package-lock.json" not in filtered
    assert "refund_policy.py" in filtered
    assert "datetime.now()" in filtered
    assert count_tokens(filtered) < count_tokens(SAMPLE_DIFF) * 0.1
