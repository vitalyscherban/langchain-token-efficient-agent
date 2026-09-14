"""Generate a realistic, large CI log so the benchmark has something to chew on.

Kept deterministic (no randomness) so token counts are reproducible run to run.
"""

from __future__ import annotations

from pathlib import Path

PREAMBLE = """##[group]Run actions/checkout@v4
Syncing repository: acme/payments-api
##[endgroup]
##[group]Install dependencies
Collecting pytest==8.2.0
  Downloading pytest-8.2.0-py3-none-any.whl (339 kB)
Collecting httpx==0.27.0
  Downloading httpx-0.27.0-py3-none-any.whl (75 kB)
Requirement already satisfied: certifi in /usr/lib/python3.11/site-packages
Installing collected packages: pytest, httpx
##[endgroup]
============================= test session starts ==============================
platform linux -- Python 3.11.9, pytest-8.2.0, pluggy-1.5.0
rootdir: /home/runner/work/payments-api/payments-api
collected 1284 items
"""

FAILURE_BLOCK = '''
_________________________ test_refund_expired_charge __________________________

    def test_refund_expired_charge(client, seeded_charge):
        seeded_charge.captured_at = datetime(2024, 1, 1)
>       response = client.post(f"/charges/{seeded_charge.id}/refund")

tests/test_refunds.py:118:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  File "/home/runner/work/payments-api/src/api/routes/refunds.py", line 88, in refund
    window = charge.captured_at - datetime.now(timezone.utc)
  File "/usr/lib/python3.11/site-packages/httpx/_client.py", line 1145, in post
    return self.request("POST", url, **kwargs)
  File "/home/runner/work/payments-api/src/services/refund_policy.py", line 42, in within_window
    return (now - captured_at) < REFUND_WINDOW
E   TypeError: can't subtract offset-naive and offset-aware datetimes

src/services/refund_policy.py:42: TypeError
'''

FAILURE_BLOCK_2 = '''
______________________ test_webhook_signature_rejected ________________________

    def test_webhook_signature_rejected(client):
>       assert response.status_code == 401
E       assert 500 == 401

  File "/home/runner/work/payments-api/src/api/webhooks.py", line 63, in verify
    digest = hmac.new(secret, payload, hashlib.sha256).hexdigest()
E   TypeError: key: expected bytes or bytearray, but got 'str'

src/api/webhooks.py:63: TypeError
'''

TRAILER = """=========================== short test summary info ============================
FAILED tests/test_refunds.py::test_refund_expired_charge - TypeError: can't su...
FAILED tests/test_webhooks.py::test_webhook_signature_rejected - TypeError: key...
=================== 2 failed, 1282 passed in 214.83s (0:03:34) =================
"""


def make_ci_log(passing_tests: int = 1282) -> str:
    """Assemble a log dominated by passing-test noise, like the real thing."""
    noise_lines = []
    for index in range(passing_tests):
        module = f"tests/test_module_{index % 40}.py"
        noise_lines.append(
            f"{module}::test_case_{index} PASSED"
            f"                                        [{(index * 100) // passing_tests:>3}%]"
        )
        if index % 25 == 0:
            noise_lines.append(
                f"DEBUG httpx: HTTP Request: GET http://localhost/health "
                f'"HTTP/1.1 200 OK" elapsed=0.00{index % 9}s'
            )
    return "\n".join([PREAMBLE, *noise_lines, FAILURE_BLOCK, FAILURE_BLOCK_2, TRAILER])


SAMPLE_DIFF = """diff --git a/package-lock.json b/package-lock.json
index 1111111..2222222 100644
--- a/package-lock.json
+++ b/package-lock.json
@@ -1,4000 +1,4200 @@
-  "lockfileVersion": 2,
+  "lockfileVersion": 3,
""" + "\n".join(f'+    "resolved": "https://registry.npmjs.org/pkg-{i}",' for i in range(600)) + """
diff --git a/src/services/refund_policy.py b/src/services/refund_policy.py
index 3333333..4444444 100644
--- a/src/services/refund_policy.py
+++ b/src/services/refund_policy.py
@@ -38,7 +38,7 @@ REFUND_WINDOW = timedelta(days=30)
 def within_window(captured_at, now=None):
-    now = now or datetime.now(timezone.utc)
+    now = now or datetime.now()
     return (now - captured_at) < REFUND_WINDOW
"""


def make_large_module(path: Path, functions: int = 120) -> Path:
    """Write a service module of realistic size (~1,200 lines).

    Benchmarking window reads against this repo's own small files would flatter
    the naive baseline; production modules are what the lever actually targets.
    The bug from the CI log lands at line 42 so the window read is comparable.
    """
    header = [
        '"""Refund eligibility rules."""',
        "",
        "from datetime import datetime, timedelta, timezone",
        "",
        "REFUND_WINDOW = timedelta(days=30)",
        "",
    ]
    while len(header) < 38:
        header.append(f"# policy note {len(header)}: see RFC-{200 + len(header)}")

    body = [
        "def within_window(captured_at, now=None):",
        '    """Return True when the charge is still refundable."""',
        "    now = now or datetime.now()",
        "    return (now - captured_at) < REFUND_WINDOW",
        "",
    ]

    for index in range(functions):
        body += [
            f"def rule_{index:03d}(charge, context=None):",
            f'    """Eligibility rule {index} applied after the window check."""',
            "    context = context or {}",
            f"    threshold = {100 + index * 7}",
            "    if charge.amount_cents < threshold:",
            f"        return False, 'below_threshold_{index}'",
            f"    if context.get('region') in ('EU', 'UK') and charge.age_days > {index + 1}:",
            f"        return False, 'regional_limit_{index}'",
            "    return True, None",
            "",
        ]

    path.write_text("\n".join(header + body), encoding="utf-8")
    return path

