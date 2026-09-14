"""Stage 1: prune a raw CI log down to the failing evidence.

A 40k-line pytest log is ~90% progress dots, install chatter, and passing tests.
The model only needs the failure blocks and the frames that point at our own
source. This module does that extraction deterministically, in Python, for zero
tokens -- the cheapest possible optimization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import DIFF_DENYLIST

# pytest section banner, e.g. "____ test_login_rejects_expired_token ____"
FAILURE_BANNER = re.compile(r"^_{3,}\s+(?P<test>[\w\.\[\]:<>-]+)\s+_{3,}$")
SHORT_SUMMARY = re.compile(r"^(FAILED|ERROR)\s+(?P<nodeid>\S+)")
# "  File "/repo/src/auth.py", line 88, in verify"
STACK_FRAME = re.compile(
    r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)'
)
EXCEPTION_LINE = re.compile(r"^E\s+(?P<exc>\w+(?:Error|Exception|Warning)[^\n]*)")
NOISE = re.compile(
    r"^(Collecting |Downloading |Requirement already satisfied|Installing collected|"
    r"\s*\d+%\s*\|| *\.+ *$|=+ warnings summary =+|##\[group\]|##\[endgroup\])"
)


@dataclass
class StackFrame:
    path: str
    line: int
    func: str

    @property
    def is_project_code(self) -> bool:
        lowered = self.path.replace("\\", "/").lower()
        if "site-packages" in lowered or "/lib/python" in lowered:
            return False
        return not any(token in lowered for token in DIFF_DENYLIST)


@dataclass
class Failure:
    test: str
    exception: str = ""
    frames: list[StackFrame] = field(default_factory=list)
    body: list[str] = field(default_factory=list)

    @property
    def culprit_frame(self) -> StackFrame | None:
        """Deepest frame that lives in our repo -- where a fix likely belongs."""
        for frame in reversed(self.frames):
            if frame.is_project_code:
                return frame
        return self.frames[-1] if self.frames else None

    def render(self, max_body_lines: int = 25) -> str:
        frame = self.culprit_frame
        location = f"{frame.path}:{frame.line} in {frame.func}" if frame else "unknown"
        body = "\n".join(self.body[-max_body_lines:])
        return f"### {self.test}\nwhere: {location}\nerror: {self.exception}\n{body}"


def prune_ci_log(log: str, max_failures: int = 5) -> list[Failure]:
    """Extract structured failures from a raw CI log."""
    failures: list[Failure] = []
    current: Failure | None = None

    for raw_line in log.splitlines():
        banner = FAILURE_BANNER.match(raw_line)
        if banner:
            current = Failure(test=banner.group("test"))
            failures.append(current)
            continue

        if current is None:
            continue

        if raw_line.startswith("=") and "short test summary" in raw_line.lower():
            current = None
            continue

        if NOISE.match(raw_line):
            continue

        frame = STACK_FRAME.match(raw_line)
        if frame:
            current.frames.append(
                StackFrame(
                    path=frame.group("path"),
                    line=int(frame.group("line")),
                    func=frame.group("func"),
                )
            )

        exception = EXCEPTION_LINE.match(raw_line)
        if exception and not current.exception:
            current.exception = exception.group("exc").strip()

        current.body.append(raw_line.rstrip())

    return failures[:max_failures]


def render_failures(failures: list[Failure]) -> str:
    if not failures:
        return "No test failures found in the log."
    return "\n\n".join(failure.render() for failure in failures)


def filter_diff(diff: str) -> str:
    """Drop hunks for generated/vendored files before they reach the prompt."""
    kept: list[str] = []
    keeping = True
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            keeping = not any(token in line for token in DIFF_DENYLIST)
        if keeping:
            kept.append(line)
    return "\n".join(kept)
