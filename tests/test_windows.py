from pathlib import Path

from agent.tokens import count_tokens, savings
from agent.windows import full_read, outline, read_window

TARGET = Path(__file__).resolve().parents[1] / "src" / "agent" / "graph.py"


def test_window_is_far_cheaper_than_full_read():
    full = count_tokens(full_read(TARGET))
    window = count_tokens(read_window(TARGET, 60, radius=15))
    assert window < full * 0.5


def test_window_is_centred_and_clamped():
    text = read_window(TARGET, 1, radius=5)
    assert "lines 1-" in text
    assert "[read_window] not found" not in text


def test_window_handles_missing_file():
    assert "not found" in read_window("does/not/exist.py", 10)


def test_outline_is_cheaper_than_full_read_and_lists_symbols():
    text = outline(TARGET)
    assert "build_graph" in text
    assert count_tokens(text) < count_tokens(full_read(TARGET)) * 0.25


def test_savings_formatting():
    assert savings(1000, 100) == "90.0%"
    assert savings(0, 0) == "n/a"
