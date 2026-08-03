from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from rich.panel import Panel
from rich.text import Text

from mutation_forge.output.panel_copy import (
    copy_text_to_clipboard_osc52,
    osc52_clipboard_sequence,
    panel_copy_path,
    render_panel_copy_text,
    save_panel_copy,
)


def test_panel_copy_path_sanitizes_panel_and_run_names(tmp_path: Path) -> None:
    assert panel_copy_path(
        "Run data!",
        "../../heg live 5",
        tmp_dir=tmp_path,
    ) == tmp_path / "panel-run-data-..-..-heg-live-5.txt"


def test_render_panel_copy_text_exports_plain_rich_text() -> None:
    rendered = render_panel_copy_text(
        "Tokens",
        Panel(Text.from_ansi("\x1b[31mexact\x1b[0m\tusage"), title="Accounting"),
        width=50,
    )
    assert rendered.startswith("# Tokens\n\n")
    assert "exact   usage" in rendered
    assert "\x1b" not in rendered
    assert not any(character in rendered for character in "╭╮╰╯│")


def test_osc52_sequence_encodes_unicode_and_wraps_tmux() -> None:
    text = "objective ↑ 0.75"
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    assert osc52_clipboard_sequence(text) == f"\x1b]52;c;{payload}\x07"
    assert osc52_clipboard_sequence(text, tmux=True) == (
        f"\x1bPtmux;\x1b\x1b]52;c;{payload}\x07\x1b\\"
    )


def test_osc52_falls_back_to_supplied_output(tmp_path: Path) -> None:
    output = io.StringIO()
    assert copy_text_to_clipboard_osc52(
        "copy me",
        tty_path=tmp_path / "missing" / "tty",
        output=output,
        tmux=False,
    )
    assert output.getvalue().startswith("\x1b]52;c;")


def test_osc52_detects_tmux_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    assert copy_text_to_clipboard_osc52(
        "copy me",
        tty_path=tmp_path / "missing" / "tty",
        output=output,
    )
    assert output.getvalue().startswith("\x1bPtmux;\x1b\x1b]52;c;")


def test_save_panel_copy_overwrites_utf8_fallback(tmp_path: Path) -> None:
    first = save_panel_copy("Quick View", "run-1", "first", tmp_dir=tmp_path)
    second = save_panel_copy("Quick View", "run-1", "zażółć", tmp_dir=tmp_path)
    assert first == second == tmp_path / "panel-quick-view-run-1.txt"
    assert second.read_text(encoding="utf-8") == "zażółć"
