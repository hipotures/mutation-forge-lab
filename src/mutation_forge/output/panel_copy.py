from __future__ import annotations

import base64
import os
import sys
from io import StringIO
from pathlib import Path
from typing import TextIO

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.text import Text


def sanitize_panel_copy_name(value: str) -> str:
    sanitized = "".join(
        character
        if character.isascii()
        and (character.isalnum() or character in "_.-")
        else "-"
        for character in value.strip()
    )
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized.strip("-") or "panel"


def panel_copy_path(
    panel_name: str,
    run_id: str,
    *,
    tmp_dir: Path = Path("/tmp"),
) -> Path:
    panel = sanitize_panel_copy_name(panel_name).lower()
    run = sanitize_panel_copy_name(run_id)
    return tmp_dir / f"panel-{panel}-{run}.txt"


def render_panel_copy_text(
    title: str,
    renderable: RenderableType,
    *,
    width: int = 100,
) -> str:
    console = Console(
        file=StringIO(),
        force_terminal=False,
        color_system=None,
        width=max(20, width),
        record=True,
    )
    content = renderable.renderable if isinstance(renderable, Panel) else renderable
    console.print(content)
    body = Text.from_ansi(console.export_text(styles=False)).plain.expandtabs(4).rstrip()
    return f"# {title.strip()}\n\n{body}\n" if body else f"# {title.strip()}\n"


def osc52_clipboard_sequence(text: str, *, tmux: bool = False) -> str:
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    sequence = f"\x1b]52;c;{payload}\x07"
    return f"\x1bPtmux;\x1b{sequence}\x1b\\" if tmux else sequence


def copy_text_to_clipboard_osc52(
    text: str,
    *,
    tty_path: Path = Path("/dev/tty"),
    output: TextIO | None = None,
    tmux: bool | None = None,
) -> bool:
    sequence = osc52_clipboard_sequence(
        text,
        tmux=("TMUX" in os.environ) if tmux is None else tmux,
    )
    try:
        with tty_path.open("w", encoding="utf-8") as stream:
            stream.write(sequence)
            stream.flush()
        return True
    except OSError:
        fallback_output = output or sys.stdout
        try:
            fallback_output.write(sequence)
            fallback_output.flush()
            return True
        except (OSError, ValueError):
            return False


def save_panel_copy(
    panel_name: str,
    run_id: str,
    text: str,
    *,
    tmp_dir: Path = Path("/tmp"),
) -> Path:
    path = panel_copy_path(panel_name, run_id, tmp_dir=tmp_dir)
    path.write_text(text, encoding="utf-8")
    return path
