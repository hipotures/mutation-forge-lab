"""Render or verify the checked-in Stage 3 prompt templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mutation_forge.stage3.prompts import (
    load_prompt_bundle,
    render_request_prompt,
    render_system_prompt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = {
        Path("prompts/ranker_v1_system.md"): render_system_prompt() + "\n",
        Path("prompts/ranker_v1_request.md"): render_request_prompt() + "\n",
    }
    bundle = load_prompt_bundle()
    for brief_path in sorted(Path("configs/stage3-slots").glob("slot-*.json")):
        brief = json.loads(brief_path.read_text(encoding="utf-8"))
        rendered[Path("prompts/stage3-slots") / f"{brief['slot_id']}.md"] = (
            bundle.render_slot_request(
                brief["slot_id"],
                brief["brief"],
                generation_mode=brief["generation_mode"],
                focus=brief["focus"],
            )
            + "\n"
        )
    drifted = [
        path
        for path, content in rendered.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if args.check:
        if drifted:
            parser.error(
                "checked-in prompt drift: " + ", ".join(str(path) for path in drifted)
            )
        return 0
    for path, content in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
