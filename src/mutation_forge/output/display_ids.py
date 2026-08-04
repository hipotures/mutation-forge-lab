from __future__ import annotations

import re

_PARENT_SLOT = re.compile(r"\bparent-(\d+)-slot-(\d+)\b")
_SLOT = re.compile(r"\bslot-(\d+)\b")


def compact_display_ids(value: object) -> str:
    text = str(value)
    text = _PARENT_SLOT.sub(r"p\1-s\2", text)
    return _SLOT.sub(r"s\1", text)
