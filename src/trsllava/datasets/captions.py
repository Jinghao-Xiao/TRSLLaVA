from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from trsllava.io_jsonl import read_jsonl


@dataclass(frozen=True)
class CaptionItem:
    image: str
    caption: str


def load_caption_items(path: Path) -> list[CaptionItem]:
    """
    Supports:
    - RSITMD/RSICD-like JSON list items with keys: caption, filename/image/image_path
    - JSONL lines with keys: image + caption (or text.caption)
    """
    if path.suffix.lower() == ".json":
        items = json.loads(path.read_text(encoding="utf-8"))
    else:
        items = list(read_jsonl(path))

    out: list[CaptionItem] = []
    for item in items:
        caption = (item.get("caption") or item.get("text", {}).get("caption") or "").strip()
        if not caption:
            continue
        image = (item.get("image") or item.get("image_path") or item.get("filename") or "").strip()
        if not image:
            continue
        out.append(CaptionItem(image=image, caption=caption))
    return out


def group_captions_by_image(items: Iterable[CaptionItem]) -> dict[str, list[str]]:
    m: dict[str, list[str]] = {}
    for it in items:
        m.setdefault(it.image, []).append(it.caption)
    return m

