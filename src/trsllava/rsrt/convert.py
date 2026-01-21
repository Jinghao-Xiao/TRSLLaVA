from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from trsllava.io_jsonl import read_jsonl, write_jsonl
from trsllava.paths import resolve_under_root, to_posix_rel


def _as_variant(rec: dict[str, Any]) -> dict[str, Any] | None:
    if "variants" in rec and isinstance(rec["variants"], list):
        # already grouped record, handled elsewhere
        return None
    if "text" in rec and isinstance(rec["text"], dict):
        return rec["text"]
    # tolerate legacy keys
    if "one_sentence_caption" in rec or "feature" in rec or "caption" in rec:
        return {
            "one_sentence": rec.get("one_sentence_caption") or rec.get("one_sentence") or "",
            "feature": rec.get("feature") or rec.get("feature_analysis") or "",
            "caption": rec.get("caption") or "",
        }
    return None


def group_rsrt_jsonl(in_jsonl: Path, out_jsonl: Path, image_root: Path) -> None:
    image_root = image_root.resolve()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []

    for rec in read_jsonl(in_jsonl):
        image_str = (rec.get("image") or rec.get("image_path") or "").strip()
        if not image_str:
            continue
        if "variants" in rec and isinstance(rec["variants"], list):
            # normalize image to relative under root if possible
            abs_path = resolve_under_root(image_str, image_root)
            rec["image"] = to_posix_rel(abs_path, image_root)
            passthrough.append(rec)
            continue

        v = _as_variant(rec)
        if v is None:
            continue
        abs_path = resolve_under_root(image_str, image_root)
        key = to_posix_rel(abs_path, image_root)
        grouped[key].append(
            {
                "one_sentence": (v.get("one_sentence") or v.get("one_sentence_caption") or "").strip(),
                "feature": (v.get("feature") or v.get("feature_analysis") or "").strip(),
                "caption": (v.get("caption") or "").strip(),
            }
        )

    out: list[dict[str, Any]] = []
    out.extend(passthrough)
    for image, variants in grouped.items():
        # keep the first 5 if more exist (common when multiple runs appended)
        if len(variants) >= 5:
            variants = variants[:5]
        out.append({"image": image, "variants": variants})

    write_jsonl(out_jsonl, out)

