from __future__ import annotations

import re
from dataclasses import dataclass

from trsllava.rsrt.schema import RSRTRecord


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    errors: list[str]


_WS_RE = re.compile(r"\\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", s.strip().lower())


def validate_record(record: RSRTRecord) -> QualityResult:
    errors: list[str] = []
    if len(record.variants) != 5:
        errors.append(f"expected 5 variants, got {len(record.variants)}")

    for i, v in enumerate(record.variants, start=1):
        if not v.one_sentence.strip():
            errors.append(f"variant[{i}].one_sentence empty")
        if not v.feature.strip():
            errors.append(f"variant[{i}].feature empty")
        if not v.caption.strip():
            errors.append(f"variant[{i}].caption empty")

    # Simple duplicate check across variants
    sigs = set()
    for i, v in enumerate(record.variants, start=1):
        sig = _norm(v.one_sentence) + "||" + _norm(v.feature) + "||" + _norm(v.caption)
        if sig in sigs:
            errors.append(f"variant[{i}] duplicate content")
        sigs.add(sig)

    return QualityResult(ok=(len(errors) == 0), errors=errors)

