from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from trsllava.io_jsonl import read_jsonl, write_jsonl
from trsllava.paths import resolve_under_root, to_posix_rel
from trsllava.rsrt.quality import validate_record
from trsllava.rsrt.schema import RSRTRecord, RSRTVariant


def _image_to_jpeg_bytes(image_path: Path) -> bytes:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        out = Path(image_path.name).with_suffix(".jpg")
        # Encode to JPEG in-memory
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def _data_url_from_image(image_path: Path) -> str:
    jpeg_bytes = _image_to_jpeg_bytes(image_path)
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _prompt_json() -> str:
    return (
        "You are given a satellite/aerial image. Analyze ONLY what is visible.\n"
        "Assume top=N, bottom=S, left=W, right=E.\n"
        "Do NOT give exact percentages; use qualitative size terms.\n"
        "Return STRICT JSON with the following schema:\n"
        "{\n"
        '  "variants": [\n'
        "    {\n"
        '      "one_sentence": \"...\",  // one sentence with total-subtotal structure\n'
        '      "feature": \"...\",       // list of visible features with direction/position\n'
        '      "caption": \"...\"        // detailed paragraph, must be non-empty\n'
        "    },\n"
        "    ... exactly 5 items ...\n"
        "  ]\n"
        "}\n"
    )


def _call_openai_vision_json(client: OpenAI, model: str, image_path: Path, max_retries: int = 3) -> dict[str, Any]:
    data_url = _data_url_from_image(image_path)
    prompt = _prompt_json()

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                        ],
                    }
                ],
                temperature=0.3,
                max_tokens=1800,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            return json.loads(content)
        except Exception as e:
            last_err = e
            # exponential backoff with jitter
            sleep_s = min(20.0, (2 ** (attempt - 1))) + float(np.random.rand()) * 0.5
            time.sleep(sleep_s)

    raise RuntimeError(f"OpenAI call failed after {max_retries} retries: {last_err}")


def generate_rsrt_jsonl(
    images_jsonl: Path,
    out_jsonl: Path,
    model: str = "gpt-4.1",
    max_images: int | None = None,
    image_root: Path | None = None,
) -> None:
    image_root = (image_root or Path(".")).resolve()
    client = OpenAI()

    records_out: list[dict[str, Any]] = []
    it = list(read_jsonl(images_jsonl))
    if max_images:
        it = it[:max_images]

    for rec in tqdm(it, desc="RSRT captioning"):
        image_str = rec.get("image") or rec.get("image_path")
        if not image_str:
            continue
        abs_path = resolve_under_root(str(image_str), image_root)
        if not abs_path.exists():
            raise FileNotFoundError(f"Image not found: {abs_path}")

        raw = _call_openai_vision_json(client, model=model, image_path=abs_path)
        variants_raw = raw.get("variants", [])
        variants = [RSRTVariant(**v) for v in variants_raw]
        rsrt = RSRTRecord(image=to_posix_rel(abs_path, image_root), variants=variants, model=model)

        q = validate_record(rsrt)
        if not q.ok:
            raise ValueError(f"Quality check failed for {rsrt.image}: {q.errors}")

        records_out.append(rsrt.model_dump())

    write_jsonl(out_jsonl, records_out)

