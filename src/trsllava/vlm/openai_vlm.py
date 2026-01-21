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
from trsllava.vlm.prompt import image_query_prompt


def _image_to_jpeg_bytes(image_path: Path) -> bytes:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def _data_url_from_image(image_path: Path) -> str:
    jpeg_bytes = _image_to_jpeg_bytes(image_path)
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _call_openai_image_to_text(client: OpenAI, model: str, image_path: Path, max_retries: int = 3) -> str:
    data_url = _data_url_from_image(image_path)
    prompt = image_query_prompt()

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
                temperature=0.2,
                max_tokens=400,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            sleep_s = min(20.0, (2 ** (attempt - 1))) + float(np.random.rand()) * 0.5
            time.sleep(sleep_s)

    raise RuntimeError(f"OpenAI image->text failed after {max_retries} retries: {last_err}")


def generate_image_query_texts_openai(
    *,
    images_jsonl: Path,
    out_jsonl: Path,
    model: str = "gpt-4.1",
    image_root: Path,
    max_images: int | None = None,
) -> None:
    image_root = image_root.resolve()
    client = OpenAI()

    items = list(read_jsonl(images_jsonl))
    if max_images:
        items = items[:max_images]

    out: list[dict[str, Any]] = []
    for rec in tqdm(items, desc="Image→Text (OpenAI)"):
        image_str = rec.get("image") or rec.get("image_path")
        if not image_str:
            continue
        abs_path = resolve_under_root(str(image_str), image_root)
        if not abs_path.exists():
            raise FileNotFoundError(f"Image not found: {abs_path}")
        txt = _call_openai_image_to_text(client, model=model, image_path=abs_path)
        out.append({"image": to_posix_rel(abs_path, image_root), "query_text": txt, "model": model})

    write_jsonl(out_jsonl, out)

