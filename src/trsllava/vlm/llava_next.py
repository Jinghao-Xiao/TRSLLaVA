from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from trsllava.io_jsonl import read_jsonl, write_jsonl
from trsllava.paths import resolve_under_root, to_posix_rel
from trsllava.vlm.prompt import image_query_prompt


def _pick_device(device: str) -> str:
    device = device.lower().strip()
    if device != "auto":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def generate_image_query_texts_llava_next(
    *,
    images_jsonl: Path,
    out_jsonl: Path,
    model_path: str,
    device: str = "auto",
    image_root: Path,
    batch_size: int = 4,
    max_images: int | None = None,
) -> None:
    try:
        import torch
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing optional dependencies for LLaVA. Install with: pip install -e '.[local-vlm]'"
        ) from e

    image_root = image_root.resolve()
    device = _pick_device(device)

    processor = LlavaNextProcessor.from_pretrained(model_path, use_fast=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    tokenizer = processor.tokenizer

    # LLaVA-Next chat template prompt
    prompt = (
        "<|im_start|>user\n<image>\n"
        + image_query_prompt()
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

    items = list(read_jsonl(images_jsonl))
    if max_images:
        items = items[:max_images]

    # Load and cache PIL images
    records: list[tuple[Path, Image.Image]] = []
    for rec in items:
        image_str = rec.get("image") or rec.get("image_path")
        if not image_str:
            continue
        abs_path = resolve_under_root(str(image_str), image_root)
        if not abs_path.exists():
            raise FileNotFoundError(f"Image not found: {abs_path}")
        img = Image.open(abs_path).convert("RGB")
        records.append((abs_path, img))

    out: list[dict[str, Any]] = []
    for i in tqdm(range(0, len(records), batch_size), desc="Image→Text (LLaVA-Next)"):
        batch = records[i : i + batch_size]
        paths = [p for p, _ in batch]
        images = [im for _, im in batch]

        inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=256, use_cache=True)
        responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for p, resp in zip(paths, responses):
            out.append({"image": to_posix_rel(p, image_root), "query_text": resp.strip(), "model": model_path})

    write_jsonl(out_jsonl, out)

