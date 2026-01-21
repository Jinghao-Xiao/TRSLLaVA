from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from trsllava.io_jsonl import read_jsonl


def _format_variant_for_embedding(v: dict[str, Any]) -> str:
    one = (v.get("one_sentence") or v.get("one_sentence_caption") or "").strip()
    feat = (v.get("feature") or v.get("feature_analysis") or "").strip()
    cap = (v.get("caption") or "").strip()
    return f"ONE_SENTENCE: {one}\nFEATURE: {feat}\nCAPTION: {cap}".strip()


def _iter_rsrt_variants(rsrt_jsonl: Path):
    for rec in read_jsonl(rsrt_jsonl):
        image = (rec.get("image") or "").strip()
        if not image:
            continue
        if "variants" in rec and isinstance(rec["variants"], list):
            variants = rec["variants"]
        elif "text" in rec and isinstance(rec["text"], dict):
            variants = [rec["text"]]
        else:
            continue
        for j, v in enumerate(variants):
            yield image, int(j), _format_variant_for_embedding(v)


def embed_rsrt_to_npz(
    rsrt_jsonl: Path,
    out_npz: Path,
    embed_model: str = "text-embedding-3-small",
    batch_size: int = 128,
) -> None:
    client = OpenAI()

    images: list[str] = []
    variant_idx: list[int] = []
    texts: list[str] = []
    for image, j, text in _iter_rsrt_variants(rsrt_jsonl):
        images.append(image)
        variant_idx.append(j)
        texts.append(text)

    embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding RSRT"):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=embed_model, input=batch)
        batch_emb = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # normalize for cosine via dot-product
        norms = np.linalg.norm(batch_emb, axis=1, keepdims=True) + 1e-12
        batch_emb = batch_emb / norms
        embs.append(batch_emb)

    mat = np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), dtype=np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        embeddings=mat,
        image=np.array(images, dtype=object),
        variant=np.array(variant_idx, dtype=np.int32),
        embed_model=np.array([embed_model], dtype=object),
    )

