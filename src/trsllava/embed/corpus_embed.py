from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from trsllava.datasets.captions import load_caption_items


def embed_texts_to_npz(
    *,
    texts: list[str],
    image_ids: list[str],
    out_npz: Path,
    embed_model: str,
    batch_size: int,
    kind: str,
) -> None:
    if len(texts) != len(image_ids):
        raise ValueError("texts and image_ids must have same length")
    client = OpenAI()

    embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({kind})"):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=embed_model, input=batch)
        batch_emb = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(batch_emb, axis=1, keepdims=True) + 1e-12
        embs.append(batch_emb / norms)

    mat = np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), dtype=np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        embeddings=mat,
        image=np.array(image_ids, dtype=object),
        text=np.array(texts, dtype=object),
        embed_model=np.array([embed_model], dtype=object),
        kind=np.array([kind], dtype=object),
    )


def embed_caption_corpus_to_npz(
    captions_path: Path,
    out_npz: Path,
    embed_model: str = "text-embedding-3-small",
    batch_size: int = 256,
) -> None:
    items = load_caption_items(captions_path)
    texts = [it.caption for it in items]
    image_ids = [it.image for it in items]
    embed_texts_to_npz(
        texts=texts,
        image_ids=image_ids,
        out_npz=out_npz,
        embed_model=embed_model,
        batch_size=batch_size,
        kind="captions",
    )

