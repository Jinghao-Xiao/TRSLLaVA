from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from trsllava.datasets.captions import group_captions_by_image
from trsllava.datasets.captions import load_caption_items
from trsllava.eval.metrics import compute_recall_at_k
from trsllava.io_jsonl import read_jsonl


def _load_image_query_texts(queries_jsonl: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for rec in read_jsonl(queries_jsonl):
        image = (rec.get("image") or rec.get("image_path") or "").strip()
        if not image:
            continue
        txt = (rec.get("query_text") or rec.get("Description", {}).get("full_response") or "").strip()
        if not txt:
            continue
        out.append((image, txt))
    return out


def _load_db_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    image_ids = [str(x).strip() for x in data["image"].astype(object).tolist()]
    texts = [str(x) for x in data["text"].astype(object).tolist()] if "text" in data else None
    embed_model = str(data["embed_model"][0]) if "embed_model" in data else "text-embedding-3-small"
    kind = str(data["kind"][0]) if "kind" in data else "unknown"
    return emb, image_ids, texts, embed_model, kind


def _embed_queries_openai(texts: list[str], model: str, batch_size: int = 128) -> np.ndarray:
    client = OpenAI()
    out: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding image-query texts"):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        em = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(em, axis=1, keepdims=True) + 1e-12
        out.append(em / norms)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def eval_i2t(
    *,
    queries_jsonl: Path,
    db_embeddings_npz: Path,
    ks: list[int] | None = None,
) -> dict[str, Any]:
    """
    Image→Text retrieval (text-to-text under the hood):
    - Query: image converted to text by a frozen VLM (provided as JSONL).
    - DB: caption texts (or other text corpus) with known image ids.
    - Hit if any caption belonging to the query image appears in top-k.
    """
    ks = sorted(set(ks or [1, 5, 10]))

    # Load query texts
    q = _load_image_query_texts(queries_jsonl)
    q_images = [im for im, _ in q]
    q_texts = [t for _, t in q]

    # Load DB embeddings
    db_emb, db_image_ids, _, embed_model, kind = _load_db_npz(db_embeddings_npz)

    # Build ground-truth map from DB itself (image -> list of row indices)
    rows_by_image: dict[str, list[int]] = {}
    for i, im in enumerate(db_image_ids):
        rows_by_image.setdefault(im, []).append(i)

    # Embed queries using same embedding model as DB
    q_emb = _embed_queries_openai(q_texts, model=embed_model)

    hits = {k: 0 for k in ks}
    for i in tqdm(range(len(q)), desc="Scoring"):
        sims = q_emb[i] @ db_emb.T
        ranked = np.argsort(sims)[::-1]
        gt = str(q_images[i]).strip()
        gt_basename = Path(gt).name
        # ground-truth rows: match by exact image id if possible, else basename match
        gt_rows = rows_by_image.get(gt)
        if gt_rows is None:
            # fallback by basename
            matching = [idx for idx, im in enumerate(db_image_ids) if Path(im).name == gt_basename]
            gt_rows = matching

        gt_row_set = set(gt_rows or [])
        for k in ks:
            topk = ranked[:k]
            if any(int(r) in gt_row_set for r in topk):
                hits[k] += 1

    metrics = compute_recall_at_k(hits, total=len(q))
    return {
        "total_queries": len(q),
        "recall": metrics.recall_at,
        "mean_recall": metrics.mean_recall,
        "ks": ks,
        "db_embeddings": str(db_embeddings_npz),
        "queries": str(queries_jsonl),
        "db_kind": kind,
    }

