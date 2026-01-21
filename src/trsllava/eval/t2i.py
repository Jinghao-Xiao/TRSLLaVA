from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from trsllava.eval.metrics import compute_recall_at_k
from trsllava.io_jsonl import read_jsonl


def _load_queries(path: Path) -> list[tuple[str, str]]:
    """
    Returns list of (gt_image, query_text).
    Supports:
    - JSON list with keys: image/image_path/filename + caption
    - JSONL lines with keys: image + caption
    """
    if path.suffix.lower() == ".json":
        items = json.loads(path.read_text(encoding="utf-8"))
    else:
        items = list(read_jsonl(path))

    out: list[tuple[str, str]] = []
    for item in items:
        caption = (item.get("caption") or item.get("text", {}).get("caption") or "").strip()
        if not caption:
            continue
        gt = (item.get("image") or item.get("image_path") or item.get("filename") or "").strip()
        if not gt:
            continue
        out.append((gt, caption))
    return out


def _load_rsrt_embeddings(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["embeddings"].astype(np.float32),
        data["image"].astype(object).tolist(),
    )


def _embed_queries_openai(texts: list[str], model: str, batch_size: int = 128) -> np.ndarray:
    client = OpenAI()
    out: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding queries"):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        em = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(em, axis=1, keepdims=True) + 1e-12
        out.append(em / norms)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def eval_t2i(
    queries_path: Path,
    rsrt_jsonl: Path,
    rsrt_embeddings_npz: Path,
    ks: list[int] | None = None,
    query_embed_model: str | None = None,
) -> dict[str, Any]:
    """
    Text→Image retrieval:
    - DB: RSRT rich-text variants per image
    - Score image = max similarity across its variants
    """
    ks = sorted(set(ks or [1, 5, 10]))

    # Load query captions
    queries = _load_queries(queries_path)
    gt_images = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    # Load RSRT embeddings and image mapping for each variant row
    db_emb, db_image_for_row = _load_rsrt_embeddings(rsrt_embeddings_npz)
    db_image_for_row = [str(x).strip() for x in db_image_for_row]
    unique_images = sorted(set(db_image_for_row))
    image_to_idx = {im: i for i, im in enumerate(unique_images)}
    row_to_image_idx = np.array([image_to_idx[im] for im in db_image_for_row], dtype=np.int32)

    # Embed queries (OpenAI by default, consistent with paper encoder)
    embed_model = query_embed_model or str(np.load(rsrt_embeddings_npz, allow_pickle=True)["embed_model"][0])
    q_emb = _embed_queries_openai(texts, model=embed_model)

    hits = {k: 0 for k in ks}
    for i in tqdm(range(len(queries)), desc="Scoring"):
        sims = q_emb[i] @ db_emb.T  # (rows,)
        # aggregate by image: max over its 5 variants
        scores = np.full((len(unique_images),), -1e9, dtype=np.float32)
        np.maximum.at(scores, row_to_image_idx, sims)
        ranked = np.argsort(scores)[::-1]
        # ground truth matching: support gt as filename or path suffix
        gt = str(gt_images[i]).strip()
        # best-effort: match by exact if possible, else by basename
        gt_basename = Path(gt).name
        ranked_images = [unique_images[j] for j in ranked[: max(ks)]]
        ranked_basenames = [Path(p).name for p in ranked_images]
        for k in ks:
            topk_images = ranked_images[:k]
            topk_basenames = ranked_basenames[:k]
            if gt in topk_images or gt_basename in topk_basenames:
                hits[k] += 1

    metrics = compute_recall_at_k(hits, total=len(queries))
    return {
        "total_queries": len(queries),
        "recall": metrics.recall_at,
        "mean_recall": metrics.mean_recall,
        "ks": ks,
        "rsrt_embeddings": str(rsrt_embeddings_npz),
        "queries": str(queries_path),
        "rsrt": str(rsrt_jsonl),
    }

