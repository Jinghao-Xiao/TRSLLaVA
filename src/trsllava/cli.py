from __future__ import annotations

import argparse
import json
from pathlib import Path

from trsllava.embed.corpus_embed import embed_caption_corpus_to_npz
from trsllava.embed.openai_embed import embed_rsrt_to_npz
from trsllava.eval.i2t import eval_i2t
from trsllava.eval.t2i import eval_t2i
from trsllava.rsrt.convert import group_rsrt_jsonl
from trsllava.rsrt.generate_openai import generate_rsrt_jsonl
from trsllava.vlm.openai_vlm import generate_image_query_texts_openai
from trsllava.vlm.llava_next import generate_image_query_texts_llava_next


def _path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trsllava")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rsrt = sub.add_parser("rsrt-generate", help="Generate 5× structured captions per image (RSRT-style).")
    p_rsrt.add_argument("--images", required=True, help="Input JSONL. Each line: {'image': 'path/to/img'}")
    p_rsrt.add_argument("--out", required=True, help="Output RSRT JSONL (one line per image).")
    p_rsrt.add_argument("--model", default="gpt-4.1", help="OpenAI model for vision captioning.")
    p_rsrt.add_argument("--max-images", type=int, default=0, help="Limit images (0 = no limit).")
    p_rsrt.add_argument("--image-root", default=".", help="Resolve relative image paths against this root.")

    p_embed = sub.add_parser("embed-rsrt", help="Embed RSRT JSONL and save as NPZ cache.")
    p_embed.add_argument("--rsrt", required=True, help="RSRT JSONL produced by rsrt-generate.")
    p_embed.add_argument("--out", required=True, help="Output .npz embeddings file.")
    p_embed.add_argument("--embed-model", default="text-embedding-3-small", help="OpenAI embedding model.")
    p_embed.add_argument("--batch-size", type=int, default=128)

    p_group = sub.add_parser("rsrt-group", help="Group flat RSRT JSONL into 1-line-per-image format.")
    p_group.add_argument("--in", dest="in_path", required=True, help="Input RSRT JSONL (flat or grouped).")
    p_group.add_argument("--out", required=True, help="Output grouped RSRT JSONL.")
    p_group.add_argument("--image-root", default=".", help="Store image paths relative to this root.")

    p_t2i = sub.add_parser("eval-t2i", help="Evaluate Text→Image retrieval (Recall@K and mR).")
    p_t2i.add_argument("--queries", required=True, help="Query captions JSON (list) or JSONL.")
    p_t2i.add_argument("--rsrt", required=True, help="RSRT JSONL (db).")
    p_t2i.add_argument("--rsrt-emb", required=True, help="RSRT embeddings NPZ (from embed-rsrt).")
    p_t2i.add_argument("--out", required=True, help="Output metrics JSON.")
    p_t2i.add_argument("--k", default="1,5,10", help="Comma-separated ks.")

    p_cap_embed = sub.add_parser("embed-captions", help="Embed a caption corpus (e.g., RSITMD/RSICD captions).")
    p_cap_embed.add_argument("--captions", required=True, help="Captions JSON (list) or JSONL.")
    p_cap_embed.add_argument("--out", required=True, help="Output .npz embeddings file.")
    p_cap_embed.add_argument("--embed-model", default="text-embedding-3-small", help="OpenAI embedding model.")
    p_cap_embed.add_argument("--batch-size", type=int, default=256)

    p_i2t = sub.add_parser("eval-i2t", help="Evaluate Image→Text retrieval from image-query texts.")
    p_i2t.add_argument("--queries", required=True, help="JSONL with {'image':..., 'Description':{'full_response':...}} or {'image':..., 'query_text':...}.")
    p_i2t.add_argument("--db-emb", required=True, help="Caption embeddings NPZ (from embed-captions) or RSRT embeddings NPZ.")
    p_i2t.add_argument("--out", required=True, help="Output metrics JSON.")
    p_i2t.add_argument("--k", default="1,5,10", help="Comma-separated ks.")

    p_iq_openai = sub.add_parser("img2txt-openai", help="Convert images to query texts using OpenAI Vision (training-free).")
    p_iq_openai.add_argument("--images", required=True, help="Input JSONL with {'image':...}.")
    p_iq_openai.add_argument("--out", required=True, help="Output JSONL with {'image':..., 'query_text':...}.")
    p_iq_openai.add_argument("--model", default="gpt-4.1", help="OpenAI vision model.")
    p_iq_openai.add_argument("--image-root", default=".", help="Resolve/store image paths relative to this root.")
    p_iq_openai.add_argument("--max-images", type=int, default=0, help="Limit images (0 = no limit).")

    p_iq_llava = sub.add_parser("img2txt-llava", help="Convert images to query texts using local LLaVA-Next (optional).")
    p_iq_llava.add_argument("--images", required=True, help="Input JSONL with {'image':...}.")
    p_iq_llava.add_argument("--out", required=True, help="Output JSONL with {'image':..., 'query_text':...}.")
    p_iq_llava.add_argument("--model-path", required=True, help="Local path or HF id for LlavaNext model.")
    p_iq_llava.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p_iq_llava.add_argument("--image-root", default=".", help="Resolve/store image paths relative to this root.")
    p_iq_llava.add_argument("--batch-size", type=int, default=4)
    p_iq_llava.add_argument("--max-images", type=int, default=0, help="Limit images (0 = no limit).")

    args = parser.parse_args(argv)

    if args.cmd == "rsrt-generate":
        generate_rsrt_jsonl(
            images_jsonl=_path(args.images),
            out_jsonl=_path(args.out),
            model=args.model,
            max_images=args.max_images if args.max_images and args.max_images > 0 else None,
            image_root=_path(args.image_root),
        )
        return 0

    if args.cmd == "embed-rsrt":
        embed_rsrt_to_npz(
            rsrt_jsonl=_path(args.rsrt),
            out_npz=_path(args.out),
            embed_model=args.embed_model,
            batch_size=args.batch_size,
        )
        return 0

    if args.cmd == "rsrt-group":
        group_rsrt_jsonl(
            in_jsonl=_path(args.in_path),
            out_jsonl=_path(args.out),
            image_root=_path(args.image_root),
        )
        return 0

    if args.cmd == "eval-t2i":
        ks = [int(x) for x in str(args.k).split(",") if x.strip()]
        metrics = eval_t2i(
            queries_path=_path(args.queries),
            rsrt_jsonl=_path(args.rsrt),
            rsrt_embeddings_npz=_path(args.rsrt_emb),
            ks=ks,
        )
        out_path = _path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return 0

    if args.cmd == "embed-captions":
        embed_caption_corpus_to_npz(
            captions_path=_path(args.captions),
            out_npz=_path(args.out),
            embed_model=args.embed_model,
            batch_size=args.batch_size,
        )
        return 0

    if args.cmd == "eval-i2t":
        ks = [int(x) for x in str(args.k).split(",") if x.strip()]
        metrics = eval_i2t(
            queries_jsonl=_path(args.queries),
            db_embeddings_npz=_path(args.db_emb),
            ks=ks,
        )
        out_path = _path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return 0

    if args.cmd == "img2txt-openai":
        generate_image_query_texts_openai(
            images_jsonl=_path(args.images),
            out_jsonl=_path(args.out),
            model=args.model,
            image_root=_path(args.image_root),
            max_images=args.max_images if args.max_images and args.max_images > 0 else None,
        )
        return 0

    if args.cmd == "img2txt-llava":
        generate_image_query_texts_llava_next(
            images_jsonl=_path(args.images),
            out_jsonl=_path(args.out),
            model_path=args.model_path,
            device=args.device,
            image_root=_path(args.image_root),
            batch_size=args.batch_size,
            max_images=args.max_images if args.max_images and args.max_images > 0 else None,
        )
        return 0

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
