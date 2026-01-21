# TRSLLAVA (export)

This folder is a clean, GitHub-ready export aligned with the paper methodology:
training-free, text-to-text retrieval for remote sensing images using a rich-text corpus (RSRT-style) and a single text embedding space.

## Method (paper-aligned)
- Build a rich-text corpus: for each image, generate **five** structured description variants (one-sentence summary, directional/relational features, detailed paragraph).
- Apply lightweight quality checks (format, empties, duplicates).
- Embed all texts into a shared space (default: OpenAI `text-embedding-3-small`).
- Retrieval is text-to-text cosine similarity; for image-level scoring, use the **max similarity across 5 variants**.
- For image queries (I2T/T2I starting from image), first convert image → text using a frozen VLM (pluggable; OpenAI Vision or local LLaVA adapter).

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Configure secrets (no leakage)
```bash
export OPENAI_API_KEY="YOUR_KEY"
```
Or copy `export/.env.example` → `export/.env` and load it yourself.

## Quickstart (your own data)
1) Prepare an image list JSONL (relative paths recommended):
```jsonl
{"image":"data/rsitmd/images/pond_2994.tif"}
{"image":"data/rsitmd/images/storagetanks_4459.tif"}
```

2) Generate 5× structured captions per image (RSRT-style):
```bash
trsllava rsrt-generate --images data/images.jsonl --out outputs/rsrt.jsonl --image-root .
```
If your images live outside this repo, set `--image-root` to your dataset root so paths stored in JSONL stay relative.

3) Build embeddings for the RSRT corpus:
```bash
trsllava embed-rsrt --rsrt outputs/rsrt.jsonl --out outputs/rsrt_embeddings.npz
```

4) Evaluate Text→Image (T2I) using your query captions JSON (RSITMD/RSICD-like):
```bash
trsllava eval-t2i --queries data/rsitmd_captions.json --rsrt outputs/rsrt.jsonl --rsrt-emb outputs/rsrt_embeddings.npz --out outputs/t2i_metrics.json
```

## Image→Text (I2T) evaluation (optional)
1) Convert images to query texts (paper uses a frozen VLM such as LLaVA):
```bash
trsllava img2txt-llava --images examples/images.jsonl --out outputs/img_queries.jsonl --model-path /path/to/llava-next --image-root .
```
Or use OpenAI Vision:
```bash
trsllava img2txt-openai --images examples/images.jsonl --out outputs/img_queries.jsonl --image-root .
```

2) Embed the candidate caption corpus:
```bash
trsllava embed-captions --captions data/rsitmd_captions.json --out outputs/caption_embeddings.npz
```

3) Evaluate:
```bash
trsllava eval-i2t --queries outputs/img_queries.jsonl --db-emb outputs/caption_embeddings.npz --out outputs/i2t_metrics.json
```

## Converting existing RSRT JSONL
If you already have a flat JSONL (multiple lines per image), group it into 1-line-per-image:
```bash
trsllava rsrt-group --in your_flat_rsrt.jsonl --out outputs/rsrt_grouped.jsonl --image-root /your/dataset/root
```

## Data layout (recommended)
Keep datasets outside Git, or put them under `export/data/` (gitignored):
- `export/data/rsitmd/images/`
- `export/data/rsicd/images/`
- `export/data/...`

## Notes
- This export does **not** include datasets, model weights, or any API keys.
- If your existing project contains hardcoded keys, rotate/revoke them before pushing anything public.
