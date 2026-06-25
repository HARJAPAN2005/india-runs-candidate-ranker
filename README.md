# Redrob — Intelligent Candidate Discovery & Ranking

Two-phase system for the Redrob hackathon challenge.
Ranks 100,000 candidates for a Senior AI Engineer role using hybrid BM25 + dense retrieval,
structured JD-fit adjustments, and a behavioural availability multiplier.

## Reproduce

### 1. Place the dataset

The challenge dataset is not committed to this repo (487 MB). Download it from the
Redrob challenge portal and place it at:

```
India_runs_data_and_ai_challenge/candidates.jsonl
```

The directory and file name must match exactly — `precompute.py` reads from that path.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10+. No GPU required.

Measured peak RAM: `rank.py` ~1 GB · `precompute.py` ~1.5 GB (embedding step)

### 3. Build artifacts (run once, ~35 min on CPU)

```bash
python precompute.py
```

Downloads `all-MiniLM-L6-v2` from HuggingFace on first run (86 MB, needs network).
Saves everything to `artifacts/` — subsequent runs skip completed steps automatically.

### 4. Rank candidates (≤5 min, no network)

```bash
python rank.py
```

Outputs `submission.csv` — 100 ranked candidates.

### 5. Validate

```bash
python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
```

Expected output: `Submission is valid.`

## Requirements

Python 3.10+. No GPU required.

Measured peak RAM:
- `rank.py` — **~1 GB** (model weights + 100 K × 384 embeddings + BM25 index)
- `precompute.py` — **~1.5 GB** peak during the embedding step (sentence-transformer
  runtime buffers + growing 100 K × 384 output array); ~0.8 GB for feature extraction
  and BM25 indexing

## How it works

### precompute.py (run once)
- Parses `candidates.jsonl`, extracts 46 structured features per candidate
- Builds a `bm25s` index over career-history text
- Encodes all 100K career docs with `sentence-transformers/all-MiniLM-L6-v2`
- Saves model to `artifacts/model/` so rank.py runs fully offline

### rank.py (the submission step)
- Loads all artifacts from `artifacts/` — **zero network calls** (`TRANSFORMERS_OFFLINE=1`, `local_files_only=True`)
- Hybrid retrieval: `0.35 × BM25 + 0.65 × dense cosine`, rank-percentile normalised
- Structural JD-fit adjustments (dominant: tiered recsys evidence score +0.22, product ratio +0.18)
- Availability multiplier [0.55, 1.10] from 23 behavioural signals
- Honeypot gate: 4 consistency checks; flagged candidates score 0.0
- Tie-break: ascending `candidate_id`

## Artifacts (generated, not committed)

```
artifacts/
  candidate_features.parquet   # 46 features × 100K rows
  candidate_ids.npy
  career_embeddings.npy        # (100000, 384) float32
  bm25_index/
  model/                       # local copy of all-MiniLM-L6-v2
```
