# Redrob — Intelligent Candidate Discovery & Ranking

Two-phase system for the Redrob hackathon challenge.
Ranks 100,000 candidates for a Senior AI Engineer role using hybrid BM25 + dense retrieval,
structured JD-fit adjustments, and a behavioural availability multiplier.

## Reproduce

```bash
# Step 1 — offline, slow (CPU, ~35 min, needs network once to download embedding model)
python precompute.py

# Step 2 — fast ranker (CPU, no network, ≤5 min)
python rank.py
```

Output: `submission.csv` — 100 ranked candidates.

Validate:
```bash
python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
```

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+. No GPU required. Peak RAM < 4 GB during ranking.

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
