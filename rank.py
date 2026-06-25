#!/usr/bin/env python3
"""
rank.py — fast ranking step for the Redrob candidate ranker.

Loads precomputed artifacts, scores all 100K candidates via hybrid BM25 + dense
retrieval, applies structured JD-fit adjustments, an availability multiplier, and a
honeypot gate, then writes the top-100 CSV.

Constraints: ≤5 min wall clock, ≤16 GB RAM, CPU only, NO network.

Usage:
    python rank.py [--artifacts DIR] [--candidates PATH] [--out PATH] [--top-k N]
"""

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import bm25s
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Safety: block any accidental network calls during ranking
# ─────────────────────────────────────────────────────────────────────────────
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

TODAY = date(2026, 6, 25)

# ─────────────────────────────────────────────────────────────────────────────
# JD query texts
# ─────────────────────────────────────────────────────────────────────────────

# Rich semantic query for dense embedding (includes the JD's "ideal candidate" paragraph)
JD_EMBED_QUERY = """
Senior AI Engineer at Redrob AI, building the intelligence layer of a talent platform:
ranking, retrieval, and matching systems that decide what recruiters see.

Ideal candidate: 6–8 years total experience, 4–5 of which are in applied ML/AI roles at
product companies (not pure services). Has shipped at least one end-to-end ranking, search,
or recommendation system to real users at meaningful scale. Has strong opinions about
retrieval (hybrid vs dense), evaluation (offline vs online), and LLM integration (when to
fine-tune vs prompt) — and can defend them with reference to systems they actually built.
Located in or willing to relocate to Noida or Pune. Active on the Redrob platform.

Required: Production experience with embeddings-based retrieval — sentence-transformers,
BGE, E5, or similar — deployed to real users, including handling embedding drift and
retrieval-quality regression. Production experience with vector databases or hybrid search:
Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. Hands-on evaluation
framework design for ranking systems: NDCG, MRR, MAP, offline-to-online correlation, A/B
test interpretation. Strong Python. Code quality matters.

Bonus: LLM fine-tuning (LoRA, QLoRA), learning-to-rank (XGBoost-based or neural),
prior HR-tech or marketplace product experience, open-source ML contributions.
""".strip()

# Keyword-focused query for BM25 (no stop-words, no prose)
JD_BM25_QUERY = (
    "ranking recommendation search engine information retrieval embeddings vector database "
    "FAISS Elasticsearch Qdrant Milvus Weaviate Pinecone OpenSearch BM25 hybrid retrieval "
    "dense retrieval NDCG MRR MAP evaluation framework A/B testing production deployed "
    "applied ML engineer product company sentence transformers BGE E5 fine-tuning LoRA "
    "Python real users shipped system pipeline inference serving learning-to-rank XGBoost "
    "embedding drift index refresh retrieval quality recall precision A/B interleaving"
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────
def _percentile_normalize(scores: np.ndarray) -> np.ndarray:
    """Rank-based linear normalization: lowest score -> 0.0, highest -> 1.0."""
    n = len(scores)
    if n <= 1:
        return np.zeros(n)
    # argsort twice gives 0-based ascending rank (handles ties by arbitrary order)
    ranks = np.argsort(np.argsort(scores))
    return ranks / (n - 1)


def _yoe_fit(yoe: float) -> float:
    """Map years-of-experience to a [0, 1] fit score. Sweet spot 5–9 yrs."""
    if yoe <= 0:
        return 0.0
    if yoe < 3:
        return yoe / 3 * 0.4          # 0 -> 0.40  (too junior)
    if yoe < 5:
        return 0.4 + (yoe - 3) / 2 * 0.4   # 3 -> 0.80
    if yoe <= 9:
        return 1.0                     # sweet spot
    if yoe <= 12:
        return 1.0 - (yoe - 9) / 3 * 0.1   # 9–12: slight decay
    return 0.85                        # very senior, possible over-qualification


def _availability_multiplier(row: pd.Series) -> float:
    """
    Compute a [0.55, 1.10] multiplier encoding how hireable this candidate is
    right now.  Baseline 0.70 per approved spec change #3.
    """
    avail = 0.70

    # Activity recency
    d = int(row["days_since_active"])
    if d <= 30:
        avail += 0.15
    elif d <= 90:
        avail += 0.07
    elif d > 365:
        avail -= 0.15

    # Explicitly open to work
    if row["open_to_work"]:
        avail += 0.10

    # Responsiveness to recruiters (0 -> +0.10)
    avail += 0.10 * float(row["response_rate"])

    # Interview reliability (0 -> +0.07)
    avail += 0.07 * float(row["interview_completion_rate"])

    # Notice period
    nd = int(row["notice_days"])
    if nd <= 30:
        avail += 0.07
    elif nd > 90:
        avail -= 0.05

    # Verified contact info (signals real candidate, not placeholder)
    if row["verified_both"]:
        avail += 0.04

    # Profile completeness proxy
    if float(row["profile_completeness"]) > 0.80:
        avail += 0.02

    return float(np.clip(avail, 0.55, 1.10))


def _structural_adjustments(row: pd.Series) -> float:
    """
    Additive JD-fit adjustments in roughly [-0.85, +0.83].
    Applied AFTER percentile normalization of retrieval scores.
    """
    adj = 0.0

    # Positive signals ───────────────────────────────────────────────────────
    adj += 0.10 * _yoe_fit(float(row["yoe"]))                 # [0, 0.10]

    prod_ratio = float(row["product_ratio"])
    adj += 0.18 * prod_ratio                                   # [0, 0.18]
    if row["recent_role_product"]:
        adj += 0.05                                            # recency bonus

    adj += 0.22 * float(row["recsys_evidence_score"])          # [0, 0.22]  biggest signal
    if row["vector_db_evidence"]:
        adj += 0.10
    if row["embedding_evidence"]:
        adj += 0.08
    if row["eval_framework_evidence"]:
        adj += 0.07
    if row["deployed_evidence"]:
        adj += 0.05

    gh = row["github_score"]
    if gh is not None and float(gh) >= 20:
        adj += 0.05                                            # OSS activity

    if row["edu_field_relevant"] and int(row["best_edu_tier"]) <= 2:
        adj += 0.03

    # Hard JD disqualifiers ──────────────────────────────────────────────────
    if row["services_only_career"]:
        adj -= 0.60    # "only at consulting firms their entire career"
    if row["is_research_only"]:
        adj -= 0.55    # "pure research without production deployment"
    if row["cv_speech_without_nlp"]:
        adj -= 0.25    # "CV/speech/robotics without NLP/IR"
    if row["langchain_only_recent"]:
        adj -= 0.20    # "under 12mo LangChain only"
    if row["is_job_hopper"]:
        adj -= 0.15    # title-chaser anti-signal
    if not row["has_python"]:
        adj -= 0.08    # Python is explicitly required

    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning generation
# ─────────────────────────────────────────────────────────────────────────────
_RECSYS_LABEL_MAP = {
    "recommendation":    "recommendation systems",
    "ranking":           "ranking systems",
    "search_system":     "search system",
    "ir":                "information retrieval",
    "hybrid_retrieval":  "hybrid retrieval",
    "dense_retrieval":   "dense retrieval",
    "vector_search":     "vector search",
    "search_infra":      "search infrastructure (BM25/ES)",
    "vector_db":         "vector databases (FAISS/Qdrant/etc.)",
    "eval_metrics":      "ranking eval (NDCG/MRR)",
    "ab_testing":        "A/B testing",
    "collab_filter":     "collaborative filtering",
    "ltr":               "learning-to-rank",
    "retrieval_quality": "retrieval quality engineering",
    "ml_serving":        "ML serving/feature stores",
    "deployed_ml":       "deployed ML in production",
}


def _generate_reasoning(row: pd.Series, rank: int) -> str:
    """
    Produce a 1–2 sentence reasoning string grounded in real profile facts.
    Varies structure based on what's most notable about this candidate.
    Never claims facts not present in the features.
    """
    title   = str(row["current_title"] or "").strip() or "candidate"
    company = str(row["current_company"] or "").strip()
    yoe     = float(row["yoe"])
    prod_r  = float(row["product_ratio"])
    recsys  = float(row["recsys_evidence_score"])
    terms   = [_RECSYS_LABEL_MAP.get(t, t)
               for t in str(row["recsys_terms_str"]).split(",")
               if t and t in _RECSYS_LABEL_MAP]
    notable = [c for c in str(row["notable_companies"]).split(",") if c]
    dsa     = int(row["days_since_active"])
    rr      = float(row["response_rate"])
    nd      = int(row["notice_days"])
    otw     = bool(row["open_to_work"])
    svc     = bool(row["services_only_career"])
    vec     = bool(row["vector_db_evidence"])
    emb     = bool(row["embedding_evidence"])
    evl     = bool(row["eval_framework_evidence"])

    parts: list[str] = []

    # ── Lead sentence: what makes this candidate notable (or not) ───────────
    if recsys >= 0.5 and not svc:
        tech_bits = terms[:2] if terms else []
        co_str    = f" (including {', '.join(notable[:2])})" if notable else ""
        if tech_bits:
            parts.append(
                f"{yoe:.0f}-yr career{co_str} shows direct evidence of "
                f"{' and '.join(tech_bits)} in career descriptions."
            )
        else:
            parts.append(
                f"{yoe:.0f}-yr career{co_str} shows strong retrieval/ranking evidence "
                f"in career descriptions."
            )
    elif prod_r >= 0.7 and not svc:
        co_str = f", most recently at {company}" if company else ""
        parts.append(
            f"{yoe:.0f} yrs primarily at product companies{co_str}; "
            f"product-company ratio {prod_r:.0%}."
        )
    elif svc:
        co_str = f" at {', '.join(notable[:2])}" if notable else ""
        parts.append(
            f"{title} with {yoe:.0f} yrs{co_str}; "
            f"career is services-only — does not match the JD's product-company requirement."
        )
    else:
        co_str = f" at {company}" if company else ""
        parts.append(f"{title}{co_str}, {yoe:.0f} yrs experience.")

    # ── Second sentence: tech depth + signal values + any concern ──────────
    detail_bits: list[str] = []
    if vec:
        detail_bits.append("vector DB experience in career history")
    if emb:
        detail_bits.append("embedding deployment mentioned")
    if evl:
        detail_bits.append("eval framework (NDCG/MRR) referenced")

    signal_bits: list[str] = []
    if dsa <= 30:
        signal_bits.append(f"active {dsa}d ago")
    elif dsa <= 90:
        signal_bits.append(f"active ~{dsa//30}mo ago")
    elif dsa > 365:
        signal_bits.append(f"inactive {dsa//30}mo — availability concern")
    if otw:
        signal_bits.append("open-to-work")
    if rr >= 0.50:
        signal_bits.append(f"response rate {rr:.0%}")
    elif rr < 0.20 and rank <= 50:
        signal_bits.append(f"low response rate ({rr:.0%}) — reachability concern")
    if nd > 90:
        signal_bits.append(f"{nd}d notice period")
    elif nd <= 30:
        signal_bits.append(f"{nd}d notice")

    all_bits = detail_bits + signal_bits
    if all_bits:
        parts.append("; ".join(all_bits) + ".")

    # Glue together, keeping to ≤2 sentences
    return " ".join(parts[:2])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Redrob fast ranker")
    parser.add_argument("--artifacts",  default="artifacts",
                        help="Directory of precomputed artifacts")
    parser.add_argument("--candidates",
                        default="India_runs_data_and_ai_challenge/candidates.jsonl",
                        help="Path to candidates.jsonl (used only for ID validation)")
    parser.add_argument("--out",  default="submission.csv",
                        help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    art = Path(args.artifacts)
    t_wall = time.time()

    # ── 1. Load artifacts ─────────────────────────────────────────────────────
    print("Loading artifacts …")
    t0 = time.time()

    df = pd.read_parquet(art / "candidate_features.parquet")
    n  = len(df)

    candidate_ids = np.load(art / "candidate_ids.npy", allow_pickle=True)
    embeddings    = np.load(art / "career_embeddings.npy")   # (N, 384) float32

    retriever = bm25s.BM25.load(str(art / "bm25_index"), load_corpus=False)

    model = SentenceTransformer(str(art / "model"), local_files_only=True)
    print(f"  Loaded {n:,} candidates, embeddings {embeddings.shape}  "
          f"({time.time()-t0:.1f}s)")

    # ── 2. Embed JD query ─────────────────────────────────────────────────────
    print("Embedding JD query …")
    t0 = time.time()
    jd_vec = model.encode(
        [JD_EMBED_QUERY],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)              # (384,)
    print(f"  Done  ({time.time()-t0:.1f}s)")

    # ── 3. BM25 scores ────────────────────────────────────────────────────────
    print("Computing BM25 scores …")
    t0 = time.time()
    query_tokens = bm25s.tokenize([JD_BM25_QUERY], stopwords="en", show_progress=False)
    results, bm25_raw_sorted = retriever.retrieve(query_tokens, k=n)
    # results[0]: doc indices sorted by score desc; bm25_raw_sorted[0]: scores
    bm25_raw = np.zeros(n, dtype=np.float32)
    bm25_raw[results[0]] = bm25_raw_sorted[0]
    print(f"  BM25 done  ({time.time()-t0:.1f}s)")

    # ── 4. Dense cosine similarity ────────────────────────────────────────────
    print("Computing dense similarity …")
    t0 = time.time()
    # embeddings already L2-normalised -> dot product = cosine similarity
    dense_scores = (embeddings @ jd_vec).astype(np.float64)   # (N,)
    print(f"  Dense done  ({time.time()-t0:.1f}s)")

    # ── 5. Hybrid retrieval + percentile normalization ────────────────────────
    print("Combining and normalizing …")
    t0 = time.time()

    # Min-max normalize BM25 raw scores to [0, 1] before weighting
    bm25_f = bm25_raw.astype(np.float64)
    bm25_span = bm25_f.max() - bm25_f.min()
    if bm25_span > 0:
        bm25_f = (bm25_f - bm25_f.min()) / bm25_span

    hybrid_raw = 0.35 * bm25_f + 0.65 * dense_scores         # [0, ~1]

    # Percentile-normalize: rank-based 0->1 across the full pool (change #1)
    retrieval_pct = _percentile_normalize(hybrid_raw)         # [0, 1]
    print(f"  Done  ({time.time()-t0:.1f}s)")

    # ── 6. Structural JD-fit adjustments ──────────────────────────────────────
    print("Applying structural adjustments …")
    t0 = time.time()

    struct_adj = df.apply(_structural_adjustments, axis=1).values.astype(np.float64)
    match_scores = np.clip(retrieval_pct + struct_adj, 0.0, 1.0)

    print(f"  Done  ({time.time()-t0:.1f}s)")

    # ── 7. Availability multiplier ────────────────────────────────────────────
    print("Computing availability multipliers …")
    t0 = time.time()
    avail_mult = df.apply(_availability_multiplier, axis=1).values.astype(np.float64)
    print(f"  Done  ({time.time()-t0:.1f}s)")

    # ── 8. Honeypot gate ──────────────────────────────────────────────────────
    honeypot_mask = df["is_honeypot"].values.astype(bool)
    hp_count = honeypot_mask.sum()
    print(f"Honeypots gated out: {hp_count}")

    # ── 9. Final scores ────────────────────────────────────────────────────────
    final_scores = match_scores * avail_mult
    final_scores[honeypot_mask] = 0.0   # hard exclusion

    # Normalize to [0, 1] (top candidate = 1.0)
    top_val = final_scores.max()
    if top_val > 0:
        final_scores = final_scores / top_val

    # ── 10. Select top-100, assign ranks ──────────────────────────────────────
    # Primary sort: final_score descending
    # Tie-break: candidate_id ascending (as required by validator)
    sorted_idx = np.lexsort((
        df["candidate_id"].values,            # secondary: ascending (numpy lexsort reverses priority)
        -final_scores,                        # primary: descending score
    ))
    top_idx = sorted_idx[:args.top_k]

    elapsed_so_far = time.time() - t_wall
    print(f"\nRanking done in {elapsed_so_far:.1f}s. Generating reasoning …")
    t0 = time.time()

    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        row   = df.iloc[idx]
        cid   = str(candidate_ids[idx])
        score = round(float(final_scores[idx]), 6)
        reason = _generate_reasoning(row, rank)
        rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        score,
            "reasoning":    reason,
        })

    print(f"  Reasoning generated  ({time.time()-t0:.1f}s)")

    # ── 11. Write CSV ─────────────────────────────────────────────────────────
    out_df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    out_df.to_csv(args.out, index=False)
    print(f"\nSubmission written -> {args.out}  ({len(out_df)} rows)")

    # ── 12. Diagnostics ───────────────────────────────────────────────────────
    total_elapsed = time.time() - t_wall
    print(f"\nTotal wall time: {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")

    top20 = out_df.head(20)
    print("\nTop 20 preview:")
    print(f"{'Rank':<5} {'candidate_id':<16} {'Score':<8} Reasoning")
    print("─" * 100)
    for _, r in top20.iterrows():
        print(f"{r['rank']:<5} {r['candidate_id']:<16} {r['score']:<8.4f} {r['reasoning'][:80]}")

    # Honeypot rate in top-100
    top100_ids = set(out_df["candidate_id"].tolist())
    hp_in_top100 = df[df["candidate_id"].isin(top100_ids)]["is_honeypot"].sum()
    print(f"\nHoneypots in top-100: {hp_in_top100}  "
          f"({'PASS' if hp_in_top100 <= 10 else 'FAIL — exceeds 10% threshold'})")

    services_in_top100 = df[df["candidate_id"].isin(top100_ids)]["services_only_career"].sum()
    print(f"Services-only in top-100: {services_in_top100}")


if __name__ == "__main__":
    main()
