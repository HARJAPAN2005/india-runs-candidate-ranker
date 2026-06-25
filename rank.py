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
    """
    Map years-of-experience to a [0, 1] fit score.
    Sweet spot: 5–11 yrs. Soft linear decay beyond 11 (-4% per year over 11).
    """
    if yoe <= 0:
        return 0.0
    if yoe < 3:
        return yoe / 3 * 0.4          # 0 -> 0.40  (too junior)
    if yoe < 5:
        return 0.4 + (yoe - 3) / 2 * 0.6   # 3 -> 0.40, 5 -> 1.0
    if yoe <= 11:
        return 1.0                     # sweet spot
    # Soft linear decay: -4% per year beyond 11, floor at 0.60
    return max(0.60, 1.0 - (yoe - 11) * 0.04)


# Services-firm substrings for current-employer gate.
# Mirrors SERVICES_FIRMS in precompute.py; checked case-insensitively against
# current_company so any variant ("Genpact AI", "Genpact Technologies", etc.) is caught.
_SERVICES_SUBSTRINGS: frozenset[str] = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "hexaware", "mphasis", "mindtree",
    "l&t infotech", "ltimindtree", "lti mindtree", "syntel", "virtusa",
    "zensar", "kpit", "persistent systems", "cyient", "igate", "mastech",
    "niit technologies", "birlasoft", "dxc technology", "ntt data", "unisys",
    "genpact", "ibm global", "epam", "globallogic", "stefanini", "atos",
    "sopra steria", "coforge",
})


def _current_company_is_services(company: str) -> bool:
    """Case-insensitive substring match against known services firms."""
    c = (company or "").lower().strip()
    return any(s in c for s in _SERVICES_SUBSTRINGS)


# Tiered recsys evidence families.
# Strong (2.0 pts): specific ranking/retrieval/recsys evidence — the real signal.
# Medium (1.0 pt): infra/tooling relevant but not proof of ranking systems.
# Weak (0.3 pts): generic signals easily faked or incidentally present.
# Normalised by 8 (4 strong families = full credit); capped at 1.0.
_RECSYS_STRONG = frozenset({
    "recommendation", "ltr", "eval_metrics", "dense_retrieval", "hybrid_retrieval",
    "vector_search", "vector_db", "collab_filter", "ir", "retrieval_quality",
})
_RECSYS_WEAK = frozenset({
    "ab_testing", "search_system", "deployed_ml",
})
# Everything else (ranking, search_infra, ml_serving) → medium (1.0 pt)


def _tiered_recsys_score(terms_str: str) -> float:
    """
    Compute a [0, 1] recsys evidence score that weights strong signals clearly
    above generic ones (A/B testing, generic search, etc.).
    """
    terms = {t for t in str(terms_str).split(",") if t}
    pts = sum(
        2.0 if t in _RECSYS_STRONG else 0.3 if t in _RECSYS_WEAK else 1.0
        for t in terms
    )
    return min(pts / 8.0, 1.0)


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
    adj += 0.18 * _yoe_fit(float(row["yoe"]))                 # [0, 0.18]

    prod_ratio = float(row["product_ratio"])
    adj += 0.18 * prod_ratio                                   # [0, 0.18]
    if row["recent_role_product"]:
        adj += 0.05                                            # recency bonus

    adj += 0.22 * _tiered_recsys_score(row["recsys_terms_str"]) # [0, 0.22]  biggest signal
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


# Priority order for lead signal selection: most specific/rare first.
# Two candidates with the same terms get the same top-1 signal, but voice
# variation (below) ensures the sentence is worded differently.
_SIGNAL_PRIORITY = [
    "ltr", "eval_metrics", "dense_retrieval", "hybrid_retrieval",
    "retrieval_quality", "collab_filter", "ir", "vector_search",
    "recommendation", "vector_db", "search_infra", "ranking",
    "ml_serving", "search_system", "ab_testing", "deployed_ml",
]


def _generate_reasoning(row: pd.Series, rank: int) -> str:
    """
    Produce a 1–2 sentence reasoning grounded in real profile facts.

    Divergence rules:
    - Lead signal is the highest-priority term the candidate actually has,
      so two candidates with different strongest signals get different leads.
    - voice = candidate_id % 4 selects one of four distinct sentence templates,
      so two candidates with *identical* term sets still read differently.
    - Tech bullet order in sentence 2 is rotated by voice, adding further variety.
    - Candidates with no/weak recsys evidence get an explicit honest qualifier.
    """
    cid     = str(row["candidate_id"])
    company = str(row["current_company"] or "").strip()
    yoe     = float(row["yoe"])
    prod_r  = float(row["product_ratio"])
    terms   = {t for t in str(row["recsys_terms_str"]).split(",") if t}
    notable = [c for c in str(row["notable_companies"]).split(",") if c]
    dsa     = int(row["days_since_active"])
    rr      = float(row["response_rate"])
    nd      = int(row["notice_days"])
    otw     = bool(row["open_to_work"])
    svc     = bool(row["services_only_career"])
    vec     = bool(row["vector_db_evidence"])
    emb     = bool(row["embedding_evidence"])
    evl     = bool(row["eval_framework_evidence"])

    tiered = _tiered_recsys_score(str(row["recsys_terms_str"]))

    # Signals in priority order, labelled
    ordered = [t for t in _SIGNAL_PRIORITY if t in terms]
    top1_key = ordered[0] if ordered else None
    top2_key = ordered[1] if len(ordered) >= 2 else None
    top1 = _RECSYS_LABEL_MAP.get(top1_key, top1_key) if top1_key else None
    top2 = _RECSYS_LABEL_MAP.get(top2_key, top2_key) if top2_key else None

    # Deterministic style selector (0–3) per candidate
    voice = int(cid.split("_")[-1]) % 4
    co_str = f" (including {', '.join(notable[:2])})" if notable else ""

    # ── Lead sentence ─────────────────────────────────────────────────────────
    if not svc and tiered >= 0.25 and top1:
        pair = f"{top1} and {top2}" if top2 else top1
        if voice == 0:
            # Evidence-first: "N-yr career (Co1, Co2) shows direct evidence of X and Y"
            lead = (f"{yoe:.0f}-yr career{co_str} shows direct evidence of "
                    f"{pair} in career descriptions.")
        elif voice == 1:
            # Span + company: "N years across Co1 and Co2, with hands-on X in production"
            co_across = (f" across {' and '.join(notable[:2])}"
                         if len(notable) >= 2 else co_str)
            lead = (f"{yoe:.0f} years{co_across}, with hands-on "
                    f"{top1} experience in production.")
        elif voice == 2:
            # Evidence-led, no "shows": "Career-spanning evidence of X and Y (Co1, Co2)"
            lead = f"Career-spanning evidence of {pair}{co_str}."
        else:
            # Company-first: "From Co1 to Co2: N-yr track record in X"
            if len(notable) >= 2:
                lead = (f"From {notable[0]} to {notable[1]}: "
                        f"{yoe:.0f}-yr track record in {top1}.")
            else:
                lead = (f"{yoe:.0f}-yr career{co_str} demonstrates "
                        f"{top1} in production.")

    elif not svc and prod_r >= 0.6:
        # Meaningful product background but limited ranking-system evidence
        co_str2 = f", most recently at {company}" if company else ""
        lead = (
            f"{yoe:.0f} yrs primarily at product companies{co_str2} "
            f"(ratio {prod_r:.0%}); limited direct ranking-system evidence "
            f"in career descriptions - included on product-company fit and "
            f"availability signals."
        )
    elif svc:
        co_str2 = f" at {', '.join(notable[:2])}" if notable else ""
        title = str(row.get("current_title", "") or "").strip() or "candidate"
        lead = (
            f"{title} with {yoe:.0f} yrs{co_str2}; career is services-only "
            f"- does not match the JD's product-company requirement."
        )
    else:
        co_str2 = f" at {company}" if company else ""
        title = str(row.get("current_title", "") or "").strip() or "candidate"
        lead = f"{title}{co_str2}, {yoe:.0f} yrs experience; limited matching evidence."

    # ── Second sentence: tech depth + availability signals ────────────────────
    tech_pool: list[str] = []
    if vec: tech_pool.append("vector DB experience")
    if emb: tech_pool.append("embedding deployment")
    if evl: tech_pool.append("eval framework (NDCG/MRR)")
    # Rotate bullet order by voice so two candidates with the same tech facts
    # don't produce identical second sentences
    if tech_pool:
        rot = voice % len(tech_pool)
        tech_pool = tech_pool[rot:] + tech_pool[:rot]

    avail_pool: list[str] = []
    if dsa <= 30:
        avail_pool.append(f"active {dsa}d ago")
    elif dsa <= 90:
        avail_pool.append(f"active ~{dsa//30}mo ago")
    elif dsa > 365:
        avail_pool.append(f"inactive {dsa//30}mo - availability concern")
    if otw:
        avail_pool.append("open-to-work")
    if rr >= 0.50:
        avail_pool.append(f"response rate {rr:.0%}")
    elif rr < 0.20 and rank <= 50:
        avail_pool.append(f"low response rate ({rr:.0%}) - reachability concern")
    if nd > 90:
        avail_pool.append(f"{nd}d notice period")
    elif nd <= 30:
        avail_pool.append(f"{nd}d notice")

    all_bits = tech_pool + avail_pool
    second = ("; ".join(all_bits) + ".") if all_bits else ""

    return f"{lead} {second}".strip()


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
    # Floor at 0 only — no ceiling. Capping at 1.0 caused all top candidates to
    # tie because even small positive adjustments overflow the [0,1] retrieval range.
    match_scores = np.maximum(0.0, retrieval_pct + struct_adj)

    print(f"  Done  ({time.time()-t0:.1f}s)")

    # ── 7. Availability multiplier ────────────────────────────────────────────
    print("Computing availability multipliers …")
    t0 = time.time()
    avail_mult = df.apply(_availability_multiplier, axis=1).values.astype(np.float64)
    print(f"  Done  ({time.time()-t0:.1f}s)")

    # ── 8. Hard gates (honeypot + yoe floor + current-employer services) ──────
    honeypot_mask  = df["is_honeypot"].values.astype(bool)
    yoe_floor_mask = df["yoe"].values < 5          # JD minimum: 5 years
    curr_svc_mask  = df["current_company"].apply(
        lambda co: _current_company_is_services(str(co))
    ).values.astype(bool)
    hp_count   = honeypot_mask.sum()
    yoe_count  = yoe_floor_mask.sum()
    csvc_count = curr_svc_mask.sum()
    print(f"Honeypots gated out:           {hp_count}")
    print(f"Under-5yr gated out:           {yoe_count}")
    print(f"Current-employer svc gated out:{csvc_count}")

    # ── 9. Final scores ────────────────────────────────────────────────────────
    final_scores = match_scores * avail_mult
    final_scores[honeypot_mask]  = 0.0   # hard exclusion
    final_scores[yoe_floor_mask] = 0.0   # hard yoe floor
    final_scores[curr_svc_mask]  = 0.0   # currently at services firm

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
