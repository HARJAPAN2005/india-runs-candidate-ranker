#!/usr/bin/env python3
"""
precompute.py — offline artifact builder for the Redrob candidate ranker.

Reads candidates.jsonl, extracts structured features, builds BAAI/bge-small-en-v1.5
embeddings over career-history text, and saves a bm25s retriever.  All artifacts
land in ./artifacts/ so rank.py runs with zero network access.

Runtime: 20-40 min on CPU (slow is fine — rank.py must be ≤5 min).

Usage:
    python precompute.py [--candidates PATH] [--model MODEL_ID] [--batch-size N]
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

# Force UTF-8 stdout on Windows so print() doesn't crash on non-ASCII progress chars
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import bm25s
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TODAY = date(2026, 6, 25)
ARTIFACTS = Path("artifacts")

SERVICES_FIRMS = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "hcl technologies", "tech mahindra", "hexaware",
    "mphasis", "mindtree", "l&t infotech", "ltimindtree", "lti mindtree",
    "syntel", "virtusa", "zensar", "kpit", "persistent systems", "cyient",
    "igate", "mastech", "niit technologies", "birlasoft", "dxc technology",
    "ntt data", "unisys",
    # Substring-matched: any "Genpact *" variant is caught because _is_services
    # does `any(firm in company_lower for firm in SERVICES_FIRMS)`.
    "genpact",
    # Other frequently-seen outsourcing firms missing from the original list
    "ibm global", "epam", "globallogic", "stefanini", "wipro bps",
    "atos", "sopra steria", "hexaware", "coforge", "mphasis",
})

# Companies that register as "IT Services" but are actually product companies
_PRODUCT_EXCEPTIONS = frozenset({
    "stripe", "twilio", "cloudflare", "github", "gitlab", "atlassian",
    "shopify", "freshworks", "zoho", "zerodha",
})

# ─────────────────────────────────────────────────────────────────────────────
# Compiled regexes
# ─────────────────────────────────────────────────────────────────────────────
# 16 distinct evidence families; score = # families matched / 8 (capped at 8)
_RECSYS_FAMILIES = [
    ("recommendation",    re.compile(r"\brecommend(?:ation|er|ing|s)?\b", re.I)),
    ("ranking",           re.compile(r"\branking\b", re.I)),
    ("search_system",     re.compile(r"\bsearch\s+(?:engine|system|relevance|quality|index|ranking)\b", re.I)),
    ("ir",                re.compile(r"\binformation\s+retrieval\b", re.I)),
    ("hybrid_retrieval",  re.compile(r"\bhybrid\s+(?:retrieval|search)\b", re.I)),
    ("dense_retrieval",   re.compile(r"\bdense\s+(?:retrieval|search)\b", re.I)),
    ("vector_search",     re.compile(r"\bvector\s+(?:search|index|database|store|db)\b", re.I)),
    ("search_infra",      re.compile(r"\b(?:BM25|TF.IDF|Lucene|Elasticsearch|Solr|OpenSearch)\b", re.I)),
    ("vector_db",         re.compile(r"\b(?:FAISS|Annoy|HNSW|ScaNN|Qdrant|Milvus|Pinecone|Weaviate)\b", re.I)),
    ("eval_metrics",      re.compile(r"\b(?:NDCG|nDCG|MRR|MAP\b|mean\s+average\s+precision|mean\s+reciprocal)\b", re.I)),
    ("ab_testing",        re.compile(r"\bA/?B\s+(?:test|experiment|testing)\b", re.I)),
    ("collab_filter",     re.compile(r"\b(?:collaborative\s+filtering|matrix\s+factorization|latent\s+factor)\b", re.I)),
    ("ltr",               re.compile(r"\b(?:learning.to.rank|LTR\b|pointwise|pairwise|listwise)\b", re.I)),
    ("retrieval_quality", re.compile(r"\b(?:embedding\s+drift|index\s+refresh|retrieval\s+quality|recall@\d|precision@\d)\b", re.I)),
    ("ml_serving",        re.compile(r"\b(?:feature\s+store|online\s+serving|model\s+serving|inference\s+(?:pipeline|server))\b", re.I)),
    ("deployed_ml",       re.compile(r"\b(?:deployed|in\s+production|production\s+traffic)\s+(?:model|system|pipeline)\b", re.I)),
]

_VECTOR_DB_RE   = re.compile(
    r"\b(?:FAISS|Milvus|Qdrant|Weaviate|Pinecone|Elasticsearch|OpenSearch|Solr|"
    r"pgvector|Chroma|vector\s+(?:database|store|index|search))\b", re.I)
_EMBEDDING_RE   = re.compile(
    r"\b(?:sentence.transform|SentenceTransformer|BGE\b|E5\b|bge-|e5-|"
    r"embedding\s+(?:model|drift|refresh|inference)|dense\s+encoder|"
    r"bi.?encoder|cross.?encoder|MTEB)\b", re.I)
_EVAL_RE        = re.compile(
    r"\b(?:NDCG|nDCG|MRR|MAP\b|precision@|recall@|A/?B\s+test|offline\s+eval|"
    r"online\s+eval|interleaving|relevance\s+judg|click.through)\b", re.I)
_DEPLOYED_RE    = re.compile(
    r"\b(?:deployed|production|serving|real\s+users?|live\s+traffic|"
    r"prod\s+system|rollout|canary|shipped\s+to\s+prod)\b", re.I)
_RESEARCH_TITLE = re.compile(
    r"\b(?:research\s+scientist|research\s+engineer|postdoc|phd\s+intern|"
    r"research\s+intern|professor|lecturer|researcher)\b", re.I)
_RESEARCH_CO    = re.compile(
    r"\b(?:university|institute\b|laboratory|academia|college|IIT\s|NIT\s|"
    r"\bMIT\b|Stanford|CMU|IISc|research\s+lab|DeepMind|"
    r"Microsoft\s+Research|Google\s+Research|Meta\s+AI)\b", re.I)
_CV_SPEECH_RE   = re.compile(
    r"\b(?:image\s+classif|object\s+detect|segmentation|computer\s+vision|"
    r"speech\s+recogni|automatic\s+speech|ASR\b|TTS\b|text.to.speech|"
    r"speaker\s+verif|speaker\s+identif|robotics?|autonomous\s+(?:driving|vehicle)|lidar)\b",
    re.I)
_NLP_IR_RE      = re.compile(
    r"\b(?:NLP\b|natural\s+language|text\s+classif|sentiment|named\s+entity|"
    r"information\s+retrieval|question\s+answer|machine\s+translation|"
    r"summarization|language\s+model|BERT\b|GPT\b|transformer\s+model|text\s+embed)\b",
    re.I)
_LANGCHAIN_RE   = re.compile(r"\b(?:langchain|llamaindex|llama.index|openai\s+api|gpt.4\s+api)\b", re.I)
_PRE_LLM_ML_RE  = re.compile(
    r"\b(?:sklearn|scikit|tensorflow|pytorch|keras|xgboost|spark\s+ml|"
    r"h2o|mlops|kubeflow|mlflow|production\s+ml|model\s+deployment|deployed\s+model)\b", re.I)

_AI_SKILL_WORDS = frozenset({
    "python", "pytorch", "tensorflow", "transformers", "bert", "gpt", "llm",
    "embeddings", "faiss", "elasticsearch", "opensearch", "qdrant", "milvus",
    "pinecone", "weaviate", "information retrieval", "ranking", "recommendation",
    "collaborative filtering", "mlops", "xgboost", "lightgbm", "deep learning",
    "nlp", "lora", "qlora", "rag", "bm25", "hybrid search", "dense retrieval",
    "ndcg", "scikit-learn", "sklearn", "sentence transformers", "a/b testing",
})

_PROF_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
_TIER_MAP  = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 4}
_EDU_FIELDS = frozenset({
    "computer science", "cs", "information technology", "it",
    "electrical engineering", "ece", "electronics",
    "statistics", "mathematics", "math", "data science",
    "machine learning", "artificial intelligence", "ai", "information systems",
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _parse_date(s) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s))
    except (ValueError, TypeError):
        return None


def _is_services(company: str, industry: str) -> bool:
    """Return True if this role is at a services/outsourcing firm."""
    if industry and industry.lower() == "it services":
        c = (company or "").lower()
        if any(ex in c for ex in _PRODUCT_EXCEPTIONS):
            return False
        return True
    c = (company or "").lower()
    return any(firm in c for firm in SERVICES_FIRMS)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(cand: dict) -> dict:
    cid      = cand["candidate_id"]
    profile  = cand.get("profile", {})
    career   = cand.get("career_history", [])
    skills   = cand.get("skills", [])
    edu      = cand.get("education", [])
    signals  = cand.get("redrob_signals", {})

    # ── Career doc: headline + summary + descriptions (most-recent first) ───
    sorted_career = sorted(
        career,
        key=lambda r: r.get("start_date", "0000-00-00"),
        reverse=True,
    )
    doc_parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]
    for r in sorted_career:
        role_line = f"{r.get('title','')} at {r.get('company','')}: {r.get('description','')}"
        doc_parts.append(role_line)
    career_doc = "\n".join(p for p in doc_parts if p.strip())

    # ── YoE ─────────────────────────────────────────────────────────────────
    yoe = float(profile.get("years_of_experience") or 0)

    # ── Career structure ─────────────────────────────────────────────────────
    total_months = product_months = services_count = product_count = 0
    recent_role_product = False
    tenure_list: list[int] = []
    all_roles_research = bool(sorted_career)  # starts True, becomes False on counter-evidence

    for i, r in enumerate(sorted_career):
        dur    = int(r.get("duration_months") or 0)
        co     = r.get("company", "")
        ind    = r.get("industry", "")
        title  = r.get("title", "")
        desc   = r.get("description", "")
        is_cur = bool(r.get("is_current", False))

        total_months += dur
        if _is_services(co, ind):
            services_count += 1
        else:
            product_count += 1
            product_months += dur

        if i == 0:
            recent_role_product = not _is_services(co, ind)

        if not is_cur and dur > 0:
            tenure_list.append(dur)

        # Research-only: any non-research role OR production evidence breaks the flag
        is_research_role = bool(_RESEARCH_TITLE.search(title)) or bool(_RESEARCH_CO.search(co))
        if not is_research_role or bool(_DEPLOYED_RE.search(desc)):
            all_roles_research = False

    product_ratio    = product_months / max(total_months, 1)
    services_only    = services_count > 0 and product_count == 0
    is_research_only = all_roles_research
    is_job_hopper    = bool(tenure_list) and (sum(tenure_list) / len(tenure_list) < 18)

    # ── Technical evidence (parsed from career_doc) ──────────────────────────
    recsys_families_hit = {lbl for lbl, pat in _RECSYS_FAMILIES if pat.search(career_doc)}
    recsys_evidence_score   = min(len(recsys_families_hit), 8) / 8.0
    vector_db_evidence      = bool(_VECTOR_DB_RE.search(career_doc))
    embedding_evidence      = bool(_EMBEDDING_RE.search(career_doc))
    eval_framework_evidence = bool(_EVAL_RE.search(career_doc))
    deployed_evidence       = bool(_DEPLOYED_RE.search(career_doc))
    has_oss = (
        float(signals.get("github_activity_score") or -1) > 0
        or bool(re.search(r"\bopen.source\b|\bgithub\.com\b", career_doc, re.I))
    )

    cv_speech_n = len(_CV_SPEECH_RE.findall(career_doc))
    nlp_ir_n    = len(_NLP_IR_RE.findall(career_doc))
    cv_speech_without_nlp = cv_speech_n >= 3 and nlp_ir_n < 2

    recent_text = (profile.get("summary", "") + " " +
                   (sorted_career[0].get("description", "") if sorted_career else ""))
    langchain_only_recent = (
        bool(_LANGCHAIN_RE.search(recent_text)) and
        not bool(_PRE_LLM_ML_RE.search(career_doc))
    )

    # ── Skills ───────────────────────────────────────────────────────────────
    ai_core_count = 0
    has_python = False
    for s in skills:
        nm = (s.get("name") or "").lower()
        pr = _PROF_RANK.get(s.get("proficiency", "beginner"), 1)
        if nm == "python" and pr >= 2:
            has_python = True
        if any(kw in nm for kw in _AI_SKILL_WORDS) and pr >= 2:
            ai_core_count += 1

    assess = signals.get("skill_assessment_scores") or {}
    skill_assessment_avg = (sum(assess.values()) / len(assess)) if assess else None

    # ── Education ────────────────────────────────────────────────────────────
    edu_tiers = [_TIER_MAP.get(e.get("tier", "unknown"), 4) for e in edu]
    best_edu_tier      = min(edu_tiers) if edu_tiers else 4
    edu_field_relevant = any(
        any(f in (e.get("field_of_study") or "").lower() for f in _EDU_FIELDS)
        for e in edu
    )

    # ── Redrob signals ────────────────────────────────────────────────────────
    last_active_d     = _parse_date(signals.get("last_active_date"))
    days_since_active = (TODAY - last_active_d).days if last_active_d else 999
    open_to_work      = bool(signals.get("open_to_work_flag", False))
    response_rate     = float(signals.get("recruiter_response_rate") or 0)
    interview_rate    = float(signals.get("interview_completion_rate") or 0)
    notice_days       = int(signals.get("notice_period_days") or 60)
    profile_compl     = float(signals.get("profile_completeness_score") or 0) / 100.0
    verified_both     = (bool(signals.get("verified_email")) and
                         bool(signals.get("verified_phone")))
    gh_raw            = signals.get("github_activity_score", -1)
    github_score      = float(gh_raw) if gh_raw is not None and float(gh_raw) >= 0 else None
    oar               = signals.get("offer_acceptance_rate", -1)
    offer_acceptance  = float(oar) if oar is not None and float(oar) >= 0 else None
    willing_relocate  = bool(signals.get("willing_to_relocate", False))
    sal               = signals.get("expected_salary_range_inr_lpa") or {}
    salary_max        = float(sal.get("max") or 0)
    salary_min        = float(sal.get("min") or 0)

    # ── Honeypot checks 1, 2, 4, 5 ──────────────────────────────────────────
    # Check 1: claimed duration_months vs actual date span (tolerance +6 months)
    hp_duration = False
    for r in career:
        start = _parse_date(r.get("start_date"))
        end   = TODAY if r.get("is_current") else _parse_date(r.get("end_date"))
        dur   = int(r.get("duration_months") or 0)
        if start and end and end >= start:
            actual = (end.year - start.year) * 12 + (end.month - start.month)
            if dur > actual + 6:
                hp_duration = True
                break

    # Check 2: proficiency "expert" with zero months of usage
    hp_expert_zero = any(
        s.get("proficiency") == "expert" and int(s.get("duration_months") or 1) == 0
        for s in skills
    )

    # Check 4: more than one role marked is_current
    hp_multi_current = sum(1 for r in career if r.get("is_current", False)) > 1

    # Check 5: non-current role with end_date in the future
    hp_future = any(
        not r.get("is_current", False)
        and _parse_date(r.get("end_date")) is not None
        and _parse_date(r.get("end_date")) > TODAY  # type: ignore[operator]
        for r in career
    )

    is_honeypot = hp_duration or hp_expert_zero or hp_multi_current or hp_future

    # ── Reasoning helpers ────────────────────────────────────────────────────
    notable_companies = ",".join(
        r.get("company", "") for r in sorted_career[:3] if r.get("company")
    )
    recsys_terms_str = ",".join(sorted(recsys_families_hit))

    return {
        "candidate_id":          cid,
        "career_doc":            career_doc,   # popped before saving to parquet
        # Profile
        "yoe":                   yoe,
        "current_title":         profile.get("current_title", ""),
        "current_company":       profile.get("current_company", ""),
        "current_industry":      profile.get("current_industry", ""),
        "country":               profile.get("country", ""),
        "location":              profile.get("location", ""),
        # Career structure
        "total_career_months":   total_months,
        "product_months":        product_months,
        "product_ratio":         product_ratio,
        "services_only_career":  services_only,
        "recent_role_product":   recent_role_product,
        "is_research_only":      is_research_only,
        "is_job_hopper":         is_job_hopper,
        "langchain_only_recent": langchain_only_recent,
        "cv_speech_without_nlp": cv_speech_without_nlp,
        # Technical evidence
        "recsys_evidence_score":        recsys_evidence_score,
        "vector_db_evidence":           vector_db_evidence,
        "embedding_evidence":           embedding_evidence,
        "eval_framework_evidence":      eval_framework_evidence,
        "deployed_evidence":            deployed_evidence,
        "has_oss":                      has_oss,
        "recsys_terms_str":             recsys_terms_str,
        # Skills
        "ai_core_skill_count":          ai_core_count,
        "has_python":                   has_python,
        "skill_assessment_avg":         skill_assessment_avg,
        # Education
        "best_edu_tier":                best_edu_tier,
        "edu_field_relevant":           edu_field_relevant,
        # Signals
        "days_since_active":            days_since_active,
        "open_to_work":                 open_to_work,
        "response_rate":                response_rate,
        "interview_completion_rate":    interview_rate,
        "notice_days":                  notice_days,
        "profile_completeness":         profile_compl,
        "verified_both":                verified_both,
        "github_score":                 github_score,
        "offer_acceptance":             offer_acceptance,
        "willing_relocate":             willing_relocate,
        "salary_max_lpa":               salary_max,
        "salary_min_lpa":               salary_min,
        "notable_companies":            notable_companies,
        # Honeypot flags
        "hp_duration":                  hp_duration,
        "hp_expert_zero":               hp_expert_zero,
        "hp_multi_current":             hp_multi_current,
        "hp_future":                    hp_future,
        "is_honeypot":                  is_honeypot,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Redrob precompute pipeline")
    parser.add_argument(
        "--candidates",
        default="India_runs_data_and_ai_challenge/candidates.jsonl",
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model ID. Default is all-MiniLM-L6-v2 (6-layer, "
             "256 max tokens — fast and memory-safe on CPU).",
    )
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size. Keep <=64 on 16 GB CPU-only.")
    parser.add_argument("--doc-max-chars", type=int, default=1500,
                        help="Truncate career_doc to this many characters before "
                             "encoding. Caps token length, prevents OOM.")
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                        help="Save partial embeddings every N docs (resume-safe).")
    parser.add_argument("--force-features", action="store_true",
                        help="Re-extract features and rebuild BM25 even if parquet "
                             "already exists. Skips embedding if npy is present.")
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    t_total = time.time()

    features_ready = (
        not args.force_features
        and (ARTIFACTS / "candidate_features.parquet").exists()
        and (ARTIFACTS / "candidate_ids.npy").exists()
        and (ARTIFACTS / "bm25_index").exists()
    )
    emb_ready = (
        (ARTIFACTS / "career_embeddings.npy").exists()
        and (ARTIFACTS / "model").exists()
    )

    career_docs: list[str] = []

    if features_ready:
        # ── Steps 1-4 already done: just reload docs for embedding ──────────
        print("Steps 1-4 artifacts already present — skipping to step 5.")
        print("Reloading candidates to rebuild career_docs …")
        t0 = time.time()
        with open(args.candidates, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                cand = json.loads(line)
                profile = cand.get("profile", {})
                ch = sorted(
                    cand.get("career_history", []),
                    key=lambda r: r.get("start_date", "0000-00-00"),
                    reverse=True,
                )
                parts = [profile.get("headline", ""), profile.get("summary", "")]
                for r in ch:
                    parts.append(
                        f"{r.get('title','')} at {r.get('company','')}: "
                        f"{r.get('description','')}"
                    )
                career_docs.append("\n".join(p for p in parts if p.strip()))
        n = len(career_docs)
        print(f"  Reloaded {n:,} career docs  ({time.time()-t0:.1f}s)")
        df = pd.read_parquet(ARTIFACTS / "candidate_features.parquet")
    else:
        # ── Full pipeline ────────────────────────────────────────────────────
        print("Step 1/5  Parsing candidates …")
        t0 = time.time()
        candidates: list[dict] = []
        with open(args.candidates, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
        n = len(candidates)
        print(f"          {n:,} candidates  ({time.time() - t0:.1f}s)")

        print("Step 2/5  Extracting features …")
        t0 = time.time()
        records = [extract_features(c) for c in candidates]
        career_docs = [r.pop("career_doc") for r in records]
        print(f"          Done  ({time.time() - t0:.1f}s)")

        print("Step 3/5  Saving feature table …")
        t0 = time.time()
        df = pd.DataFrame(records)
        df.to_parquet(ARTIFACTS / "candidate_features.parquet", index=False)
        np.save(ARTIFACTS / "candidate_ids.npy", df["candidate_id"].values.astype(str))
        print(f"          {df.shape[0]:,} rows x {df.shape[1]} cols -> "
              f"candidate_features.parquet  ({time.time() - t0:.1f}s)")

        print("Step 4/5  Building BM25 index …")
        t0 = time.time()
        corpus_tokens = bm25s.tokenize(career_docs, stopwords="en", show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(str(ARTIFACTS / "bm25_index"))
        print(f"          Indexed {n:,} docs -> bm25_index/  ({time.time() - t0:.1f}s)")

    # ── Step 5: Embeddings (with truncation + checkpointing) ─────────────────
    emb_path   = ARTIFACTS / "career_embeddings.npy"
    ckpt_path  = ARTIFACTS / "career_embeddings_ckpt.npy"
    ckpt_n_path = ARTIFACTS / "career_embeddings_ckpt_n.txt"

    if emb_ready:
        print("Step 5/5  career_embeddings.npy and model already present — skipping embedding.")
        elapsed = time.time() - t_total
        print(f"\n{'='*55}")
        print(f"Total wall time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
        print(f"\nArtifact sizes:")
        for p in sorted(ARTIFACTS.rglob("*")):
            if p.is_file():
                sz = p.stat().st_size
                label = f"{sz/1024**2:.1f} MB" if sz > 1024**2 else f"{sz/1024:.0f} KB"
                print(f"  {p.relative_to(ARTIFACTS)}  {label}")
        hp_count = df["is_honeypot"].sum()
        print(f"\nHoneypot candidates flagged: {hp_count}")
        print(f"Services-only careers:       {df['services_only_career'].sum()}")
        print(f"Research-only:               {df['is_research_only'].sum()}")
        print(f"Has recsys evidence (>0):    {(df['recsys_evidence_score'] > 0).sum()}")
        return

    # Truncate docs to cap token length and prevent OOM
    truncated_docs = [d[:args.doc_max_chars] for d in career_docs]

    # Check for partial checkpoint
    start_idx = 0
    partial: np.ndarray | None = None
    if ckpt_path.exists() and ckpt_n_path.exists():
        saved_n = int(ckpt_n_path.read_text().strip())
        if 0 < saved_n < n:
            partial = np.load(str(ckpt_path))
            start_idx = saved_n
            print(f"Resuming from checkpoint at doc {start_idx:,} / {n:,}")

    print(f"Step 5/5  Embedding with {args.model} "
          f"(batch={args.batch_size}, doc_max_chars={args.doc_max_chars}) …")
    if start_idx > 0:
        print(f"          Resuming from {start_idx:,} (checkpoint found)")
    t0 = time.time()

    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()

    # Preallocate output array
    embeddings = np.zeros((n, dim), dtype=np.float32)
    if partial is not None:
        embeddings[:start_idx] = partial

    # Encode in chunks, saving checkpoints
    chunk_size = args.checkpoint_every
    for chunk_start in range(start_idx, n, chunk_size):
        chunk_end  = min(chunk_start + chunk_size, n)
        chunk_docs = truncated_docs[chunk_start:chunk_end]

        chunk_emb = model.encode(
            chunk_docs,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        embeddings[chunk_start:chunk_end] = chunk_emb

        # Save checkpoint
        np.save(str(ckpt_path), embeddings[:chunk_end])
        ckpt_n_path.write_text(str(chunk_end))
        elapsed_so_far = time.time() - t0
        rate = (chunk_end - start_idx) / elapsed_so_far
        remaining = (n - chunk_end) / rate if rate > 0 else 0
        print(f"  Checkpoint {chunk_end:,}/{n:,}  "
              f"elapsed={elapsed_so_far/60:.1f}m  "
              f"ETA={remaining/60:.1f}m")

    np.save(str(emb_path), embeddings)
    # Clean up checkpoint files
    if ckpt_path.exists():
        ckpt_path.unlink()
    if ckpt_n_path.exists():
        ckpt_n_path.unlink()
    print(f"  Embeddings {embeddings.shape} saved  ({(time.time()-t0)/60:.1f}m)")

    model_dir = str(ARTIFACTS / "model")
    print(f"  Saving model to {model_dir} for offline rank.py …")
    model.save(model_dir)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'='*55}")
    print(f"Total wall time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"\nArtifact sizes:")
    for p in sorted(ARTIFACTS.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            label = f"{sz/1024**2:.1f} MB" if sz > 1024**2 else f"{sz/1024:.0f} KB"
            print(f"  {p.relative_to(ARTIFACTS)}  {label}")

    hp_count = df["is_honeypot"].sum()
    print(f"\nHoneypot candidates flagged: {hp_count}")
    print(f"Services-only careers:       {df['services_only_career'].sum()}")
    print(f"Research-only:               {df['is_research_only'].sum()}")
    print(f"Has recsys evidence (>0):    {(df['recsys_evidence_score'] > 0).sum()}")


if __name__ == "__main__":
    main()
