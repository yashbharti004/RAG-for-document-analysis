"""
Offline evaluation script to benchmark retrieval modes (vector, hybrid, rerank)
against the labeled question set.
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from rag_chain import RAGChain
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rag.eval")


def load_questions(path: Optional[str] = None) -> List[Dict[str, Any]]:
    file_path = Path(path) if path else (Path(config.EVAL_DIR) / "questions.json")
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation questions file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("questions", [])


UNANSWERABLE_PHRASES = [
    "don't have enough information",
    "not available",
    "cannot find",
    "no relevant",
    "don't know",
    "not mentioned",
    "couldn't find",
    "unable to find",
    "does not contain",
    "not present",
]


def score_prediction(
    predicted: str,
    expected: str,
    retrieved_sources: List[Dict],
    expected_source: str,
    expected_page: Optional[int],
) -> Dict[str, Any]:
    pred_clean = predicted.lower().strip()
    exp_clean = expected.lower().strip()

    # Check whether ground-truth document was retrieved
    retrieval_hit = False
    if expected_source:
        for src in retrieved_sources:
            src_name = src.get("source", "").lower()
            if expected_source.lower() in src_name:
                if expected_page is None or src.get("page") == expected_page:
                    retrieval_hit = True
                    break

    # Check unanswerable expected outputs
    if exp_clean in ("unanswerable", "not available", "n/a"):
        is_refusal = any(p in pred_clean for p in UNANSWERABLE_PHRASES)
        return {
            "verdict": "correct" if is_refusal else "wrong",
            "retrieval_hit": retrieval_hit,
            "score": 1.0 if is_refusal else 0.0,
        }

    # Token overlap heuristic for factual alignment
    keywords = [w for w in exp_clean.split() if len(w) > 3]
    if not keywords:
        keywords = exp_clean.split()

    matched_keywords = sum(1 for kw in keywords if kw in pred_clean)
    overlap_ratio = (matched_keywords / len(keywords)) if keywords else 0.0

    if overlap_ratio >= 0.7:
        return {"verdict": "correct", "retrieval_hit": retrieval_hit, "score": 1.0}
    elif overlap_ratio >= 0.3:
        return {"verdict": "partial", "retrieval_hit": retrieval_hit, "score": 0.5}
    else:
        return {"verdict": "wrong", "retrieval_hit": retrieval_hit, "score": 0.0}


def evaluate_mode(mode: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Starting evaluation for mode: %s (%d test queries)", mode, len(questions))

    chain = RAGChain(retriever_mode=mode)
    results = []
    total_score = 0.0
    retrieval_hits = 0

    for i, q in enumerate(questions, 1):
        question_text = q["question"]
        preview = (question_text[:55] + "...") if len(question_text) > 55 else question_text
        logger.info("[%2d/%2d] %s", i, len(questions), preview)

        try:
            res = chain.ask(question_text)
            scoring = score_prediction(
                predicted=res["answer"],
                expected=q["expected_answer"],
                retrieved_sources=res["sources"],
                expected_source=q.get("source_document", ""),
                expected_page=q.get("source_page"),
            )
        except Exception as exc:
            logger.error("Failed query: %s", exc)
            scoring = {"verdict": "wrong", "retrieval_hit": False, "score": 0.0}
            res = {"answer": f"ERROR: {exc}", "sources": [], "num_chunks_retrieved": 0}

        results.append({
            "question": question_text,
            "expected": q["expected_answer"],
            "predicted": res["answer"],
            "sources": res["sources"],
            "scoring": scoring,
            "difficulty": q.get("difficulty", "medium"),
            "type": q.get("type", "factual"),
        })

        total_score += scoring["score"]
        retrieval_hits += int(scoring["retrieval_hit"])

    n = len(questions)
    accuracy = (total_score / n * 100) if n else 0.0
    hit_rate = (retrieval_hits / n * 100) if n else 0.0

    correct = sum(1 for r in results if r["scoring"]["verdict"] == "correct")
    partial = sum(1 for r in results if r["scoring"]["verdict"] == "partial")
    wrong = sum(1 for r in results if r["scoring"]["verdict"] == "wrong")

    summary = {
        "mode": mode,
        "total": n,
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "accuracy": round(accuracy, 1),
        "retrieval_hit_rate": round(hit_rate, 1),
        "results": results,
    }

    logger.info(
        "Mode '%s' finished: Accuracy=%.1f%%, Retrieval Hit Rate=%.1f%% (Correct: %d, Partial: %d, Wrong: %d)",
        mode,
        accuracy,
        hit_rate,
        correct,
        partial,
        wrong,
    )
    return summary


def run_evaluation(
    modes: Optional[List[str]] = None,
    questions_path: Optional[str] = None,
) -> Dict[str, Any]:
    modes = modes or ["vector", "hybrid", "rerank"]
    questions = load_questions(questions_path)

    all_summaries: Dict[str, Any] = {}
    for mode in modes:
        all_summaries[mode] = evaluate_mode(mode, questions)

    # Print summary comparison table
    mode_titles = {
        "vector": "Baseline (Dense Vector)",
        "hybrid": "Hybrid (FAISS + BM25)",
        "rerank": "Hybrid + Cross-Encoder Rerank",
    }

    print("\n" + "=" * 62)
    print("                EVALUATION BENCHMARK RESULTS")
    print("=" * 62)
    print(f"{'Configuration':<35} {'Accuracy':>10} {'Hit Rate':>12}")
    print("-" * 62)
    for mode in modes:
        s = all_summaries[mode]
        title = mode_titles.get(mode, mode)
        print(f"{title:<35} {s['accuracy']:>9.1f}% {s['retrieval_hit_rate']:>11.1f}%")
    print("=" * 62 + "\n")

    # Save timestamped results
    output_dir = Path(config.EVAL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"results_{ts}.json"

    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    m: {k: v for k, v in s.items() if k != "results"}
                    for m, s in all_summaries.items()
                },
                "details": all_summaries,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Saved benchmark report to %s", out_file)
    return all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RAG pipeline accuracy")
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid", "rerank", "all"],
        default="all",
        help="Retrieval configuration to evaluate (default: all)",
    )
    parser.add_argument(
        "--questions",
        default=None,
        help="Path to questions JSON file",
    )
    args = parser.parse_args()

    selected_modes = ["vector", "hybrid", "rerank"] if args.mode == "all" else [args.mode]
    run_evaluation(modes=selected_modes, questions_path=args.questions)
