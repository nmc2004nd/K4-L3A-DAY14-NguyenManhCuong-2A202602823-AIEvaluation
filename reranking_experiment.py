"""Reproduce the Exercise 3.5 before/after retrieval measurements.

The experiment deliberately reranks the same chunk objects.  Context recall is
therefore invariant, while rank-aware average precision may change.
"""

from __future__ import annotations

import json
from pathlib import Path

from template import RAGASEvaluator, rerank_by_overlap


ROOT = Path(__file__).resolve().parent
CASE_IDS = ("E02", "E03", "M05", "M06", "H02")


def main() -> None:
    golden_payload = json.loads((ROOT / "golden_dataset.json").read_text(encoding="utf-8"))
    trace_payload = json.loads(
        (ROOT / "artifacts" / "actual_answers.json").read_text(encoding="utf-8")
    )
    golden = {case["id"]: case for case in golden_payload["qa_pairs"]}
    traces = {case["id"]: case for case in trace_payload["answers"]}
    evaluator = RAGASEvaluator()
    rows: list[tuple[str, float, float, float, float]] = []

    for case_id in CASE_IDS:
        trace = traces[case_id]
        expected = golden[case_id]["expected_answer"]
        contexts = [chunk["text"] for chunk in trace["retrieved_contexts"]]
        reranked = rerank_by_overlap(contexts, trace["question"])

        # This assertion proves that reranking changed only order, including in
        # the unlikely event that two chunks have identical text.
        assert sorted(contexts) == sorted(reranked)
        rows.append(
            (
                case_id,
                evaluator.evaluate_context_recall(contexts, expected),
                evaluator.evaluate_context_recall(reranked, expected),
                evaluator.evaluate_context_precision(contexts, expected),
                evaluator.evaluate_context_precision(reranked, expected),
            )
        )

    print("ID   Recall before  Recall after  Precision before  Precision after  Delta")
    for case_id, recall_before, recall_after, precision_before, precision_after in rows:
        print(
            f"{case_id:<3}  {recall_before:>13.3f}  {recall_after:>12.3f}"
            f"  {precision_before:>16.3f}  {precision_after:>15.3f}"
            f"  {precision_after - precision_before:>+.3f}"
        )
    averages = [sum(row[column] for row in rows) / len(rows) for column in range(1, 5)]
    print(
        f"Avg  {averages[0]:>13.3f}  {averages[1]:>12.3f}"
        f"  {averages[2]:>16.3f}  {averages[3]:>15.3f}"
        f"  {averages[3] - averages[2]:>+.3f}"
    )


if __name__ == "__main__":
    main()
