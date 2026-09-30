"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable
import json 


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    # TODO: define fields
    # Hints:
    #   context: str = ""
    #   metadata: dict = field(default_factory=dict)
    #   retrieved_contexts: list = field(default_factory=list)
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    # TODO: define fields
    # Hints:
    #   failure_type: str | None = None
    #   context_precision: float | None = None
    #   context_recall: float | None = None
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0

        TODO: Return mean of the three metric scores
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        # TODO
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context)
        if not answer_tokens:
            return 1.0
        overlap = answer_tokens & context_tokens
        faithfulness = len(overlap) / len(answer_tokens)
        return max(0.0, min(1.0, faithfulness))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        # TODO
        question_tokens = _tokenize(question)
        answer_tokens = _tokenize(answer)
        if not question_tokens:
            return 1.0
        overlap = answer_tokens & question_tokens
        relevance = len(overlap) / len(question_tokens)
        return max(0.0, min(1.0, relevance))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        # TODO
        expected_tokens = _tokenize(expected)
        answer_tokens = _tokenize(answer)
        if not expected_tokens:
            return 1.0
        overlap = answer_tokens & expected_tokens
        completeness = len(overlap) / len(expected_tokens)
        return max(0.0, min(1.0, completeness))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        """
        # TODO
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens |= _tokenize(chunk)
        overlap = expected_tokens & union_tokens
        recall = len(overlap) / len(expected_tokens)
        return max(0.0, min(1.0, recall))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        # TODO
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0
        
        # 1. Xác định tính relevant cho từng chunk
        relevant_flags: list[bool] = []
        for chunk in contexts:
            chunk_tokens = _tokenize(chunk)
            overlap = chunk_tokens & expected_tokens
            rel_score = len(overlap) / len(expected_tokens)
            relevant_flags.append(rel_score >= relevance_threshold)

        total_relevant = sum(relevant_flags)
        if total_relevant == 0:
            return 0.0

        # 2 & 3. Tính Precision@k tại các vị trí relevant và cộng dồn AP@K
        precision_sum = 0.0
        relevant_count_so_far = 0

        for k, is_rel in enumerate(relevant_flags, start=1):
            if is_rel:
                relevant_count_so_far += 1
                precision_at_k = relevant_count_so_far / k
                precision_sum += precision_at_k

        ap_score = precision_sum / total_relevant
        return max(0.0, min(1.0, float(ap_score)))

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.
        """
        # TODO

        # 1. Evaluate answer-side metrics
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        # 2. Evaluate retrieval-side metrics if contexts are provided
        ctx_recall: float | None = None
        ctx_precision: float | None = None
        if contexts is not None:
            ctx_recall = self.evaluate_context_recall(contexts, expected)
            ctx_precision = self.evaluate_context_precision(contexts, expected)

        # 3. Determine pass/fail status (passed if all three answer metrics >= 0.5)
        passed = (faithfulness >= 0.5) and (relevance >= 0.5) and (completeness >= 0.5)

        # 4. Determine failure_type (First match wins)
        failure_type: str | None = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        # 5. Build QAPair and EvalResult
        qa_pair = QAPair(
            question=question,
            expected_answer=expected,
            context=context,
            retrieved_contexts=contexts if contexts is not None else [],
        )

        return EvalResult(
            qa_pair=qa_pair,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=ctx_precision,
            context_recall=ctx_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.

    Reordering relevant chunks toward the top increases the rank-aware
    Context Precision WITHOUT changing the retrieved set.

    Hint: sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)),
                 reverse=True)
    """
    # TODO (Bonus — Exercise 3.5): implement the reranker
    query_tokens = _tokenize(query)
    ranked_contexts = sorted(
        contexts,
        key=lambda c: len(_tokenize(c) & query_tokens),
        reverse=True
    )
    return ranked_contexts


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        # TODO: store judge_llm_fn  
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,                                                                                                                                    
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        Score an AI response using the judge LLM.

        Args:
            question: The original question.
            answer:   The AI's answer to score.
            rubric:   Dict mapping criterion name → description.
                      Example: {"accuracy": "Is the answer factually correct?",
                                "clarity": "Is the answer clear and well-structured?"}

        Behavior:
            1. Build a judge prompt that includes the question, answer, and rubric.
            2. Call judge_llm_fn(prompt).
            3. Parse the response for scores.

        For simplicity, if the LLM response can't be parsed as JSON scores,
        return a default score of 0.5 for each criterion.

        Returns:
            {
                "scores":    dict[str, float],  # criterion → score 0-1
                "reasoning": str,               # raw LLM explanation
            }
        """
        # TODO
        # 1. Format rubric criteria
        rubric_str = "\n".join([f"- {crit}: {desc}" for crit, desc in rubric.items()])

        # 2. Build prompt
        prompt = (
            f"You are an expert AI evaluator. Evaluate the following AI answer based on the rubric.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Rubric Criteria:\n{rubric_str}\n\n"
            f"Provide your evaluation in JSON format with two keys:\n"
            f'1. "scores": a dictionary mapping each criterion name to a float score from 0.0 to 1.0\n'
            f'2. "reasoning": a string explaining your evaluation rationale.\n\n'
            f"JSON Output:"
        )

        # 3. Call judge LLM function
        raw_response = self.judge_llm_fn(prompt)

        # 4. Parse response for scores
        try:
            # Try parsing raw output or extracting JSON block if wrapped in markdown fences
            response_text = raw_response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            data = json.loads(response_text)
            if not isinstance(data, dict):
                raise ValueError("Judge response must be a JSON object")

            nested_scores = data.get("scores")
            scores = nested_scores if isinstance(nested_scores, dict) else data
            reasoning = data.get("reasoning", raw_response)

            # Ensure all criteria from rubric exist in parsed scores
            validated_scores = {}
            for criterion in rubric.keys():
                value = scores.get(criterion)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    validated_scores[criterion] = max(0.0, min(1.0, float(value)))
                else:
                    validated_scores[criterion] = 0.5

            return {
                "scores": validated_scores,
                "reasoning": str(reasoning),
            }

        except Exception:
            # Fallback score of 0.5 for each criterion if JSON parsing fails
            default_scores = {criterion: 0.5 for criterion in rubric.keys()}
            return {
                "scores": default_scores,
                "reasoning": raw_response,
            }

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria

        Args:
            scores_batch: List of score dicts from score_response().

        Returns:
            {
                "positional_bias": bool,
                "leniency_bias":   bool,
                "severity_bias":   bool,
            }
        """
        # TODO
        if not scores_batch:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
            }

        response_averages: list[float] = []
        all_flat_scores: list[float] = []

        for item in scores_batch:
            scores_dict = item.get("scores", {})
            numeric_scores = [
                float(value)
                for value in scores_dict.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if numeric_scores:
                response_averages.append(sum(numeric_scores) / len(numeric_scores))
                all_flat_scores.extend(numeric_scores)

        avg_score = (
            sum(all_flat_scores) / len(all_flat_scores)
            if all_flat_scores
            else None
        )

        # 1. Leniency bias: Average score > 0.8
        leniency_bias = avg_score is not None and avg_score > 0.8

        # 2. Severity bias: Average score < 0.3
        severity_bias = avg_score is not None and avg_score < 0.3

        # 3. Positional bias: Heuristic check if first evaluated criteria/responses score consistently higher
        # In a batch check, verify if earlier items or first-scored items have a noticeably higher average than later ones
        positional_bias = (
            len(response_averages) > 1
            and response_averages[0] > max(response_averages[1:])
        )

        return {
            "positional_bias": positional_bias,
            "leniency_bias": leniency_bias,
            "severity_bias": severity_bias,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.

        Args:
            qa_pairs:   List of QAPair objects.
            agent_fn:   Function str → str (the agent's answer function).
            evaluator:  RAGASEvaluator instance.

        Returns:
            List of EvalResult, one per qa_pair.
        """
        # TODO: for each pair, call agent_fn(pair.question), then run_full_eval.
        # Pass pair.retrieved_contexts as the optional contexts argument and
        # preserve the original pair on the returned EvalResult.
        results: list[EvalResult] = []

        for pair in qa_pairs:
            # 1. Gọi agent để lấy câu trả lời thực tế
            actual_answer = agent_fn(pair.question)

            # 2. Lấy danh sách retrieved_contexts nếu có
            retrieved_ctx = pair.retrieved_contexts

            # 3. Chạy đánh giá đầy đủ qua evaluator
            eval_res = evaluator.run_full_eval(
                answer=actual_answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                contexts=retrieved_ctx,
            )

            # 4. Giữ nguyên qa_pair gốc trên EvalResult
            eval_res.qa_pair = pair
            results.append(eval_res)

        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.

        Returns:
            {
                "total":            int,
                "passed":           int,
                "pass_rate":        float,  # passed / total
                "avg_faithfulness": float,
                "avg_relevance":    float,
                "avg_completeness": float,
                "avg_context_recall": float | None,
                "avg_context_precision": float | None,
                "failure_types":    dict[str, int],  # type → count
            }

        Average only non-None retrieval scores. Return None for a retrieval
        average when no result contains that metric.
        """
        # TODO
        if not results:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "avg_context_recall": None,
                "avg_context_precision": None,
                "failure_types": {},
            }

        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        pass_rate = passed_count / total

        avg_faithfulness = sum(r.faithfulness for r in results) / total
        avg_relevance = sum(r.relevance for r in results) / total
        avg_completeness = sum(r.completeness for r in results) / total

        # Tính trung bình các chỉ số Retrieval (chỉ tính những kết quả khác None)
        recalls = [r.context_recall for r in results if r.context_recall is not None]
        precisions = [r.context_precision for r in results if r.context_precision is not None]

        avg_recall = (sum(recalls) / len(recalls)) if recalls else None
        avg_precision = (sum(precisions) / len(precisions)) if precisions else None

        # Thống kê số lượng từng loại lỗi
        failure_types: dict[str, int] = {}
        for r in results:
            if r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        return {
            "total": total,
            "passed": passed_count,
            "pass_rate": pass_rate,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "avg_completeness": avg_completeness,
            "avg_context_recall": avg_recall,
            "avg_context_precision": avg_precision,
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05 vs baseline.

        Args:
            new_results: List of EvalResult instances (current run)
            baseline_results: List of EvalResult instances (reference/baseline)

        Returns:
            dict with keys:
              - 'new_avg_faithfulness': float
              - 'new_avg_relevance': float
              - 'new_avg_completeness': float
              - 'baseline_avg_faithfulness': float
              - 'baseline_avg_relevance': float
              - 'baseline_avg_completeness': float
              - 'regressions': list[str] — names of metrics that regressed
              - 'passed': bool — True if no regressions

        TODO: Compute avg per metric, compare, list regressions, set passed flag
        """
        # Tính điểm trung bình đợt mới (New)
        n_total = len(new_results) if new_results else 1
        new_faith = sum(r.faithfulness for r in new_results) / n_total
        new_rel = sum(r.relevance for r in new_results) / n_total
        new_comp = sum(r.completeness for r in new_results) / n_total

        # Tính điểm trung bình đợt chuẩn (Baseline)
        b_total = len(baseline_results) if baseline_results else 1
        base_faith = sum(r.faithfulness for r in baseline_results) / b_total
        base_rel = sum(r.relevance for r in baseline_results) / b_total
        base_comp = sum(r.completeness for r in baseline_results) / b_total

        regressions: list[str] = []

        # Kiểm tra tụt điểm > 0.05
        if (base_faith - new_faith) > 0.05:
            regressions.append("faithfulness")
        if (base_rel - new_rel) > 0.05:
            regressions.append("relevance")
        if (base_comp - new_comp) > 0.05:
            regressions.append("completeness")

        return {
            "new_avg_faithfulness": new_faith,
            "new_avg_relevance": new_rel,
            "new_avg_completeness": new_comp,
            "baseline_avg_faithfulness": base_faith,
            "baseline_avg_relevance": base_rel,
            "baseline_avg_completeness": base_comp,
            "regressions": regressions,
            "passed": len(regressions) == 0,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.

        Args:
            results:   Full list of EvalResults.
            threshold: Minimum acceptable score for any metric.

        Returns:
            List of failing EvalResults.
        """
        # TODO
        failing_results: list[EvalResult] = []

        for r in results:
            if (
                r.faithfulness < threshold
                or r.relevance < threshold
                or r.completeness < threshold
            ):
                failing_results.append(r)

        return failing_results


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.

        Returns:
            dict mapping failure_type → count.
            Example: {"hallucination": 3, "irrelevant": 2, "incomplete": 5}
        """
        # TODO
        categories: dict[str, int] = {}
        for f in failures:
            # Lấy failure_type nếu có, mặc định gán là "off_topic" hoặc "unknown" nếu chưa gắn nhãn
            ftype = f.failure_type or "unknown"
            categories[ftype] = categories.get(ftype, 0) + 1
        return categories

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.

        Returns one of these strings based on which score is lowest:
            "Context is missing or irrelevant — improve retrieval"
            "Answer does not address the question — improve prompt clarity"
            "Answer is missing key information — increase context window or improve generation"
            "Multiple issues detected — review full pipeline"
        """
        # TODO: compare faithfulness, relevance, completeness, return appropriate string
        faith = failure.faithfulness
        rel = failure.relevance
        comp = failure.completeness

        # Tìm điểm số thấp nhất để xác định nguyên nhân cốt lõi chính
        min_score = min(faith, rel, comp)

        # Kiểm tra trường hợp nhiều hơn 1 chỉ số bằng min_score hoặc có đa lỗi cùng lúc
        lowest_scores_count = sum(1 for score in (faith, rel, comp) if score == min_score)

        if lowest_scores_count > 1:
            return "Multiple issues detected — review full pipeline"

        if min_score == faith:
            return "Context is missing or irrelevant — improve retrieval"
        elif min_score == rel:
            return "Answer does not address the question — improve prompt clarity"
        else:
            return "Answer is missing key information — increase context window or improve generation"

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |

        Args:
            failures: List of EvalResult instances where passed=False
            suggestions: List of suggestion strings (one per failure, can be shorter list)

        Returns:
            Markdown table string with a row per failure. Status is always "Open".

        TODO: Build markdown table with failure details + matched suggestions
        """
        headers = ["Failure ID", "Type", "Root Cause", "Suggested Fix", "Status"]
        table_lines = [
            f"| {' | '.join(headers)} |",
            f"|{'|'.join(['------------', '------', '------------', '---------------', '--------'])}|",
        ]

        for idx, f in enumerate(failures, start=1):
            failure_id = f"F{idx:03d}"
            ftype = f.failure_type or "unknown"
            root_cause = self.find_root_cause(f)

            # Lấy suggestion tương ứng theo index, nếu thiếu thì gán chuỗi mặc định
            fix = suggestions[idx - 1] if idx - 1 < len(suggestions) else "Review and refine pipeline component"
            status = "Open"

            row = f"| {failure_id} | {ftype} | {root_cause} | {fix} | {status} |"
            table_lines.append(row)

        return "\n".join(table_lines)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Each suggestion should be a concrete, actionable string.

        Examples:
            "Increase chunk size in RAG pipeline to reduce context fragmentation"
            "Add few-shot examples showing complete answers to improve completeness"
            "Implement hallucination checker to filter unsupported claims"

        Returns:
            List of at least 3 suggestion strings (or fewer if failures is empty).
        """
        # TODO: analyze categorized failures and return suggestions
        if not failures:
            return []

        categories = self.categorize_failures(failures)
        suggestions: list[str] = []

        # 1. Phân tích lỗi Hallucination / Faithfulness
        if categories.get("hallucination", 0) > 0:
            suggestions.append("Implement hallucination checker to filter unsupported claims")

        # 2. Phân tích lỗi Irrelevant / Off-topic / Prompt clarity
        if categories.get("irrelevant", 0) > 0 or categories.get("off_topic", 0) > 0:
            suggestions.append("Improve prompt clarity and refine system instructions to align with question intent")

        # 3. Phân tích lỗi Incomplete / Context window
        if categories.get("incomplete", 0) > 0:
            suggestions.append("Add few-shot examples showing complete answers to improve completeness")
            suggestions.append("Increase chunk size in RAG pipeline to reduce context fragmentation")

        # Đảm bảo có ít nhất 3 gợi ý nếu tập lỗi không rỗng
        default_pool = [
            "Increase chunk size in RAG pipeline to reduce context fragmentation",
            "Add few-shot examples showing complete answers to improve completeness",
            "Implement hallucination checker to filter unsupported claims",
        ]

        for default_sugg in default_pool:
            if len(suggestions) >= 3:
                break
            if default_sugg not in suggestions:
                suggestions.append(default_sugg)

        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
