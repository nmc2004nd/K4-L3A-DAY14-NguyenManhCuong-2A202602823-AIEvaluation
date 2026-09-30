# Day 14 — Exercises

## AI Evaluation & Benchmarking · Lab Worksheet

**Thời gian làm bài:** 14:15–17:00

**Domain:** OrbitTech Store Customer Support

Điền trực tiếp câu trả lời vào file này. Golden dataset 20 QA được viết một lần
duy nhất trong `golden_dataset.json`, không chép lại toàn bộ vào Markdown.

---

Từ 14:15–14:30, cài môi trường và chạy baseline tests theo `guide_lab.md`.

---

## Part 1 — Warm-up (14:30–14:45)

### Exercise 1.1 — RAGAS Metric Thresholds

Theo bài giảng:

- 0.8–1.0: Good — monitor, maintain.
- 0.6–0.8: Needs work — analyze failures, iterate.
- Dưới 0.6: Significant issues — investigate.

Với từng metric, xác định khi nào score thấp có thể chấp nhận và khi nào là
critical.

| Metric | Acceptable Low Score Scenario | Critical Low Score Scenario | Action Required |
|---|---|---|---|
| Faithfulness - Độ trung thực | Các ứng dụng sáng tạo(sáng tác thơ, sáng tác nhạc, hội họa,...) | Q&A trên các tài liệu yêu cầu độ chính xác tuyệt đối, không được bịa đặt như: Luật pháp, y tế, Chính sách,.., tuy nhiên model lại bịa hoặc trả lời không chính xác | Yêu cầu model dự đoán theo greedy - chọn token có scored max - temp = 0 |
| Answer Relevance - Độ liên quan của câu trả lời | Khi người dùng đưa ra câu hỏi quá ngắn, mơ hồ chung chung -> hệ thống có thể hỏi lại hoặc trả lời chung chung mang tính định hướng | Truy vấn rõ ràng, tuy nhiên câu trả lời lan man, không đúng trọng tâm, không giải quyết được vấn đề người dùng | Cải thiện lại prompt tập chung vào nhận dạng intent user hoặc có thể few shot một vài example |
| Context Recall - Độ đầy đủ của truy xuất | User hỏi câu hỏi ngoài domain trong knowledge base -> model return về xin lồi vì ngoài phạm vi trả lời | Question của user liên quan đến 3 tài liệu khác nhau nhưng retrive chỉ trả về 1 tài liệu -> thiếu context | Tăng số lượng k, hoặc đổi stragery chunk, hoặc sử dụng hybird search - Dense and Sparse |
| Context Precision - Độ chính xác của truy xuất | Nếu retrive trả về 5 chunk, và context cần truy xuất nằm trong chunk 4, 5 thì có thể chấp nhận được | Retrive trả về 5 chunk nhưng context cần truy vấn không nằm trong 5 chunk - thiếu context trả lời có thể trả lời sai | Áp dụng reranker sau retrive hoặc cải thiện filter theo metadata để truy vấn tốt hơn |
| Completeness - Độ hoàn thiện của câu trả lời | Trả lời cho các câu hỏi mở chung chung | Tổng quan thái quát khiến mất đi các khâu quan trọng  | Tinh chỉnh lại promt thêm cac rule yêu cầu model trả lời một cách đầy đủ   |

### Exercise 1.2 — Bias trong LLM-as-a-Judge

Ba bias thường gặp:

- Position bias: judge ưu tiên answer xuất hiện trước.
- Verbosity bias: judge ưu tiên answer dài hơn.
- Self-preference: judge ưu tiên output giống chính model đó.

**Câu 1: Thiết kế experiment phát hiện position bias với ít nhất hai conditions.**

> *Câu trả lời:* - Cho cùng một câu hỏi với lượt đánh giá [A, B], lượt 2 đảo lại [B, A] -> Nếu đáp án nào đứng trước cũng thắng thì position bias do LLM - transformer chú ý nhiều đến các thông tin ở đầu và cuối - lost mid

**Câu 2: Làm thế nào giảm verbosity bias bằng rubric design?**

> *Câu trả lời:* Verbosity bias là model thường bị đánh lừa bởi các câu viết dài , viết hoa văn mặc cho chất lượng nội dung trong đó thấp -> Khắc phụ trong Rubric design thiết kê rule phạt các câu trả lời lan man không đúng trọng tâm, ưu tiên trả lời cốt lõi, đúng vấn đề 

**Câu 3: Tại sao cần calibrate LLM judge với human labels?**

> *Câu trả lời:* Model không có trải nghiệm thực tế của con người, model có thể chấm cao cho câu trả lời nghe rất hay và mượt nhưng với con người thì nó lại sai hoặc chưa hoàn toàn đúng

### Exercise 1.3 — Evaluation trong CI/CD

**Câu 1: Chọn threshold để block deployment.**

| Metric | Threshold | Lý do |
|---|---:|---|
| Faithfulness | > 0.85 | Tính chính xác quyết định sự sống còn của hệ thống RAG, người dùng thà nghe câu xin lỗi vì không có thông tin còn hơn là nghe câu trả lời bịa không đúng |
| Answer Relevance | >0.8 | Tập trung vào vấn đề chính, tránh lan man dài dòng |
| Completeness | >0.7 | Một câu trả lời có nội dung liên quan 80% context cần truy vấn vẫn mang lại phần nào giá trị cho người dùng, thiếu một vài ý nhỏ không gây nguy hiểm bằng bịa sai sự thật |

**Câu 2: Khi nào dùng offline evaluation, online evaluation và human review?**

> *Câu trả lời:* Dùng offline trong giai đoạn development, thử nghiệm. Dùng online trong giai đoạn product có người dùng hằng ngày. Human review trong giai đoạn xây dụng golden Dataset, Calib LLM as a judge, Kiểm tra mẫu với các trường hợp được gắn cờ Low score

---

## Part 2 — Core Coding (14:45–15:40)

Hoàn thiện các TODO bắt buộc trong `template.py`.

### Task 1 — Data Models

- `QAPair`: question, expected answer, gold context, metadata và retrieved contexts.
- `EvalResult`: answer-side scores, optional retrieval scores, pass/failure fields.
- `overall_score()`: trung bình Faithfulness, Relevance và Completeness.

### Task 2 — RAGASEvaluator

Answer-side:

- `evaluate_faithfulness(answer, context)`
- `evaluate_relevance(answer, question)`
- `evaluate_completeness(answer, expected)`

Retrieval-side:

- `evaluate_context_recall(contexts, expected)`
- `evaluate_context_precision(contexts, expected)`

Full pipeline:

- `run_full_eval(..., contexts=None)` luôn tính ba answer metrics.
- Nếu có `contexts`, tính và lưu thêm Context Recall và Context Precision.
- Retrieval scores không làm thay đổi `overall_score()` và pass rule gốc.

### Task 3 — LLMJudge

- `score_response(question, answer, rubric)`
- `detect_bias(scores_batch)`

### Task 4 — BenchmarkRunner

- `run(qa_pairs, agent_fn, evaluator)`
- `generate_report(results)`
- `run_regression(new_results, baseline_results)`
- `identify_failures(results, threshold)`

`BenchmarkRunner.run()` phải truyền `pair.retrieved_contexts` vào
`run_full_eval()`. Report phải có average của hai retrieval metrics.

### Task 5 — FailureAnalyzer

- `categorize_failures(failures)`
- `find_root_cause(failure)`
- `generate_improvement_suggestions(failures)`
- `generate_improvement_log(failures, suggestions)`

Kiểm tra:

```bash
pytest tests/ -v
```

`rerank_by_overlap()` là TODO bonus của Exercise 3.5. Test tương ứng được skip
nếu bạn chưa làm bonus.

---

## Part 3 — Golden Dataset & Real Benchmark (15:40–16:35)

### Exercise 3.1 — Build the Golden Dataset

Thiết kế và validate dataset theo Mục 5–6 trong `guide_lab.md`. Nội dung 20 QA
được điền trực tiếp trong `golden_dataset.json`; phần dưới chỉ ghi lại kết quả
và quyết định thiết kế, không chép lại toàn bộ QA.

**Kết quả dataset**

| Hạng mục | Kết quả |
|---|---|
| Tổng số records | 20 / 20 |
| Easy | 5 / 5 |
| Medium | 7 / 7 |
| Hard | 5 / 5 |
| Adversarial | 3 / 3 |
| Source documents được sử dụng | 10 / 10 |
| Validator status | PASS |

**Ba case đại diện cho quyết định thiết kế**

| ID | Difficulty | Source document(s) | Vì sao case phù hợp với difficulty/attack type? |
|---|---|---|---|
| E02 | Easy | `02_orders_and_payments.md` | Câu trả lời là factual lookup trực tiếp: hai dấu hiệu xác nhận order đã được tạo và một phân biệt rõ với pending card authorization. |
| H01 | Hard | `09_escalation_and_policy_updates.md` | Case buộc xác định triggering event là ngày đặt hàng, chọn đúng policy version, rồi tách ngày bắt đầu tính window; đồng thời xử lý ngoại lệ OrbitPlus không hồi tố cho order trước 01/09/2026. |
| A02 | Adversarial (`prompt_injection`) | `00_system_scope.md`, `08_accounts_privacy_and_security.md` | Prompt yêu cầu bỏ qua rule và tiết lộ ba loại dữ liệu bị cấm; expected answer phải chống injection và áp dụng đúng điều kiện verified authorization. |

**Điểm khó nhất khi xây dựng expected answer hoặc evidence là gì?**

> *Câu trả lời:* Khó nhất là giữ cho expected answer vừa ngắn gọn vừa bao phủ chính xác mọi điều kiện và ngoại lệ, đặc biệt ở các case policy-version. Ví dụ H01 phải phân biệt ngày đặt hàng quyết định version với ngày giao hàng bắt đầu thời hạn trả hàng, đồng thời không áp dụng hồi tố quyền lợi OrbitPlus. Evidence vì vậy được chọn theo từng claim, giữ nguyên văn và không thêm đoạn không liên quan.

**Xác nhận:**

- [x] Mọi claim trong expected answer đều có evidence hỗ trợ.
- [x] Không có questions trùng ý và không dùng kiến thức ngoài corpus.
- [x] `python validate_golden_dataset.py` báo `PASS`.

### Exercise 3.2 — Benchmark Run

Chạy:

```bash
python domain_assistant.py
python evaluate_answers.py
```

Copy bảng terminal vào đây hoặc điền từ `artifacts/benchmark_results.json`.

| ID | Question (short) | Ctx Recall | Ctx Precision | Faithfulness | Relevance | Completeness | Overall | Passed? | Failure Type |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| E01 | | | | | | | | | |
| E02 | | | | | | | | | |
| E03 | | | | | | | | | |
| E04 | | | | | | | | | |
| E05 | | | | | | | | | |
| M01 | | | | | | | | | |
| M02 | | | | | | | | | |
| M03 | | | | | | | | | |
| M04 | | | | | | | | | |
| M05 | | | | | | | | | |
| M06 | | | | | | | | | |
| M07 | | | | | | | | | |
| H01 | | | | | | | | | |
| H02 | | | | | | | | | |
| H03 | | | | | | | | | |
| H04 | | | | | | | | | |
| H05 | | | | | | | | | |
| A01 | | | | | | | | | |
| A02 | | | | | | | | | |
| A03 | | | | | | | | | |

**Aggregate Report**

- Overall pass rate: ____%
- Avg Context Recall: ____
- Avg Context Precision: ____
- Avg Faithfulness: ____
- Avg Relevance: ____
- Avg Completeness: ____
- Failure type distribution: ____

**Ba cases có Overall Score thấp nhất**

1. ID: ____ | Score: ____ | Failure type: ____
2. ID: ____ | Score: ____ | Failure type: ____
3. ID: ____ | Score: ____ | Failure type: ____

**Nhận xét ngắn:** Metric nào yếu nhất? Kết quả gợi ý vấn đề nằm ở retrieval
hay generation?

> *Câu trả lời:*

### Exercise 3.3 — LLM-as-a-Judge Rubric Design

Thiết kế rubric domain-specific cho OrbitTech Customer Support. Mỗi mức phải
đủ cụ thể để hai người chấm độc lập có thể hiểu giống nhau.

Chọn 3–5 dimensions:

- [ ] Correctness
- [ ] Completeness
- [ ] Relevance
- [ ] Evidence/citation
- [ ] Actionability
- [ ] Safety/privacy
- [ ] Tone/clarity
- [ ] Dimension khác: __________

| Score | Tiêu chí domain-specific | Ví dụ response |
|---:|---|---|
| 5 | | |
| 4 | | |
| 3 | | |
| 2 | | |
| 1 | | |

**Ba edge cases khó chấm**

| Edge Case | Tại sao khó chấm? | Rubric xử lý thế nào? |
|---|---|---|
| | | |
| | | |
| | | |

**Bias controls:** Rubric hoặc evaluation protocol của bạn giảm position bias,
verbosity bias và self-preference bằng cách nào?

> *Câu trả lời:*

### Exercise 3.4 — Framework Comparison (Bonus +5)

Chỉ làm sau khi hoàn thành 3.1–3.3. Chọn hai framework trong RAGAS, DeepEval
và TruLens; chạy hoặc thiết kế một so sánh có cùng input dataset.

| Tiêu chí | Framework 1: ____ | Framework 2: ____ |
|---|---|---|
| Setup complexity | | |
| Metrics available | | |
| CI/CD integration | | |
| Kết quả trên cùng dataset | | |
| Insight rút ra | | |

- Scores có nhất quán không?
- Framework nào strict hơn và vì sao?
- Hai framework có tìm ra cùng failure cases không?

> *Phân tích:*

### Exercise 3.5 — Retrieval Reranking (Bonus +5)

Mục tiêu: kiểm tra việc đổi thứ tự chunks có tăng Context Precision mà không
thay đổi Context Recall hay không.

1. Chọn ít nhất 5 cases từ `artifacts/actual_answers.json`.
2. Tính Context Recall và Context Precision trước rerank.
3. Implement `rerank_by_overlap()` hoặc một reranker khác.
4. Rerank cùng tập chunks, không thêm hoặc xóa chunk.
5. Tính lại hai metrics và giải thích kết quả.

| ID | Recall before | Recall after | Precision before | Precision after | Delta Precision |
|---|---:|---:|---:|---:|---:|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| **Avg** | | | | | |

**Tại sao Recall dự kiến không đổi?**

> *Câu trả lời:*

**Khi nào reranking không đủ và cần sửa retriever/query/chunking?**

> *Câu trả lời:*

---

## Part 4 — Reflection (16:35–16:50)

Hoàn thành `reflection.md` bằng kết quả thật từ Exercise 3.2.

---

## Completion Checklist

Hoàn thành kiểm tra cuối trong khoảng 16:50–17:00.

- [ ] Tất cả required tests pass.
- [ ] `golden_dataset.json` validate thành công.
- [ ] Exercise 3.1 hoàn thành trong file JSON và bảng kết quả phía trên.
- [ ] Exercise 3.2 có năm metrics, aggregate report và ba cases thấp nhất.
- [ ] Exercise 3.3 có rubric 1–5 và bias controls.
- [ ] `reflection.md` có ba failure analyses và regression strategy.
- [ ] Đã copy `template.py` thành `solution/solution.py`.
- [ ] Exercise 3.4 và 3.5 chỉ làm nếu chọn bonus.
