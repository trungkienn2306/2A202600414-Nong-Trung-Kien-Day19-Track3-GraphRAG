# Báo cáo Benchmark Lab 19 (Bản chính thức)

## 1) Thiết lập thí nghiệm

- Corpus: 10 bài Wikipedia về các công ty AI.
- Bộ câu hỏi:
  - test: `data/multi-hops-question/test-multi-hops-question.json`
  - full: `data/multi-hops-question/multi-hops-question.json`
- Mô hình sử dụng:
  - LLM: `gpt-4o-mini`
  - Embedding: `text-embedding-3-small`
- Cấu hình GraphRAG đã so sánh:
  - `top_k_seeds = 2`
  - `top_k_seeds = 3`
  - `top_k_seeds = 5`
- Baseline: FlatRAG chạy cùng bộ câu hỏi và cùng cơ chế LLM judge.

## 2) Kết quả benchmark theo seed

| Run | Bộ câu hỏi | Seeds | Accuracy GraphRAG (%) | Accuracy FlatRAG (%) | Delta (%) | Multi-hop GraphRAG (%) | Multi-hop FlatRAG (%) | Latency TB GraphRAG (s) | Latency TB FlatRAG (s) | Cost GraphRAG ($) | Cost FlatRAG ($) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seeds2_test | test | 2 | 20.0 | 40.0 | -20.0 | 20.0 | 40.0 | 2.108 | 2.130 | 0.000623 | 0.000993 |
| seeds3_test | test | 3 | 40.0 | 40.0 | +0.0 | 40.0 | 40.0 | 2.276 | 2.238 | 0.000808 | 0.001010 |
| seeds5_test | test | 5 | 40.0 | 60.0 | -20.0 | 40.0 | 60.0 | 2.932 | 2.097 | 0.001096 | 0.000985 |
| seeds3_full | full | 3 | 40.0 | 55.0 | -15.0 | 40.0 | 55.0 | 5.557 | 2.097 | 0.003014 | 0.003870 |
| seeds3_full_v2 | full_v2 | 3 | 20.0 | 40.0 | -20.0 | 20.0 | 40.0 | 2.192 | 1.942 | 0.003787 | 0.004535 |
| smoke_bal_v2 | v2_smoke(8) | 3 | 25.0 | 75.0 | -50.0 | 25.0 | 75.0 | 2.850 | 2.470 | 0.001966 | 0.001756 |
| seeds3_full_bal_v3 | full_v2_balanced | 3 | 10.0 | 45.0 | -35.0 | 10.0 | 45.0 | 2.881 | 1.911 | 0.004601 | 0.004537 |

Nguồn dữ liệu:

- `report/benchmark_runs/<run_name>/summary.json`
- `report/benchmark_runs/<run_name>/benchmark_details.csv`

## 2.1) Pha nâng cấp v2 (benchmark/judge + seed filtering + bộ câu hỏi mới)

Các thay đổi đã triển khai đúng theo mục ưu tiên:

- Nâng chuẩn judge trong `src/benchmark.py`:
  - Chấm nghiêm hơn theo quy tắc bắt buộc entity/fact cốt lõi.
  - Hỗ trợ metadata v2 (`hops`, `chain`) để judge hiểu rõ chuỗi multi-hop kỳ vọng.
  - Lưu thêm trường `*_missing_facts` để phân tích failure mode chi tiết.
- Lọc seed trong `src/graphrag.py`:
  - Loại seed generic/mơ hồ.
  - Khử trùng lặp alias thô theo canonical key.
  - Có fallback nạp thêm seed khi lọc quá mạnh.
- Chạy benchmark v2:
  - Command: `python scripts/05_benchmark.py --questions-file data/multi-hops-question/multi-hops-question-v2.json --seeds 3 --run-name seeds3_full_v2`
  - Artifact:
    - `report/benchmark_runs/seeds3_full_v2/summary.json`
    - `report/benchmark_runs/seeds3_full_v2/benchmark_details.csv`
    - `report/benchmark_runs/seeds3_full_v2/run_metadata.json`
- Pha cân bằng (Accuracy-first but bounded):
  - Đã thêm smoke gate (`--smoke`) và ngưỡng efficiency (`--max-latency-multiplier`, `--max-cost-multiplier`) trong `scripts/05_benchmark.py`.
  - Đã chạy:
    - `smoke_bal_v2` (8 câu khó): gate fail ở accuracy.
    - `seeds3_full_bal_v3` (full v2 sau tối ưu KG/retrieval): vẫn chưa vượt FlatRAG.
  - Vì smoke gate fail, tạm **không sweep seeds=2/5** để tránh tăng token khi cấu hình chưa đạt ngưỡng.

Đoạn mô tả cá nhân hóa theo yêu cầu:

> Em: Nông Trung Kiên - 2A202600414. Sau khi chạy thử bộ câu hỏi test 1, em phát hiện bộ cũ có chất lượng multi-hop chưa tốt (Multi-hop thực sự: 7/20, giả multi-hop: 10/20, thực chất 1-hop: 3/20). Vì vậy em chủ động tạo bộ câu hỏi `multi-hops-question-v2.json` có chuỗi hop rõ hơn và train/benchmark lại để phản ánh đúng năng lực GraphRAG.

## 3) Phân tích các kiểu lỗi (Failure Modes)

### FM1 — Thiếu bằng chứng ở bước retrieval

- Triệu chứng: câu trả lời đúng một phần nhưng thiếu fact bắt buộc.
- Ví dụ: `Q9`, `Q11`.
- Nhận định: GraphRAG lấy được ngữ cảnh liên quan nhưng chưa gom đủ bằng chứng ở hop thứ hai.
- Hướng xử lý: lọc seed tốt hơn (loại thực thể quá chung chung), tăng chuẩn hóa tên thực thể trước BFS.

### FM2 — Câu multi-hop bị trả lời kiểu single-hop

- Triệu chứng: chỉ trả lời được một nửa câu hỏi.
- Ví dụ: `Q6`, `Q7`, `Q14`.
- Nhận định: liên kết xuyên tài liệu chưa được mã hóa đủ rõ trong tập triple hiện tại.
- Hướng xử lý: tăng chất lượng trích xuất quan hệ ở bước extraction, siết danh sách predicate quan trọng.

### FM3 — Nhầm thực thể/quan hệ

- Triệu chứng: cùng chủ đề nhưng sai entity đích.
- Ví dụ: `Q12` (xAI bị trượt thành Neuralink), `Q20` (judge chấm lệch hai hệ).
- Nhận định: context nhiễu + tiêu chí judge chưa chặt ở các thực thể bắt buộc.
- Hướng xử lý: chỉnh prompt judge theo quy tắc “thiếu entity bắt buộc => fail”.

### FM4 (mới ở v2) — GraphRAG mất đích ở hop cuối dù seed đã sạch hơn

- Triệu chứng: có context liên quan nhưng câu trả lời dừng ở thực thể trung gian, không chạm entity đích.
- Ví dụ: `Q2`, `Q3`, `Q4`, `Q6`, `Q11`, `Q16` (FlatRAG pass, GraphRAG fail).
- Nhận định: bước duyệt graph và tổ chức context hiện tại chưa đủ mạnh để "khóa" đáp án cuối ở các chuỗi 2-3 hop có nhiều thực thể cùng chủ đề.
- Hướng xử lý tiếp theo: ưu tiên relation weighting + rerank subgraph theo signal của `hops/chain` (không chỉ theo seed similarity).

### FM5 (mới ở v2) — Judge strict làm lộ lỗi thiếu fact nhưng cũng giảm tỷ lệ pass tổng

- Triệu chứng: nhiều câu bị fail do thiếu 1 entity bắt buộc dù ý chính gần đúng.
- Nhận định: đây là hiệu ứng mong đợi sau khi siết benchmark (giảm false-pass), giúp đo chính xác hơn.
- Hướng xử lý: tối ưu answer synthesis để luôn xuất đủ cặp entity/quan hệ bắt buộc trong ground-truth.

### FM6 (pha cân bằng) — Trượt semantic type ở đích trả lời

- Triệu chứng: mô hình trả đúng bối cảnh nhưng sai loại đích cần hỏi (ví dụ trả công ty thay vì product line, hoặc trả tên rút gọn).
- Ví dụ: `Q4` (trả `Mistral` thay vì `Mistral AI`), `Q6` (trả `Anthropic` thay vì `Claude`), `Q11` (trả `LP` thay vì `PBC`).
- Nhận định: context graph đã có tín hiệu nhưng prompt synthesis chưa “khóa” output type theo câu hỏi.
- Hướng xử lý tiếp theo: thêm answer-type constraint (YEAR/ORG/PRODUCT/LEGAL_FORM) + post-check theo regex/type rule trước khi finalize.

## 4) Đối chiếu bộ câu hỏi với `data/raw`

Kết luận: bộ câu hỏi không sai hoàn toàn, nhưng có một số câu khiến kết quả benchmark bị méo do diễn đạt và tiêu chí chấm.

- `Q6`: yêu cầu tổng hợp 3 công ty có DeepMind ties; mô hình thường thiếu một công ty dù dữ liệu có trong `data/raw`.
- `Q12`: ground-truth là `xAI`, nhưng mô hình dễ trượt sang `Neuralink`.
- `Q20`: có trường hợp FlatRAG được judge pass với câu trả lời chưa đủ chặt.

=> Chênh lệch hiện tại không chỉ do thuật toán GraphRAG, mà còn do thiết kế benchmark + độ chặt của judge.

## 5) Kết luận mục tiêu “vượt +20%”

Mục tiêu bài lab: `GraphRAG accuracy - FlatRAG accuracy >= +20%`.

- Delta tốt nhất trên bộ test: `0.0%` (`seeds=3`).
- Delta trên full run (`seeds3_full`): `-15.0%`.
- Delta trên full_v2 (`seeds3_full_v2`): `-20.0%`.
- Delta trên full_v2_balanced (`seeds3_full_bal_v3`): `-35.0%`.
- Kết luận hiện tại: **CHƯA ĐẠT MỤC TIÊU +20%**.

Kết luận bổ sung cho bài lab:

- Với phạm vi chỉ `10 Wikipedia company`, dữ liệu hiện tại **chưa đủ để chứng minh GraphRAG có benchmark cao hơn FlatRAG**.
- Nếu có thêm thời gian và hạ tầng (bao gồm token/cost), có thể mở rộng dữ liệu và tối ưu thêm để đưa ra benchmark tốt hơn.

Bằng chứng:

- Run artifacts: `report/benchmark_runs/seeds3_full/`
- Summary: `report/benchmark_runs/seeds3_full/summary.json`
- Head-to-head: FlatRAG đang thắng ở `Q9`, `Q11`, `Q20`.
- Run artifacts v2: `report/benchmark_runs/seeds3_full_v2/`
- Summary v2: `report/benchmark_runs/seeds3_full_v2/summary.json`
- Head-to-head v2: GraphRAG win `Q12`, `Q13`; FlatRAG win `Q2`, `Q3`, `Q4`, `Q6`, `Q11`, `Q16`.
- Run artifacts smoke cân bằng: `report/benchmark_runs/smoke_bal_v2/`
- Run artifacts full cân bằng: `report/benchmark_runs/seeds3_full_bal_v3/`
- Nhận định pha cân bằng hiện tại: efficiency giữ gần baseline nhưng accuracy giảm, cần thêm vòng tối ưu answer-type.

## 6) Đề xuất thay đổi cụ thể để kéo GraphRAG vượt FlatRAG

### Ưu tiên A — Chuẩn hóa benchmark

1. Rà soát lại 20 câu để đảm bảo mỗi câu thực sự cần multi-hop và có đủ evidence trong `data/raw`.
2. Viết lại ground-truth theo kiểu bắt buộc entity (ví dụ Q12 phải có `xAI`).
3. Siết prompt judge trong `src/benchmark.py` để tránh pass “lỏng”.

### Ưu tiên B — Nâng chất lượng Knowledge Graph

1. Tăng canonicalization thực thể ở `src/extract_triples.py` (gom alias cùng thực thể).
2. Giảm thực thể generic làm nhiễu graph.
3. Ưu tiên relation có giá trị suy luận (`FOUNDED_BY`, `INVESTED_IN`, `ACQUIRED`, `PARTNERED_WITH`).

### Ưu tiên C — Nâng chất lượng retrieval của GraphRAG

1. Lọc seed entity trong `src/graphrag.py` (loại seed mơ hồ).
2. Tối ưu BFS theo trọng số quan hệ.
3. Tổ chức context theo block fact thay vì chuỗi phẳng để LLM dễ bắc cầu.

### Ưu tiên D — Quy trình đánh giá

1. Chạy lại ablation seeds trên full set (`2/3/5`) sau khi chỉnh chất lượng dữ liệu.
2. Lưu artifact theo run để so sánh trực tiếp trước/sau cải tiến.
3. Chốt cấu hình theo thứ tự ưu tiên: Accuracy -> Latency -> Cost.

## 7) Khuyến nghị cấu hình hiện tại

- Seed tạm khuyến nghị: `top_k_seeds = 3`.
- Lý do: cân bằng nhất ở giai đoạn hiện tại, không tăng latency/cost quá mạnh như `seeds=5`.

## 8) Minh chứng Neo4j Visualization (phần nộp bắt buộc)

- Neo4j chạy qua Docker Compose (`neo4j:5.23`, cổng `7474/7687`).
- Endpoint: `http://localhost:7474`
- Cypher kiểm chứng:
  - `MATCH (n:Entity)-[r]->(m) RETURN n,r,m LIMIT 100`
  - `MATCH p=(n:Entity {name:'OpenAI'})-[*1..2]-(m) RETURN p LIMIT 80`
  - `MATCH (s:Entity)-[r:RELATION {type:'INVESTED_IN'}]->(o) RETURN s.name, o.name`
- Ảnh minh chứng:
  - `visualizations/graph_full.png`
