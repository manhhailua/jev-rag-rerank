# Jev RAG Rerank — chấm relevance + benchmark latency A/B

Dùng model **Jev** (TypeSafe AI, "System One" model) chấm relevance từng chunk trong `top_k`
của retriever, rồi shortlist để **giảm noise, tăng relevancy**. Kèm benchmark 2 chiến lược gọi API.

## Mục tiêu

- Thử primitive **Noul** (yes/no → probability ∈ [0,1]) làm relevance score.
- Pipeline: `retrieve top_k (giả lập)` → `Jev chấm relevance` → `shortlist` → `metric trước/sau`.
- Benchmark A/B latency: **fan-out per-pair** vs **single-call multi-question**.
- Real mode (API thật) + mock mode (fallback thiếu key).

## Cách chạy

```bash
# real mode: key đọc từ ~/.config/env (dòng JEV_API_KEY=...)
uv run --with jupyter jupyter nbconvert --to notebook --execute \
  jev_rag_rerank.ipynb --output /tmp/executed.ipynb

# mở notebook tương tác
uv run --with jupyter jupyter notebook jev_rag_rerank.ipynb
```

Notebook đọc key: `~/.config/env` (ưu tiên) → env var `JEV_API_KEY`. Không hardcode.

## API shape Jev (đã xác minh từ docs chính thức)

- Endpoint: `POST https://api.typesafe.ai/v1/systemone` (KHÔNG phải `jevtypesafe.org`).
- Auth: `Authorization: Bearer <API_KEY>` (key `apikey_…`).
- Request (**`model` bắt buộc**): `{"state": <string|object|array>, "model": "jev-latest", "questions": {<tên>: {"type":"noul","instructions":"..."}}}`.
- Noul response: `answers.<tên>.noul` = probability [0,1].
- **Parallel questions**: mọi câu hỏi 1 request đánh giá song song; reference field `state` bằng backtick path (`` `candidates.c01` ``).
- **Không có batch endpoint** → rerank bằng fan-out hoặc multi-question.
- Error: 401/422/429/529.

Nguồn: https://docs.typesafe.ai/api.md · /cookbooks/parallel_questions.md · /patterns/fan-out.md · /cookbooks/rerank_typesafe.md

## Kết quả benchmark (real mode, key từ ~/.config/env)

### Rerank quality (top_k=10, 6 liên quan / 4 noise)

| Giai đoạn | n | noise_ratio | precision | recall |
|---|---|---|---|---|
| BEFORE (top_k theo score) | 10 | 0.4 | 0.6 | 1.0 |
| AFTER (ngưỡng prob ≥ 0.5) | 6 | 0.0 | **1.0** | 1.0 |
| AFTER (top_n = 5) | 5 | 0.0 | **1.0** | 0.83 |

Jev tách rõ: chunk liên quan 0.77–0.98, noise 0.00–0.01.

### Latency A/B

| variant | calls | wall_s | latency/call | in_tokens | cost | retries |
|---|---|---|---|---|---|---|
| fan-out w=1 | 10 | 3.35 | 335ms | 3290 | $0.00014 | 0 |
| fan-out w=4 | 10 | 1.03 | 103ms | 3290 | $0.00014 | 0 |
| fan-out w=8 | 10 | 0.66 | 66ms | 3290 | $0.00014 | 0 |
| fan-out w=16 | 10 | 0.37 | 37ms | 3290 | $0.00014 | 0 |
| **single-call n=10** | **1** | **0.32** | — | **890** | **$0.00004** | 0 |
| single-call n=16 | 1 | 0.31 | — | 1226 | $0.00005 | 0 |
| single-call n=30 | 1 | 0.35 | — | 2010 | $0.00008 | 0 |
| single-call n=50 | 1 | 0.37 | — | 3130 | $0.00013 | 0 |

### Accuracy giữ nguyên

Single-call vs fan-out: max |Δnoul| = 0.15 (khác nhẹ do state khác cấu trúc), nhưng
**shortlist ra cùng kết quả** (precision 1.0, recall 1.0, noise 0).

## Khuyến nghị

- **Dùng single-call multi-question** cho rerank: 1 call ~0.32s bất kể 10–50 candidate,
  rẻ ~3–4x token (query chỉ gửi 1 lần), accuracy không đổi. Đây là variant latency thấp nhất.
- Fan-out chỉ đáng khi cần hỏi **nhiều câu khác nhau** mỗi cặp, hoặc cần retry độc lập từng cặp.
- Giới hạn single-call: `state` ~8000 ký tự → không scale vô hạn (50 candidate vẫn ổn).

## Giới hạn

- Mock mode là heuristic lexical, không phản ánh Jev thật.
- Ground truth nhỏ (6/16 chunk); latency có nhiễu giữa các lần đo.
- Chưa gặp 429 (retry backoff đã cài sẵn nhưng chưa kích hoạt).
- Cost dùng giá $0.042/M input (tham khảo từ cookbook).

## Bước tiếp theo

1. Đo single-call ở 100–200 candidate tìm ngưỡng thật của giới hạn state/token.
2. So Jev rerank vs cross-encoder (`bge-reranker`) cùng dataset.
3. Thử primitive `score` (thang 0..5) thay `noul` để phân cấp relevance mịn hơn.
