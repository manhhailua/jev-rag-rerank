# Jev RAG Rerank — chấm relevance + benchmark A/B (data 512-token)

Dùng model **Jev** (TypeSafe AI, "System One") chấm relevance từng chunk trong `top_k` rồi
shortlist để giảm noise. Kèm benchmark 2 chiến lược gọi API trên chunk chuẩn RAG (~512 tokens).

## Mục tiêu

- Primitive **Noul** (yes/no → probability) làm relevance score cho re-rank.
- Pipeline: retriever (lexical, giả lập BM25) → `top_k` → Jev chấm relevance → shortlist.
- Benchmark A/B: **fan-out per-pair** vs **single-call multi-question**.
- Real mode (API thật) + mock mode (fallback thiếu key).

## Cấu trúc repo

```
data/sample_inputs.json   # 20 queries + 40 chunks (~512 tokens) + ground truth (committed)
make_data.py              # generator data (uv run --with tiktoken python3 make_data.py)
build_nb.py               # sinh jev_rag_rerank.ipynb
jev_rag_rerank.ipynb      # notebook chính
```

## Data — `data/sample_inputs.json`

- **20 queries**, mỗi câu **≤ 20 tokens** (min 8, max 20, avg 13.7 — đo bằng tiktoken).
- **40 chunks**, mỗi chunk **~512 tokens** (min 434, max 722, avg 536 — đo bằng tiktoken `cl100k_base`).
- **Ground truth**: `query.relevant_chunks` (mỗi query 2 chunk liên quan).
- Text tiếng Việt kỹ thuật mạch lạc, 20 chủ đề đa dạng (SQL, Raft, Docker, K8s, TLS, ML, JWT, OWASP, ...).

## Cách chạy

```bash
# real mode: key đọc từ ~/.config/env (dòng JEV_API_KEY=...)
uv run --with jupyter jupyter nbconvert --to notebook --execute \
  jev_rag_rerank.ipynb --output /tmp/executed.ipynb

# sinh lại data (cần tiktoken)
uv run --with tiktoken python3 make_data.py
```

Notebook đọc key: `~/.config/env` (ưu tiên) → env var `JEV_API_KEY`. Không hardcode.

## API shape Jev (đã xác minh từ docs chính thức)

- Endpoint: `POST https://api.typesafe.ai/v1/systemone`; auth `Bearer <apikey_…>`.
- Request: `{"state", "model", "questions"}` — `model` bắt buộc (`jev-latest` → `jev-1.13.0`).
- Noul: `answers.<tên>.noul` = probability [0,1]. Parallel questions: reference field bằng backtick path.
- **Giới hạn state: 32k tokens cho `state` + câu hỏi dài nhất** (64k tổng). Giá `$0.042/Mtok` input.
- Error 401/422/429/529; retry backoff cho 429/529.

Nguồn: https://docs.typesafe.ai/models.md · /api.md · /cookbooks/parallel_questions.md · /patterns/fan-out.md

## Kết quả (real mode, key từ ~/.config/env)

### Accuracy trên 20 queries (top_k=10, threshold 0.5)

| Giai đoạn | noise_ratio | precision | recall |
|---|---|---|---|
| BEFORE (retriever lexical) | 0.805 | 0.195 | 0.975 |
| **AFTER (Jev rerank)** | **0.05** | **0.95** | 0.95 |

Jev Noul lọc noise mạnh: precision 0.195 → 0.95, noise_ratio 0.805 → 0.05.

### Benchmark A/B latency (10 chunk ~512 token, query q-sql-index)

| variant | calls | wall_s | latency/call | in_tokens | cost |
|---|---|---|---|---|---|
| fan-out w=1 | 10 | 3.60 | 360ms | 7210 | $0.00030 |
| fan-out w=4 | 10 | 1.13 | 113ms | 7210 | $0.00030 |
| fan-out w=8 | 10 | 0.73 | 72ms | 7210 | $0.00030 |
| fan-out w=16 | 10 | 0.40 | 40ms | 7210 | $0.00030 |
| **single-call n=10** | **1** | **0.43** | — | **4131** | **$0.00017** |
| single-call n=20 | 1 | 0.48 | — | 7878 | $0.00033 |
| single-call n=30 | 1 | 0.47 | — | 11406 | $0.00048 |
| single-call n=40 | 1 | 0.63 | — | 14956 | $0.00063 |

### Giới hạn state

- 40 chunks × 512 cl100k-token = 21,452 cl100k → Jev đếm **14,956 token** (tokenizer Jev khác tiktoken, thấp hơn ~30%).
- 14,956 < 32k → **single-call gói gọn 1 request**. Adaptive cài sẵn (demo ép budget 6000 → chia nhiều request, vẫn đúng).

## Khuyến nghị

- **Single-call multi-question** cho re-rank: 1 call, query gửi 1 lần, token & cost thấp hơn fan-out ~1.7x, accuracy giữ nguyên.
- Fan-out chỉ đáng khi cần hỏi nhiều câu khác nhau mỗi cặp, hoặc retry độc lập từng cặp.

## Giới hạn

- Retriever là lexical giả lập (không phải embedding thật).
- Ground truth 2 chunk/query, corpus 40 chunk — minh hoạ, chưa phải benchmark quy mô.
- Jev tối ưu tiếng Anh; tiếng Việt có thể kém chính xác hơn.
- Tokenizer đo chunk = tiktoken `cl100k_base` (khác tokenizer Jev "Btok").

## Bước tiếp theo

1. Scale 100-200 chunks tìm ngưỡng thật của giới hạn 32k state token.
2. Thử primitive `score` (thang 0..5) thay `noul`.
3. So Jev rerank vs cross-encoder (`bge-reranker`) cùng dataset.
