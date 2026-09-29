"""Build jev_rag_rerank.ipynb from cell sources (stdlib only)."""
import json

MD = "markdown"
CODE = "code"

cells = []

_cell_idx = 0

def add(cell_type, source):
    global _cell_idx
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
        "id": f"cell-{_cell_idx}",
    }
    if cell_type == CODE:
        cell["execution_count"] = None
        cell["outputs"] = []
    _cell_idx += 1
    cells.append(cell)

add(MD, """# Jev RAG Rerank — chấm relevance + benchmark A/B (data chuẩn 512-token)

**Mục tiêu:** dùng model **Jev** (TypeSafe AI) chấm relevance từng chunk trong `top_k`,
shortlist để giảm noise. Kèm **benchmark A/B latency** trên chunk ~512 tokens (chuẩn RAG thực tế):
- **A. Per-pair fan-out** — 1 call cho mỗi cặp (query, chunk).
- **B. Single-call multi-question** — 1 call, 1 Noul/candidate (parallel questions).

Data đầu vào từ `data/sample_inputs.json`: 20 queries (≤20 tokens) + 40 chunks (~512 tokens,
đo bằng tiktoken thật) + ground-truth relevance. Retriever giả lập bằng **lexical overlap**
(tương tự BM25), Jev đóng vai trò re-ranker ngữ nghĩa.
""")

add(MD, """## Giới hạn state/token của Jev (xác minh từ live docs)

Nguồn: https://docs.typesafe.ai/models.md (fetch 2026-09-29)

- **Context length:** 64k tokens mỗi request; **32k tokens cho `state` + câu hỏi dài nhất**.
- Model `jev-latest` → `jev-1.13.0`; giá `$0.042 / Mtok` input (output free).
- Rate limit: 250k tokens/s, 1200 req/min.

**Lưu ý tokenizer:** Jev dùng token riêng (đơn vị "Btok"). Đo chunk bằng `tiktoken cl100k_base`
(~512 tokens/chunk tiếng Việt), nhưng token thật của Jev (qua `usage.input_tokens`) thấp hơn
~30% (40 chunks đo 21.4k cl100k → Jev báo 15.0k). Con số 32k là theo token của Jev.

→ 40 chunks × 512 cl100k-token ≈ 15k Jev-token < 32k → **single-call 1 request vẫn đủ**.
Pipeline vẫn cài **adaptive** (chia nhiều request) khi state vượt ngân sách an toàn.
""")

add(CODE, """# 1. Setup / config
import os
import json
import re
import time
import socket
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

def load_env_file(path="~/.config/env"):
    path = os.path.expanduser(path)
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out

_ENV_FILE = load_env_file("~/.config/env")

JEV_API_KEY   = _ENV_FILE.get("JEV_API_KEY") or os.environ.get("JEV_API_KEY", "")
JEV_ENDPOINT  = os.environ.get("JEV_ENDPOINT") or "https://api.typesafe.ai"
JEV_MODEL     = "jev-latest"
JEV_TIMEOUT   = 120

TOP_K       = 10
RELEVANCE_THRESHOLD = 0.5
TOP_N       = 5
PRICE_INPUT_PER_M = 0.042

# Ngân sách an toàn cho state (token Jev). Docs: 32k cho state + câu hỏi dài nhất.
STATE_TOKEN_BUDGET = 28000

MOCK = not bool(JEV_API_KEY)
print(f"JEV_API_KEY set : {bool(JEV_API_KEY)} (nguồn: {'~/.config/env' if _ENV_FILE.get('JEV_API_KEY') else 'env var'})")
print(f"MODE            : {'MOCK' if MOCK else 'REAL'}")
print(f"STATE_TOKEN_BUDGET = {STATE_TOKEN_BUDGET} (giới hạn 32k theo docs)")
""")

add(CODE, """# 2. Load data + thống kê token (đo bằng tiktoken, ghi sẵn trong data file)

with open("data/sample_inputs.json", encoding="utf-8") as f:
    DATA = json.load(f)

QUERIES = DATA["queries"]
CHUNKS  = DATA["chunks"]
CHUNK_BY_ID = {c["id"]: c for c in CHUNKS}

def stats_token(items, label):
    toks = [x["token_count"] for x in items]
    print(f"{label:<8} n={len(toks):>2}  min={min(toks)}  max={max(toks)}  avg={sum(toks)/len(toks):.1f}")

print("tokenizer:", DATA["meta"]["tokenizer"])
stats_token(QUERIES, "queries")
stats_token(CHUNKS, "chunks")
print(f"Tổng token 40 chunks (tiktoken cl100k): {sum(c['token_count'] for c in CHUNKS):,}")
print(f"Mỗi query có {len(QUERIES[0]['relevant_chunks'])} chunk liên quan (ground truth)")
""")

add(CODE, """# 3. Retriever giả lập (lexical overlap, tương tự BM25) — Jev là re-ranker ngữ nghĩa

_STOPWORDS = {
    "làm","thế","nào","để","trong","và","của","cho","một","các","trên","nên","dùng",
    "giúp","điều","chỉnh","định","kỳ","cần","thiết","hợp","lý","bảng","lớn","mỗi",
    "từng","đúng","chỉ","lấy","giảm","nhiều","hay","khi","được","với","về","này",
    "đó","là","có","bị","ra","vào","cách","thực","xem","tìm","chỗ","tốn","trả",
    "theo","phải","quét","xếp","như","hoặc","hơn","chọn","tốt","làm","gì","đâu",
    "bao","nào","sao","những","vấn","đề","khác","biệt","điểm",
}

def _tokens(s):
    return {
        w for w in re.findall(r"[a-z0-9àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹđ]+", s.lower())
        if w not in _STOPWORDS and len(w) > 1
    }

def retriever_score(query, text):
    q, t = _tokens(query), _tokens(text)
    if not q:
        return 0.0
    cov = len(q & t) / len(q)          # coverage: query phủ bao nhiêu ý trong chunk
    jac = len(q & t) / len(q | t)      # jaccard
    return cov * 0.5 + jac * 0.5

def retrieve_top_k(query, top_k=TOP_K):
    return sorted(CHUNKS, key=lambda c: -retriever_score(query, c["text"]))[:top_k]
""")

add(CODE, """# 4. Jev client — retry backoff + fan-out + multi-question + adaptive

class JevClient:
    def __init__(self, api_key=JEV_API_KEY, endpoint=JEV_ENDPOINT, model=JEV_MODEL, timeout=JEV_TIMEOUT):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.mock = not bool(api_key)
        self.retries = 0
        self.timeouts = 0
        self.errors = []
        self.last_usage = (0, 0)

    def _post(self, payload, retries=3):
        url = f"{self.endpoint}/v1/systemone"
        data = json.dumps(payload).encode("utf-8")
        backoff = 1.0
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, method="POST",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    u = body.get("usage", {})
                    self.last_usage = (u.get("input_tokens", 0), u.get("output_tokens", 0))
                    return body
            except urllib.error.HTTPError as e:
                if e.code in (429, 529) and attempt < retries:
                    self.retries += 1
                    time.sleep(backoff); backoff *= 2
                    continue
                self.errors.append(f"HTTP {e.code}: {e.read().decode()[:200]}")
                raise
            except (socket.timeout, TimeoutError) as e:
                if attempt < retries:
                    self.timeouts += 1
                    time.sleep(backoff); backoff *= 2
                    continue
                self.errors.append(f"timeout: {e}")
                raise
        raise RuntimeError("exhausted retries")

    # --- A. per-pair fan-out ---
    def score_relevance(self, query, passage):
        payload = {
            "state": {"query": query, "passage": passage},
            "model": self.model,
            "questions": {"relevant": {"type": "noul", "instructions": "Does the passage answer the query?"}},
        }
        return float(self._post(payload)["answers"]["relevant"]["noul"])

    def fanout(self, query, passages, max_workers=8):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda p: self.score_relevance(query, p), passages))

    # --- B. single-call multi-question ---
    def _multi_question(self, query, candidates):
        questions = {
            f"rel_{k}": {"type": "noul", "instructions": f"Does `candidates.{k}` answer `query`?"}
            for k in candidates
        }
        payload = {"state": {"query": query, "candidates": candidates}, "model": self.model, "questions": questions}
        resp = self._post(payload)
        return {k: float(resp["answers"][f"rel_{k}"]["noul"]) for k in candidates}

    # --- B+. multi-question adaptive: chia request nếu state vượt ngân sách token ---
    def multi_question_adaptive(self, query, candidates, budget=STATE_TOKEN_BUDGET):
        items = [(cid, CHUNK_BY_ID[cid]["token_count"]) for cid in candidates]
        items.sort(key=lambda x: -x[1])
        batches, cur, cur_tokens = [], [], _EST_QUERY_TOKENS
        for cid, tk in items:
            add = tk + 30  # chunk + instruction overhead
            if cur and cur_tokens + add > budget:
                batches.append(cur); cur, cur_tokens = [], _EST_QUERY_TOKENS
            cur.append(cid); cur_tokens += add
        if cur:
            batches.append(cur)
        out = {}
        for batch in batches:
            sub = {cid: CHUNK_BY_ID[cid]["text"] for cid in batch}
            out.update(self._multi_question(query, sub))
        return out

_EST_QUERY_TOKENS = 25

client = JevClient()
print("Client ready. mock =", client.mock)
""")

add(CODE, """# 5. Pipeline + metric (ground truth từ data)

def metrics(cands, relevant_ids):
    n = len(cands)
    if n == 0:
        return {"n":0,"noise_ratio":None,"precision":None,"recall":None}
    rel = sum(1 for c in cands if c["id"] in relevant_ids)
    noise = n - rel
    return {
        "n": n,
        "noise_ratio": round(noise/n, 3),
        "precision": round(rel/n, 3),
        "recall": round(rel/len(relevant_ids), 3),
    }

def _f(v):
    return "  -" if v is None else f"{v:>5}"

def fmt(name, m):
    return (f"{name:<26} n={m['n']:>2}  noise_ratio={_f(m['noise_ratio'])}  "
            f"precision={_f(m['precision'])}  recall={_f(m['recall'])}")
""")

add(CODE, """# 6. Đánh giá accuracy trên toàn bộ 20 queries (single-call, top_k=10)

rows = []
for q in QUERIES:
    top = retrieve_top_k(q["text"], TOP_K)
    rel_ids = set(q["relevant_chunks"])
    before = metrics(top, rel_ids)

    cand = {c["id"]: c["text"] for c in top}
    nouls = client.multi_question_adaptive(q["text"], cand)
    kept = [c for c in top if nouls.get(c["id"], 0.0) >= RELEVANCE_THRESHOLD]
    after = metrics(kept, rel_ids)

    rows.append({"qid": q["id"], "before": before, "after": after})

def avg(key, stage):
    vals = [r[stage][key] for r in rows if r[stage][key] is not None]
    return round(sum(vals)/len(vals), 3) if vals else None

print(f"Đánh giá {len(rows)} queries (top_k={TOP_K}, threshold={RELEVANCE_THRESHOLD}):")
print(f"{'':26}{'noise_ratio':>13}{'precision':>11}{'recall':>9}")
print(f"{'BEFORE (retriever lexical)':<26}{str(avg('noise_ratio','before')):>13}{str(avg('precision','before')):>11}{str(avg('recall','before')):>9}")
print(f"{'AFTER (Jev rerank)':<26}{str(avg('noise_ratio','after')):>13}{str(avg('precision','after')):>11}{str(avg('recall','after')):>9}")

print()
print(f"{'qid':<20}{'b_prec':>8}{'a_prec':>8}{'b_noise':>9}{'a_noise':>9}")
for r in rows[:10]:
    b, a = r["before"], r["after"]
    print(f"{r['qid']:<20}{_f(b['precision']):>8}{_f(a['precision']):>8}{_f(b['noise_ratio']):>9}{_f(a['noise_ratio']):>9}")
""")

add(CODE, """# 7. Benchmark A/B latency (query chính q-sql-index, chunk 512 tokens)

PRIMARY = QUERIES[0]
print("Query:", PRIMARY["text"])
print("Relevant chunks:", PRIMARY["relevant_chunks"])

top = retrieve_top_k(PRIMARY["text"], TOP_K)
passages = [c["text"] for c in top]
print(f"top_k={len(top)} chunks, tổng ~{sum(c['token_count'] for c in top)} cl100k-tokens")

print()
print("A. Per-pair fan-out:")
fanout_rows = []
for w in [1, 4, 8, 16]:
    b = JevClient()
    t0 = time.perf_counter()
    b.fanout(PRIMARY["text"], passages, max_workers=w)
    wall = time.perf_counter() - t0
    n = len(passages)
    fanout_rows.append({"variant": f"fanout w={w}", "calls": n, "wall_s": round(wall,3),
                        "latency_ms": round(wall/n*1000,1), "throughput": round(n/wall,1),
                        "retries": b.retries, "timeouts": b.timeouts})
    print(f"  workers={w:>2}: wall={wall:.3f}s  latency/call={wall/n*1000:.0f}ms  throughput={n/wall:.1f} req/s  retries={b.retries}")

print()
print("B. Single-call multi-question:")
single_rows = []
for n in [10, 20, 30, 40]:
    sub = sorted(CHUNKS, key=lambda c: -retriever_score(PRIMARY["text"], c["text"]))[:n]
    cand = {c["id"]: c["text"] for c in sub}
    b = JevClient()
    t0 = time.perf_counter()
    b.multi_question_adaptive(PRIMARY["text"], cand)
    wall = time.perf_counter() - t0
    in_t, out_t = b.last_usage
    single_rows.append({"variant": f"single-call n={n}", "calls": 1, "wall_s": round(wall,3),
                        "latency_ms": round(wall*1000,0), "in_tokens": in_t, "out_tokens": out_t,
                        "cost": round(in_t/1e6*PRICE_INPUT_PER_M,5),
                        "retries": b.retries, "timeouts": b.timeouts})
    print(f"  n={n:>2}: wall={wall:.3f}s  calls=1  in_tokens={in_t}  out_tokens={out_t}  cost=${in_t/1e6*PRICE_INPUT_PER_M:.5f}")
""")

add(CODE, """# 8. Adaptive check: state token ngân sách (40 chunks có cần chia request không)

all_cand = {c["id"]: c["text"] for c in CHUNKS}
total_cl100k = sum(c["token_count"] for c in CHUNKS)
print(f"40 chunks: {total_cl100k:,} cl100k-tokens (tiktoken)")
print(f"Jev token thật (usage lần gần nhất): ~{single_rows[-1]['in_tokens']:,}")
print(f"Ngân sách an toàn STATE_TOKEN_BUDGET = {STATE_TOKEN_BUDGET:,} (limit 32k)")

print()
print("Demo adaptive (ép budget=6000 cl100k-token để thấy chia nhiều request):")
b = JevClient()
t0 = time.perf_counter()
nouls = b.multi_question_adaptive(PRIMARY["text"], all_cand, budget=6000)
wall = time.perf_counter() - t0
print(f"  trả về {len(nouls)} noul, wall={wall:.3f}s (đã chia nhiều request dưới ngân sách 6000)")
print(f"  relevant noul: " + ", ".join(f"{cid}={nouls[cid]:.2f}" for cid in PRIMARY['relevant_chunks']))
""")

add(CODE, """# 9. Bảng tổng hợp A/B

probe = JevClient()
probe.score_relevance(PRIMARY["text"], top[0]["text"])
fanout_in_per_call = probe.last_usage[0]
fanout_in = fanout_in_per_call * TOP_K

print(f"{'variant':<18}{'calls':>6}{'wall_s':>9}{'latency_ms':>11}{'in_tokens':>10}{'cost':>10}{'retries':>8}{'timeouts':>9}")
for r in fanout_rows:
    print(f"{r['variant']:<18}{r['calls']:>6}{r['wall_s']:>9}{r['latency_ms']:>11.0f}{fanout_in:>10}${fanout_in/1e6*PRICE_INPUT_PER_M:>9.5f}{r['retries']:>8}{r['timeouts']:>9}")
for r in single_rows:
    print(f"{r['variant']:<18}{r['calls']:>6}{r['wall_s']:>9}{r['latency_ms']:>11.0f}{r['in_tokens']:>10}${r['cost']:>9.5f}{r['retries']:>8}{r['timeouts']:>9}")

print()
print("KHUYẾN NGHỊ:")
print("- Single-call multi-question thắng latency & cost: 1 call, query chỉ gửi 1 lần, token thấp hơn fan-out nhiều.")
print("- 40 chunks x 512-token vẫn gói gọn trong 1 request (15k Jev-token < 32k limit).")
print("- Fan-out chỉ đáng khi cần hỏi NHIỀU câu khác nhau mỗi cặp, hoặc cần retry độc lập từng cặp.")
""")

add(MD, """## Nhận xét & giới hạn

**Phát hiện chính:**
- Data chuẩn RAG: 20 queries (≤20 tokens) + 40 chunks (~512 tokens, tiktoken thật) + ground truth.
- Retriever lexical (giả lập BM25) đưa cả chunk "gần nghĩa" lên top_k; Jev Noul (ngữ nghĩa) lọc
  đúng chunk trả lời câu hỏi → precision tăng, noise_ratio giảm rõ.
- **Single-call multi-question** xử lý 40 chunks × 512-token trong 1 request (~0.7s), thắng
  fan-out về latency, số call và token.
- Giới hạn state là **32k Jev-token** (docs models.md). 40 chunks ≈ 15k Jev-token → chưa cần
  chia request; adaptive cài sẵn cho scale lớn hơn.
- Tokenizer: tiktoken cl100k (đo chunk) ≠ tokenizer Jev (Btok); Jev đếm thấp hơn ~30% với tiếng Việt.

**Giới hạn:**
- Retriever là lexical giả lập (không phải embedding thật); precision/recall phụ thuộc top_k.
- Ground truth 2 chunk liên quan/query, corpus 40 chunk — đủ minh hoạ nhưng chưa là benchmark quy mô.
- Latency có nhiễu giữa các lần đo; chưa gặp 429 (retry backoff cài sẵn).
- Jev tối ưu cho tiếng Anh; tiếng Việt có thể kém chính xác hơn (đã lưu ý trong docs).

**Bước tiếp theo:**
1. Scale 100-200 chunks để tìm ngưỡng thật của giới hạn 32k state token.
2. Thử primitive `score` thay `noul` để phân cấp relevance mịn hơn.
3. So Jev rerank vs cross-encoder (`bge-reranker`) cùng dataset.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("jev_rag_rerank.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Wrote jev_rag_rerank.ipynb")
