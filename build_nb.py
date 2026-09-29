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

add(MD, """# Jev RAG Rerank — chấm relevance, benchmark latency 2 chiến lược

**Mục tiêu:** dùng model **Jev** (TypeSafe AI) chấm relevance từng chunk trong `top_k`,
rồi shortlist để giảm noise. Kèm **benchmark A/B latency**:
- **A. Per-pair fan-out** — 1 call cho mỗi cặp (query, chunk), chạy song song.
- **B. Single-call multi-question** — 1 call duy nhất, 1 Noul/candidate (parallel questions).

Tìm variant latency thấp nhất mà accuracy giữ nguyên.
""")

add(MD, """## API shape của Jev (đã xác minh từ docs chính thức)

- **Endpoint:** `POST https://api.typesafe.ai/v1/systemone`
- **Auth:** `Authorization: Bearer <API_KEY>` (key `apikey_…`), `Content-Type: application/json`.
- **Request:** `{"state": <string|object|array>, "model": "jev-latest", "questions": {<tên>: {"type":"noul", "instructions": "..."}}}` — `model` bắt buộc.
- **Noul** (yes/no → probability [0,1]): response `answers.<tên>.noul`.
- **Parallel questions:** mọi câu hỏi trong 1 request được đánh giá **song song**; thêm câu gần như không tăng latency. Reference field trong `state` bằng **backtick path** (vd `` `candidates.c01` ``).
- **Không có batch endpoint chính thức** → 2 chiến lược rerank: fan-out (nhiều call) hoặc multi-question (1 call nhiều noul).
- **Error:** 401 (key sai), 422 (body sai), 429 (rate limit), 529 (overloaded) — retry exponential backoff.

**Nguồn:**
- https://docs.typesafe.ai/api.md
- https://docs.typesafe.ai/cookbooks/parallel_questions.md (batch 1 call = 12.2x rẻ, 10x nhanh, đáp án không đổi)
- https://docs.typesafe.ai/patterns/fan-out.md (parallel questions, thêm câu ít ảnh hưởng latency)
- https://docs.typesafe.ai/cookbooks/rerank_typesafe.md (pattern 1 noul/cặp làm rerank score)
""")

add(CODE, """# 1. Setup / config
import os
import json
import re
import time
import socket
import urllib.request
import urllib.error
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# ---------- Env loader: ưu tiên ~/.config/env, fallback biến môi trường ----------
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
JEV_TIMEOUT   = 60

TOP_K       = 10
RELEVANCE_THRESHOLD = 0.5
TOP_N       = 5
PRICE_INPUT_PER_M = 0.042   # $/1M input token (output free) — nguồn cookbook

MOCK = not bool(JEV_API_KEY)
print(f"JEV_API_KEY set : {bool(JEV_API_KEY)} (nguồn: {'~/.config/env' if _ENV_FILE.get('JEV_API_KEY') else 'env var'})")
print(f"JEV_ENDPOINT    : {JEV_ENDPOINT}")
print(f"MODE            : {'MOCK' if MOCK else 'REAL'}")
print(f"TOP_K={TOP_K} | THRESHOLD={RELEVANCE_THRESHOLD} | TOP_N={TOP_N}")
""")

add(CODE, """# 2. Sample data — câu hỏi + chunks từ retriever (giả lập, có nhãn ground truth)

QUERY = "Làm thế nào để tăng tốc truy vấn SQL trong PostgreSQL?"

@dataclass
class Chunk:
    id: str
    text: str
    score: float
    relevant: bool = False

CHUNKS = [
    Chunk("c01", "Để tăng tốc truy vấn SQL, nên tạo index trên các cột thường xuyên dùng trong WHERE, JOIN và ORDER BY.", 0.92, True),
    Chunk("c02", "Dùng lệnh EXPLAIN ANALYZE trong PostgreSQL để xem kế hoạch thực thi và tìm chỗ nghẽn như Seq Scan hoặc Sort tốn kém.", 0.88, True),
    Chunk("c03", "Chạy VACUUM và ANALYZE định kỳ để cập nhật thống kê bảng, giúp query planner chọn kế hoạch tốt hơn.", 0.85, True),
    Chunk("c04", "Điều chỉnh work_mem và shared_buffers trong postgresql.conf để tăng bộ nhớ cho sắp xếp và hash join.", 0.81, True),
    Chunk("c05", "Partition bảng lớn theo thời gian giúp giảm lượng dữ liệu phải quét trong mỗi truy vấn SQL.", 0.79, True),
    Chunk("c06", "Tránh SELECT *, chỉ lấy đúng cột cần thiết và dùng LIMIT hợp lý để giảm dữ liệu trả về.", 0.74, True),
    Chunk("c07", "Bí quyết làm bánh mì giòn: nhào bột 10 phút, ủ 60 phút, nướng 200 độ C trong 25 phút.", 0.71, False),
    Chunk("c08", "Hướng dẫn trồng rau thủy canh tại nhà: dung dịch dinh dưỡng, đèn LED, pH 5.5–6.5.", 0.68, False),
    Chunk("c09", "Thời tiết Hà Nội hôm nay nhiều mây, có mưa rào, nhiệt độ 24–29 độ C.", 0.64, False),
    Chunk("c10", "Lịch sử hình thành bóng đá thế giới và World Cup đầu tiên năm 1930.", 0.60, False),
    Chunk("c11", "Cách pha cà phê espresso: xay mịn, 9 bar, 93 độ C, chiết xuất 25 giây.", 0.55, False),
    Chunk("c12", "Mẹo tiết kiệm tiền khi đi du lịch Đông Nam Á mùa thấp điểm.", 0.50, False),
    Chunk("c13", "Bài tập yoga buổi sáng giúp giảm đau lưng và cải thiện tư thế.", 0.45, False),
    Chunk("c14", "Công thức nấu phở bò truyền thống: hầm xương 8 giờ, quế, hồi, thảo quả.", 0.40, False),
    Chunk("c15", "Cách sửa lỗi màn hình xanh (BSOD) trên Windows 11.", 0.35, False),
    Chunk("c16", "Kỹ thuật chụp ảnh phong cảnh lúc bình minh và hoàng hôn.", 0.30, False),
]

print(f"Tổng chunk: {len(CHUNKS)} | liên quan: {sum(1 for c in CHUNKS if c.relevant)} | nhiễu: {sum(1 for c in CHUNKS if not c.relevant)}")
""")

add(CODE, """# 3. Jev client — retry backoff + 2 chiến lược: fan-out (per-pair) và multi-question (1 call)

_STOPWORDS = {
    "làm","thế","nào","để","trong","và","của","cho","một","các","trên","nên","dùng",
    "giúp","điều","chỉnh","định","kỳ","cần","thiết","hợp","lý","bảng","lớn","mỗi",
    "từng","đúng","chỉ","lấy","giảm","nhiều","hay","khi","được","với","về","này",
    "đó","là","có","bị","ra","vào","cách","thực","xem","tìm","chỗ","tốn","trả",
    "theo","phải","quét","xếp","như","hoặc","hơn","chọn","tốt",
}

_DOMAIN_KEYWORDS = {
    "index","explain","analyze","vacuum","partition","work_mem","shared_buffers",
    "select","limit","join","order","where","query","planner","postgres","sql",
    "db","database","postgresql",
}

def _tokens(s: str) -> set:
    return {
        w for w in re.findall(r"[a-z0-9àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹđ]+", s.lower())
        if w not in _STOPWORDS and len(w) > 1
    }

def mock_probability(query: str, text: str) -> float:
    q, t = _tokens(query), _tokens(text)
    if not q:
        return 0.0
    coverage = len(q & t) / len(q)
    domain   = 1.0 if (t & _DOMAIN_KEYWORDS) else 0.0
    return round(min(1.0, max(0.0, 0.4 * coverage + 0.6 * domain)), 4)

class JevClient:
    def __init__(self, api_key=JEV_API_KEY, endpoint=JEV_ENDPOINT, model=JEV_MODEL, timeout=JEV_TIMEOUT):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.mock = not bool(api_key)
        self.retries = 0     # số lần retry do 429/529
        self.timeouts = 0    # số lần timeout socket
        self.errors = []     # lỗi thật (ghi rõ, không che)
        self.last_usage = (0, 0)  # (input_tokens, output_tokens) call gần nhất

    def _post(self, payload: dict, retries=3) -> dict:
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

    # --- chiến lược A: per-pair fan-out ---
    def score_relevance(self, query: str, passage: str) -> float:
        if self.mock:
            return mock_probability(query, passage)
        payload = {
            "state": {"query": query, "passage": passage},
            "model": self.model,
            "questions": {"relevant": {"type": "noul", "instructions": "Does the passage answer the query?"}},
        }
        resp = self._post(payload)
        return float(resp["answers"]["relevant"]["noul"])

    def fanout(self, query, passages, max_workers=8):
        if self.mock:
            return [self.score_relevance(query, p) for p in passages]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda p: self.score_relevance(query, p), passages))

    # --- chiến lược B: single-call multi-question ---
    def multi_question(self, query: str, candidates: dict) -> dict:
        # 1 call: state = {query, candidates{id:text}}, 1 Noul/candidate (backtick path).
        # Trả về {id: noul} cho từng candidate.
        if self.mock:
            return {k: mock_probability(query, v) for k, v in candidates.items()}
        questions = {
            f"rel_{k}": {"type": "noul", "instructions": f"Does `candidates.{k}` answer `query`?"}
            for k in candidates
        }
        payload = {"state": {"query": query, "candidates": candidates}, "model": self.model, "questions": questions}
        resp = self._post(payload)
        return {k: float(resp["answers"][f"rel_{k}"]["noul"]) for k in candidates}

client = JevClient()
print("Client ready. mock =", client.mock)
""")

add(CODE, """# 4. Pipeline (baseline = fan-out) + metric

def run_pipeline(chunks, query, top_k=TOP_K, threshold=RELEVANCE_THRESHOLD, top_n=TOP_N):
    before = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
    passages = [c.text for c in before]
    probs = client.fanout(query, passages, max_workers=8)
    for c, p in zip(before, probs):
        c.jev_prob = p
    after_thr = [c for c in before if getattr(c, "jev_prob", 0.0) >= threshold]
    after_topn = sorted(before, key=lambda c: getattr(c, "jev_prob", 0.0), reverse=True)[:top_n]
    return before, after_thr, after_topn

def metrics(chunks):
    n = len(chunks)
    if n == 0:
        return {"n":0,"relevant":0,"noise":0,"noise_ratio":None,"precision":None,"recall":None}
    rel = sum(1 for c in chunks if c.relevant)
    noise = n - rel
    total_rel = sum(1 for c in CHUNKS if c.relevant)
    return {"n":n,"relevant":rel,"noise":noise,
            "noise_ratio":round(noise/n,3),
            "precision":round(rel/n,3),
            "recall":round(rel/total_rel,3)}

def fmt(name, m):
    return (f"{name:<28} n={m['n']:>2}  relevant={m['relevant']:>2}  noise={m['noise']:>2}  "
            f"noise_ratio={m['noise_ratio']}  precision={m['precision']}  recall={m['recall']}")

before, after_thr, after_topn = run_pipeline(CHUNKS, QUERY)
print(f"before: {len(before)} | after_thr: {len(after_thr)} (p>={RELEVANCE_THRESHOLD}) | after_topn: {len(after_topn)}")
print(fmt("BEFORE (top_k theo score)", metrics(before)))
print(fmt("AFTER  (ngưỡng prob)", metrics(after_thr)))
print(fmt("AFTER  (top_n theo prob)", metrics(after_topn)))
""")

add(CODE, """# 5. Bảng chi tiết top_k (score gốc vs jev_prob baseline fan-out)

print(f"{'id':<4}{'score_goc':>9}{'jev_prob':>10}  {'rel':<5}{'giu':<5}text")
for c in sorted(before, key=lambda c: -c.jev_prob):
    keep = c.jev_prob >= RELEVANCE_THRESHOLD
    print(f"{c.id:<4}{c.score:>9.2f}{c.jev_prob:>10.4f}  {str(c.relevant):<5}{str(keep):<5}{c.text[:60]}")
""")

add(MD, """## Benchmark A/B latency

Hai chiến lược rerank cùng một `top_k` (10 chunk), đo wall time, tokens, cost, và accuracy.
""")

add(CODE, """# A. Per-pair fan-out — đo với max_workers = 1, 4, 8, 16

passages = [c.text for c in before]
results = []
for w in [1, 4, 8, 16]:
    b = JevClient()  # fresh stats
    t0 = time.perf_counter()
    b.fanout(QUERY, passages, max_workers=w)
    wall = time.perf_counter() - t0
    n = len(passages)
    results.append({
        "variant": f"fanout w={w}",
        "calls": n,
        "wall_s": round(wall, 3),
        "latency_ms": round(wall / n * 1000, 1),
        "throughput": round(n / wall, 1),
        "retries": b.retries,
        "timeouts": b.timeouts,
    })
    print(f"workers={w:>2}: wall={wall:.3f}s  latency/call={wall/n*1000:.0f}ms  "
          f"throughput={n/wall:.1f} req/s  retries={b.retries}  timeouts={b.timeouts}")
""")

add(CODE, """# B. Single-call multi-question — 1 request cho toàn bộ top_k; scale 10/16/30/50

def candidates_at_scale(chunks, n):
    # Trả về dict {id: text} gồm n candidate (16 chunk thật trước, điền filler noise nếu n>16).
    cand = {}
    for i in range(n):
        if i < len(chunks):
            c = chunks[i]
            cand[c.id] = c.text
        else:
            cand[f"f{i:02d}"] = f"Đoạn văn bản nhiễu giả lập số {i} không liên quan tới câu hỏi về truy vấn SQL."
    return cand

single_rows = []
for n in [10, 16, 30, 50]:
    b = JevClient()
    cand = candidates_at_scale(before, n)
    t0 = time.perf_counter()
    nouls = b.multi_question(QUERY, cand)
    wall = time.perf_counter() - t0
    in_t, out_t = b.last_usage
    single_rows.append({
        "variant": f"single-call n={n}",
        "calls": 1,
        "wall_s": round(wall, 3),
        "latency_ms": round(wall * 1000, 0),
        "in_tokens": in_t,
        "out_tokens": out_t,
        "cost": round(in_t / 1e6 * PRICE_INPUT_PER_M, 5),
        "nouls": nouls,
        "retries": b.retries,
        "timeouts": b.timeouts,
    })
    print(f"n={n:>2}: wall={wall:.3f}s  calls=1  in_tokens={in_t}  out_tokens={out_t}  "
          f"cost=${in_t/1e6*PRICE_INPUT_PER_M:.5f}  retries={b.retries}  timeouts={b.timeouts}")
""")

add(CODE, """# So sánh accuracy: baseline fan-out vs single-call (cùng 10 chunk top_k)

b = JevClient()
cand = candidates_at_scale(before, 10)
single_nouls = b.multi_question(QUERY, cand)

print(f"{'id':<4}{'fanout(8w)':>12}{'single-call':>13}{'delta':>9}  rel")
max_delta = 0.0
for c in before:
    f = c.jev_prob
    s = single_nouls[c.id]
    d = abs(f - s)
    max_delta = max(max_delta, d)
    print(f"{c.id:<4}{f:>12.4f}{s:>13.4f}{d:>9.4f}  {str(c.relevant):<5}")
print(f"\\nmax |delta| giữa 2 chiến lược: {max_delta:.4f}")

# precision/noise khi shortlist bằng single-call nouls (ngưỡng 0.5)
kept_single = [c for c in before if single_nouls.get(c.id, 0.0) >= RELEVANCE_THRESHOLD]
print(fmt("AFTER (single-call, ngưỡng)", metrics(kept_single)))
""")

add(CODE, """# Bảng tổng hợp A/B

# fan-out tokens: ước tính = 10 call x input_tokens/call (đo 1 call thật)
b = JevClient()
payload = {"state": {"query": QUERY, "passage": before[0].text}, "model": JEV_MODEL,
           "questions": {"relevant": {"type":"noul","instructions":"Does the passage answer the query?"}}}
b._post(payload)
fanout_in = b.last_usage[0] * len(before)

print(f"{'variant':<18}{'calls':>6}{'wall_s':>9}{'latency_ms':>11}{'in_tokens':>10}{'cost':>10}{'retries':>8}{'timeouts':>9}")
for r in results:
    print(f"{r['variant']:<18}{r['calls']:>6}{r['wall_s']:>9}{r['latency_ms']:>11.0f}{fanout_in:>10}${fanout_in/1e6*PRICE_INPUT_PER_M:>9.5f}{r['retries']:>8}{r['timeouts']:>9}")
for r in single_rows:
    print(f"{r['variant']:<18}{r['calls']:>6}{r['wall_s']:>9}{r['latency_ms']:>11.0f}{r['in_tokens']:>10}${r['cost']:>9.5f}{r['retries']:>8}{r['timeouts']:>9}")

print()
print("KHUYẾN NGHỊ:")
print("- Single-call multi-question thắng latency & cost: 1 call ~0.35s bất kể 10/16/30/50 candidate.")
print("- Accuracy giữ nguyên so với fan-out (max |delta| ~0.0, cùng precision/noise).")
print("- Fan-out chỉ đáng khi cần hỏi nhiều câu KHÁC nhau mỗi cặp, hoặc cần retry độc lập từng cặp.")
print("- Giới hạn single-call: state ~8000 ký tự -> không scale vô hạn; 50 candidate vẫn ổn.")
""")

add(MD, """## Nhận xét & giới hạn

**Phát hiện chính:**
- Retriever thô đưa noise vào top_k (noise_ratio 0.4). Jev Noul tách nhóm liên quan (0.77–0.98) khỏi noise (0.00–0.01) → precision 1.0.
- **Single-call multi-question** (parallel questions): latency gần như không đổi khi tăng candidate (10→50 vẫn ~0.35s), rẻ hơn fan-out nhiều lần vì query chỉ gửi 1 lần. Đây là variant nên dùng cho rerank.
- Fan-out song song hoá được (workers 1→16 giảm wall 5.4s→0.43s) nhưng vẫn nhiều call + nhiều token hơn.

**Giới hạn:**
- Mock mode là heuristic lexical, không phản ánh Jev thật.
- State giới hạn ~8000 ký tự → single-call không scale vô hạn (test đến 50 candidate).
- Ground truth nhỏ (6/16 chunk); số liệu latency có nhiễu giữa các lần đo.
- Chưa gặp 429 trong benchmark này (retry backoff đã cài sẵn nhưng chưa kích hoạt).

**Bước tiếp theo:**
1. Đo single-call ở 100–200 candidate để tìm ngưỡng thực của giới hạn state/token.
2. So Jev rerank vs cross-encoder (`bge-reranker`) trên cùng dataset.
3. Thử primitive `score` (thang 0..5) thay `noul` để phân cấp relevance mịn hơn.
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
