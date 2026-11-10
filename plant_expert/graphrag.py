"""Agricultural GraphRAG: subgraph extraction, vector search on extension manuals, Ollama calls."""
import json
import urllib.error
import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np


class OllamaError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise OllamaError(
            f"Không kết nối được Ollama tại {url}. Hãy chạy `ollama serve` và pull model (`qwen2.5:7b-instruct`, `bge-m3`). Chi tiết: {e}"
        ) from e


def ollama_generate(prompt: str, model: str = "qwen2.5:7b-instruct", host: str = "http://localhost:11434") -> str:
    res = _post_json(
        f"{host.rstrip('/')}/api/generate",
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
    )
    return res.get("response", "").strip()


def ollama_embed(texts: list[str], model: str = "bge-m3", host: str = "http://localhost:11434") -> np.ndarray:
    res = _post_json(f"{host.rstrip('/')}/api/embed", {"model": model, "input": texts})
    arr = np.asarray(res["embeddings"], dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def subgraph(G: nx.MultiDiGraph, disease_id: str, hops: int = 2) -> nx.MultiDiGraph:
    root = f"disease:{disease_id}"
    if root not in G:
        return nx.MultiDiGraph()
    ego = nx.ego_graph(G.to_undirected(), root, radius=hops)
    keep = [n for n in ego.nodes if not n.startswith("context:")]
    return G.subgraph(keep).copy()


def facts(sub: nx.MultiDiGraph, disease_id: str) -> list[str]:
    root = f"disease:{disease_id}"
    if root not in sub:
        return []
    d_label = sub.nodes[root].get("label", disease_id)
    out = []
    for u, v, data in sub.edges(data=True):
        rel = data.get("rel")
        if u == root and rel == "CAUSED_BY":
            out.append(f"Bệnh {d_label} do tác nhân {sub.nodes[v]['label']} gây ra.")
        elif u == root and rel == "HAS_VISUAL_SIGN":
            out.append(f"Triệu chứng nhận diện của {d_label}: {sub.nodes[v]['label']}.")
        elif u == root and rel == "TREATED_BY":
            ing_name = sub.nodes[v]["label"]
            dose = data.get("dose") or "theo hướng dẫn trên nhãn sản phẩm"
            phi = f"{data['phi_days']} ngày" if data.get("phi_days") is not None else "theo hướng dẫn trên nhãn sản phẩm"
            src = data.get("source", "")
            out.append(
                f"Hoạt chất điều trị {d_label}: {ing_name} | Liều lượng: {dose} | Thời gian cách ly: {phi} (Nguồn: {src})."
            )
        elif rel == "RESTRICTED_WITH":
            subj = sub.nodes[u]["label"]
            out.append(f"CẢNH BÁO: [{subj}] {sub.nodes[v]['label']}")
    return out


class VectorIndex:
    def __init__(self, chunks: list[str], classes: list[str], vecs: np.ndarray):
        self.chunks = chunks
        self.classes = classes
        self.vecs = vecs

    @classmethod
    def build(cls, manual_dir: str = "kg/manual", embed=ollama_embed) -> "VectorIndex":
        chunks, classes = [], []
        for p in sorted(Path(manual_dir).glob("*.md")):
            cid = p.stem
            for para in p.read_text(encoding="utf-8").split("\n\n"):
                s = para.strip()
                if s and not s.startswith("# "):
                    chunks.append(f"[{cid}] {s}")
                    classes.append(cid)
        vecs = embed(chunks) if chunks else np.zeros((0, 1), dtype=float)
        return cls(chunks, classes, vecs)

    def save(self, path: str = "kg/index.npz") -> None:
        np.savez(path, chunks=np.array(self.chunks, dtype=object),
                 classes=np.array(self.classes, dtype=object), vecs=self.vecs)

    @classmethod
    def load(cls, path: str = "kg/index.npz") -> "VectorIndex":
        d = np.load(path, allow_pickle=True)
        return cls(d["chunks"].tolist(), d["classes"].tolist(), d["vecs"])

    def search(self, query: str, k: int = 4, embed=ollama_embed) -> list[str]:
        if len(self.chunks) == 0:
            return []
        qv = embed([query])[0]
        scores = self.vecs @ qv
        top = np.argsort(-scores)[:k]
        return [self.chunks[int(i)] for i in top]


def build_prompt(question: str, disease_vi: str, mode: str, fact_lines: list[str], passages: list[str]) -> str:
    if mode == "llm":
        return (
            f"Bạn là chuyên gia bảo vệ thực vật. Cây cà chua được chẩn đoán mắc: {disease_vi}.\n"
            f"Câu hỏi của nông dân: {question}\n"
            "Hãy trả lời ngắn gọn bằng tiếng Việt về hoạt chất, liều lượng, thời gian cách ly và lưu ý an toàn."
        )
    if mode == "naive":
        ctx = "\n\n".join(passages)
        return (
            f"Bạn là chuyên gia bảo vệ thực vật. Cây cà chua mắc: {disease_vi}.\n"
            f"Tài liệu tham khảo:\n{ctx}\n\n"
            f"Câu hỏi: {question}\n"
            "Hãy trả lời bằng tiếng Việt dựa trên tài liệu tham khảo."
        )
    ctx_facts = "\n".join(f"- {f}" for f in fact_lines)
    ctx_pass = "\n\n".join(passages)
    return (
        f"Bạn là chuyên gia bảo vệ thực vật. Cây cà chua đã được xác nhận mắc: {disease_vi}.\n\n"
        f"=== TRI THỨC ĐỒ THỊ (BẮT BUỘC TUÂN THỦ TUYỆT ĐỐI) ===\n{ctx_facts}\n\n"
        f"=== CẨM NANG KHUYẾN NÔNG ===\n{ctx_pass}\n\n"
        f"Câu hỏi của nông dân: {question}\n\n"
        "YÊU CẦU BẮT BUỘC:\n"
        "1. Chỉ sử dụng đúng tên hoạt chất, liều lượng và thời gian cách ly có trong phần TRI THỨC ĐỒ THỊ ở trên; tuyệt đối không tự bịa thêm tên hoạt chất hoặc số liệu.\n"
        "2. Nếu liều lượng ghi 'theo hướng dẫn trên nhãn sản phẩm' thì giữ nguyên cụm từ đó.\n"
        "3. Bắt buộc có mục riêng tiêu đề 'Cảnh báo chống chỉ định' trích dẫn đầy đủ mọi dòng CẢNH BÁO trong đồ thị."
    )


def advise(
    question: str,
    disease_id: str,
    kg: dict,
    G: nx.MultiDiGraph,
    index: VectorIndex | None,
    mode: str = "graph",
    generate=ollama_generate,
    embed=ollama_embed,
) -> dict:
    if disease_id == "healthy":
        msg = "Cây cà chua hiện tại khỏe mạnh, không phát hiện dấu hiệu bệnh lý. Không cần phun thuốc bảo vệ thực vật; chỉ cần duy trì tưới nước, bón phân cân đối và thăm đồng định kỳ."
        return {"answer": msg, "facts": [], "passages": [], "context": msg}
    d_obj = next(d for d in kg["diseases"] if d["id"] == disease_id)
    disease_vi = d_obj["name_vi"]
    sub = subgraph(G, disease_id, hops=2)
    fact_lines = facts(sub, disease_id) if mode == "graph" else []
    passages = index.search(f"{disease_vi} {question}", k=4, embed=embed) if (index is not None and mode in ("naive", "graph")) else []
    context = "\n".join(fact_lines + passages)
    prompt = build_prompt(question, disease_vi, mode, fact_lines, passages)
    ans = generate(prompt)
    return {"answer": ans, "facts": fact_lines, "passages": passages, "context": context}
