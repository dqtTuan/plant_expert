import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from plant_expert.kg import build_graph
from plant_expert.graphrag import (
    OllamaError, VectorIndex, advise, build_prompt, facts, ollama_generate, subgraph,
)

MINI = Path(__file__).parent / "fixtures" / "mini_kg.json"


@pytest.fixture
def kg():
    return json.loads(MINI.read_text(encoding="utf-8"))


def fake_embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        v = np.zeros(32, dtype=float)
        for ch in t.lower():
            v[ord(ch) % 32] += 1.0
        n = np.linalg.norm(v) or 1.0
        vecs.append(v / n)
    return np.stack(vecs, axis=0)


def test_subgraph_reaches_contraindication(kg):
    G = build_graph(kg)
    sub = subgraph(G, "Late_blight", hops=2)
    assert "contra:c01" in sub.nodes
    assert "ingredient:mancozeb" in sub.nodes


def test_subgraph_excludes_context_nodes(kg):
    G = build_graph(kg)
    sub = subgraph(G, "Late_blight", hops=2)
    assert not any(n.startswith("context:") for n in sub.nodes)
    assert [n for n in sub.nodes if n.startswith("disease:")] == ["disease:Late_blight"]


def test_index_dim_mismatch_raises_ollama_error(tmp_path):
    mdir = tmp_path / "manual"
    mdir.mkdir()
    (mdir / "Late_blight.md").write_text("Đoạn 1\n\nĐoạn 2\n\nĐoạn 3", encoding="utf-8")
    idx = VectorIndex.build(str(mdir), embed=fake_embed)
    with pytest.raises(OllamaError):
        idx.search("mốc sương", k=1, embed=lambda ts: np.ones((len(ts), 64)))



def test_facts_has_warning_and_label_fallback(kg):
    G = build_graph(kg)
    sub = subgraph(G, "Late_blight", hops=2)
    flist = facts(sub, "Late_blight")
    joined = "\n".join(flist)
    assert "CẢNH BÁO:" in joined
    assert "theo hướng dẫn trên nhãn sản phẩm" in joined
    assert "Mancozeb" in joined


def test_prompt_modes():
    p_llm = build_prompt("Phun thuốc gì?", "Bệnh mốc sương", "llm", ["FACT_X"], ["PASS_Y"])
    assert "FACT_X" not in p_llm and "PASS_Y" not in p_llm
    p_naive = build_prompt("Phun thuốc gì?", "Bệnh mốc sương", "naive", ["FACT_X"], ["PASS_Y"])
    assert "PASS_Y" in p_naive and "FACT_X" not in p_naive
    p_graph = build_prompt("Phun thuốc gì?", "Bệnh mốc sương", "graph", ["FACT_X"], ["PASS_Y"])
    assert "FACT_X" in p_graph and "PASS_Y" in p_graph
    assert "Cảnh báo chống chỉ định" in p_graph


def test_healthy_skips_llm(kg, tmp_path):
    G = build_graph(kg)
    (tmp_path / "Late_blight.md").write_text("Đoạn 1\n\nĐoạn 2\n\nĐoạn 3", encoding="utf-8")
    idx = VectorIndex.build(str(tmp_path), embed=fake_embed)
    gen = Mock()
    res = advise("Cây có bệnh gì không?", "healthy", kg, G, idx, generate=gen, embed=fake_embed)
    gen.assert_not_called()
    assert "khỏe" in res["answer"].lower()


def test_ollama_down_raises():
    with pytest.raises(OllamaError):
        ollama_generate("xin chào", host="http://127.0.0.1:9")


def test_index_roundtrip(tmp_path):
    mdir = tmp_path / "manual"
    mdir.mkdir()
    (mdir / "Late_blight.md").write_text(
        "Mốc sương do Phytophthora.\n\nDùng Mancozeb hoặc Metalaxyl.\n\nKhông pha kiềm.",
        encoding="utf-8",
    )
    idx = VectorIndex.build(str(mdir), embed=fake_embed)
    idx_path = tmp_path / "index.npz"
    idx.save(str(idx_path))
    idx2 = VectorIndex.load(str(idx_path))
    hits = idx2.search("Mancozeb trị mốc sương", k=2, embed=fake_embed)
    assert len(hits) == 2
    assert all(isinstance(h, str) for h in hits)


def test_advise_offline_fallback(kg, tmp_path):
    mdir = tmp_path / "manual"
    mdir.mkdir()
    (mdir / "Late_blight.md").write_text("Mốc sương do Phytophthora.\n\nDùng Mancozeb.", encoding="utf-8")
    idx = VectorIndex.build(str(mdir), embed=fake_embed)
    G = build_graph(kg)

    def boom(_prompt: str) -> str:
        raise OllamaError("offline")

    res = advise("Cách trị?", "Late_blight", kg, G, idx, mode="graph", generate=boom, embed=fake_embed, allow_fallback=True)
    assert res["fallback"] is True
    assert "Cảnh báo chống chỉ định" in res["answer"]
    assert "Mancozeb" in res["answer"]
