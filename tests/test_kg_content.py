from pathlib import Path
import networkx as nx
import pytest
from plant_expert.kg import CLASSES, build_graph, likelihood_matrix, load_kg

KG_PATH = Path("kg/kg.json")
MANUAL_DIR = Path("kg/manual")

CONFUSABLE_PAIRS = [
    ("Early_blight", "Target_Spot"),
    ("Septoria_leaf_spot", "Bacterial_spot"),
    ("Late_blight", "Early_blight"),
    ("Leaf_Mold", "Late_blight"),
    ("Yellow_Leaf_Curl_Virus", "Mosaic_virus"),
    ("Spider_mites", "Mosaic_virus"),
]


def test_real_kg_valid():
    kg = load_kg(str(KG_PATH), strict_counts=True)
    assert len(kg["diseases"]) == 10
    for d in kg["diseases"]:
        for t in d.get("treatments", []):
            assert t.get("dose"), f"{d['id']}/{t['ingredient']} missing reference dose"
            assert "25/2024/TT-BNNPTNT" in t["source"], f"{d['id']}/{t['ingredient']} missing Thông tư 25/2024/TT-BNNPTNT"


def test_each_sick_class_has_manual():
    for c in CLASSES:
        if c == "healthy":
            continue
        p = MANUAL_DIR / f"{c}.md"
        assert p.exists(), f"missing manual {p}"
        text = p.read_text(encoding="utf-8").strip()
        assert len(text) > 200
        assert len([para for para in text.split("\n\n") if para.strip()]) >= 3


def test_every_confusable_pair_has_a_separating_symptom():
    kg = load_kg(str(KG_PATH), strict_counts=True)
    _, L = likelihood_matrix(kg, CLASSES)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for a, b in CONFUSABLE_PAIRS:
        diffs = abs(L[:, idx[a]] - L[:, idx[b]])
        assert diffs.max() >= 0.4, f"pair ({a}, {b}) max diff {diffs.max()}"


def test_every_sick_class_has_a_reachable_contraindication():
    kg = load_kg(str(KG_PATH), strict_counts=True)
    G = build_graph(kg)
    for c in CLASSES:
        if c == "healthy":
            continue
        dn = f"disease:{c}"
        reach = nx.single_source_shortest_path_length(G, dn, cutoff=2)
        assert any(n.startswith("contra:") for n in reach), f"{c} has no reachable contraindication"
