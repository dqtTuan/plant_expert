import copy
import json
from pathlib import Path

import pytest

from plant_expert.kg import CLASSES, build_graph, likelihood_matrix, validate

MINI = Path(__file__).parent / "fixtures" / "mini_kg.json"


@pytest.fixture
def kg():
    return json.loads(MINI.read_text(encoding="utf-8"))


def sym(kg, sid):
    return next(s for s in kg["context_symptoms"] if s["id"] == sid)


def test_mini_kg_valid(kg):
    assert validate(kg, strict_counts=False) == []


def test_missing_class_prob_reported(kg):
    del sym(kg, "s01")["p"]["healthy"]
    assert any("s01" in e and "healthy" in e for e in validate(kg, strict_counts=False))


def test_prob_out_of_range_reported(kg):
    sym(kg, "s01")["p"]["healthy"] = 1.2
    assert any("s01" in e for e in validate(kg, strict_counts=False))


def test_treatment_without_source_reported(kg):
    kg["diseases"][2]["treatments"][0]["source"] = ""
    assert any("source" in e for e in validate(kg, strict_counts=False))


def test_unknown_ingredient_reported(kg):
    kg["diseases"][2]["treatments"][0]["ingredient"] = "nope"
    assert any("nope" in e for e in validate(kg, strict_counts=False))


def test_virus_with_fungicide_reported(kg):
    vir = next(d for d in kg["diseases"] if d["id"] == "Mosaic_virus")
    vir["treatments"] = [{"ingredient": "mancozeb", "dose": None, "phi_days": None, "source": "x"}]
    assert any("Mosaic_virus" in e for e in validate(kg, strict_counts=False))


def test_bacteria_needs_bactericide(kg):
    bac = next(d for d in kg["diseases"] if d["id"] == "Bacterial_spot")
    bac["treatments"] = [{"ingredient": "mancozeb", "dose": None, "phi_days": None, "source": "x"}]
    assert any("Bacterial_spot" in e for e in validate(kg, strict_counts=False))


def test_strict_counts_requires_15_to_20_symptoms_and_10_diseases(kg):
    assert any("symptom" in e for e in validate(kg, strict_counts=True))
    k2 = copy.deepcopy(kg)
    k2["diseases"].pop()
    assert any("healthy" in e for e in validate(k2, strict_counts=False))


def test_build_graph_edges(kg):
    G = build_graph(kg)
    rels = {(u, v, d["rel"]) for u, v, d in G.edges(data=True)}
    assert ("disease:Late_blight", "ingredient:mancozeb", "TREATED_BY") in rels
    assert ("ingredient:mancozeb", "contra:c01", "RESTRICTED_WITH") in rels
    assert ("disease:Late_blight", "pathogen:Phytophthora infestans", "CAUSED_BY") in rels
    assert G.nodes["context:s01"]["type"] == "ContextSymptom"


def test_likelihood_matrix_shape_and_order(kg):
    ids, L = likelihood_matrix(kg, CLASSES)
    assert ids == ["s01", "s02"]
    assert L.shape == (2, 10)
    assert L[0, 9] == pytest.approx(0.2)
    assert L[1, 7] == pytest.approx(0.9)
