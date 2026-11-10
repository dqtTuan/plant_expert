import json
from pathlib import Path

import pytest

from plant_expert.kg import CLASSES, load_kg
from experiments.exp3_rag import mentioned_ingredients, score

Q_PATH = Path("experiments/exp3_questions.json")


def test_mentioned_ingredients_casefold():
    assert mentioned_ingredients("Nên phun MANCOZEB 80WP luân phiên", ["mancozeb", "metalaxyl"]) == {"mancozeb"}


def test_score_halluc():
    kg = load_kg("kg/kg.json")
    q = {"id": "q1", "disease": "Late_blight", "question": "?", "must_contra_keywords": ["kiềm"], "trap": False}
    vocab = ["mancozeb", "metalaxyl", "cymoxanil", "abamectin"]
    # abamectin is not a Late_blight treatment -> halluc True
    s_bad = score("Phun Abamectin và không pha kiềm", q, kg, "Hoạt chất Mancozeb", vocab)
    assert s_bad["halluc"] is True
    s_ok = score("Phun Mancozeb, không pha với thuốc có tính kiềm", q, kg, "Hoạt chất Mancozeb", vocab)
    assert s_ok["halluc"] is False


def test_trap_question_refusal_not_counted_as_halluc():
    kg = load_kg("kg/kg.json")
    q = {
        "id": "q08",
        "disease": "Bacterial_spot",
        "question": "Cà chua bị đốm vi khuẩn, tôi lấy thuốc trừ nấm Mancozeb ra phun có khỏi không?",
        "must_contra_keywords": ["kiềm"],
        "trap": True,
    }
    vocab = ["mancozeb", "copper hydroxide", "kasugamycin"]
    ans = "Không nên phun Mancozeb vì đây là bệnh vi khuẩn. Hãy dùng Copper hydroxide và không pha kiềm."
    assert score(ans, q, kg, "Copper hydroxide", vocab)["halluc"] is False



def test_score_contra_hit():
    kg = load_kg("kg/kg.json")
    q = {"id": "q1", "disease": "Late_blight", "question": "?", "must_contra_keywords": ["kiềm", "kháng thuốc"], "trap": False}
    vocab = ["mancozeb"]
    assert score("Không pha kiềm và luân phiên tránh kháng thuốc", q, kg, "mancozeb", vocab)["contra_hit"] is True
    assert score("Chỉ lưu ý không pha kiềm", q, kg, "mancozeb", vocab)["contra_hit"] is False


def test_faithfulness_no_mentions_is_one():
    kg = load_kg("kg/kg.json")
    q = {"id": "q1", "disease": "Late_blight", "question": "?", "must_contra_keywords": [], "trap": False}
    assert score("Tỉa bỏ lá bệnh và vệ sinh đồng ruộng", q, kg, "", ["mancozeb"])["faithfulness"] == pytest.approx(1.0)


def test_questions_file_valid():
    assert Q_PATH.exists(), "exp3_questions.json missing"
    data = json.loads(Q_PATH.read_text(encoding="utf-8"))
    qs = data["questions"]
    assert len(qs) >= 30
    assert sum(1 for q in qs if q.get("trap")) >= 8
    assert len(data["known_ingredients"]) >= 30
    for q in qs:
        assert q["disease"] in CLASSES and q["disease"] != "healthy"
        assert isinstance(q["must_contra_keywords"], list) and len(q["must_contra_keywords"]) >= 1
