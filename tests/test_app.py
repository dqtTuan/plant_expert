from pathlib import Path
import json
import torch
from streamlit.testing.v1 import AppTest

from plant_expert.kg import CLASSES
from plant_expert.vision import build_model

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_without_artifacts_shows_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANT_ARTIFACTS", str(tmp_path))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    assert len(at.error) == 1


def test_app_runs_with_artifacts(tmp_path, monkeypatch):
    m = build_model(len(CLASSES), p_drop=0.3, pretrained=False)
    torch.save(m.state_dict(), tmp_path / "model.pt")
    cfg = {
        "classes": CLASSES,
        "tau": 1.5,
        "p_drop": 0.3,
        "img_size": 64,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "T": 5,
        "bootstrap": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("PLANT_ARTIFACTS", str(tmp_path))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=20)
    assert not at.exception
    assert len(at.error) == 0
