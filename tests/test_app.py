from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_without_artifacts_shows_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANT_ARTIFACTS", str(tmp_path))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    assert len(at.error) == 1
