import ast
import json
from pathlib import Path

NB_PATH = Path("notebooks/train_uncertainty.ipynb")

EXPECTED_ARTIFACTS = [
    "model.pt",
    "config.json",
    "test_mc_probs.csv",
    "exp1_metrics.csv",
    "exp1_reliability.png",
    "exp1_risk_coverage.png",
    "exp1_entropy_hist.png",
]


def test_notebook_cells_parse():
    assert NB_PATH.exists(), "notebook missing"
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 6
    for i, c in enumerate(code_cells):
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        clean = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        ast.parse(clean, filename=f"cell_{i}")


def test_notebook_writes_expected_artifacts():
    raw = NB_PATH.read_text(encoding="utf-8")
    for name in EXPECTED_ARTIFACTS:
        assert name in raw, f"missing artifact {name} in notebook"
