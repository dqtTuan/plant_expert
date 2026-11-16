import ast
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from sklearn.model_selection import train_test_split

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
    assert "np.trapz(" not in raw, "np.trapz removed in NumPy 2.0+; use np.trapezoid"


def test_notebook_recovers_incomplete_plantvillage_checkout(tmp_path, monkeypatch):
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in nb["cells"]]
    source = next(src for src in sources if "PV_MAP =" in src)
    monkeypatch.chdir(tmp_path)
    village = Path("data/PlantVillage-Dataset")
    (village / ".git").mkdir(parents=True)
    Path("data/PlantDoc-Dataset").mkdir()

    def finish_checkout(args, *, check):
        assert check
        assert args[:6] == ["git", "-C", str(village), "sparse-checkout", "set", "--cone"]
        assert len(args[6:]) == 10
        for folder in args[6:]:
            path = village / folder
            path.mkdir(parents=True)
            for i in range(20):
                (path / f"{i}.jpg").touch()

    git = Mock(side_effect=finish_checkout)
    namespace = dict(Path=Path, subprocess=Mock(run=git), train_test_split=train_test_split, SMOKE=True, SEED=42)
    exec(source, namespace)

    git.assert_called_once()
    for split in ("train_items", "val_items", "test_items"):
        assert len({label for _, label in namespace[split]}) == 10


@pytest.mark.parametrize("village_mount, doc_mount", [
    ("plantvillage-dataset/color", "plant-doc-dataset"),
    ("datasets/abdallahalidev/plantvillage-dataset/color", "datasets/abdulhasibuddin/plant-doc-dataset"),
    ("datasets/emmarex/plantdisease/PlantVillage", "datasets/abdulhasibuddin/plant-doc-dataset"),
])
def test_notebook_loads_both_kaggle_datasets(tmp_path, monkeypatch, village_mount, doc_mount):
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in nb["cells"]]
    source = next(src for src in sources if "PV_MAP =" in src)
    map_assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                          and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "PV_MAP")
    monkeypatch.chdir(tmp_path)
    Path("data/PlantVillage-Dataset/raw/color").mkdir(parents=True)
    color = tmp_path / "input" / village_mount
    plantdoc = tmp_path / "input" / doc_mount / "PlantDoc-Dataset"
    test_leaf = plantdoc / "test/Tomato leaf/leaf.jpg"
    test_leaf.parent.mkdir(parents=True)
    test_leaf.touch()
    folders = ast.literal_eval(map_assignment.value)
    if color.name == "PlantVillage":
        folders = [
            "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
            "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
            "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy",
        ]
        (tmp_path / "input/plantvillage-dataset/color").mkdir(parents=True)
    for folder in folders:
        path = color / folder
        path.mkdir(parents=True)
        for i in range(20):
            (path / f"{i}.JPG").touch()

    git = Mock()
    result = dict(
        Path=lambda value: tmp_path / "input" if value == "/kaggle/input" else Path(value),
        subprocess=Mock(run=git), train_test_split=train_test_split, SMOKE=True, SEED=42,
    )
    exec(source, result)

    git.assert_not_called()
    assert result["pv_color"] == color
    assert result["pd_items"] == [(str(test_leaf), "healthy")]
    assert sum(len(result[split]) for split in ("train_items", "val_items", "test_items")) == 200

