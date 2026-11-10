import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from plant_expert.vision import (
    build_model, diagnose, enable_mc_dropout, load, mc_predict, mutual_info,
    predictive_entropy, preprocess,
)

CFG = {"img_size": 32, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def tiny(C=4):
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16), nn.BatchNorm1d(16),
                         nn.Dropout(0.5), nn.Linear(16, C))


def test_entropy_and_mi_values():
    u = np.ones((1, 4)) / 4
    assert predictive_entropy(u)[0] == pytest.approx(np.log(4))
    same = np.tile(np.array([[[.7, .1, .1, .1]]]), (20, 1, 1))
    assert mutual_info(same)[0] == pytest.approx(0, abs=1e-9)
    split = np.array([[[1., 0.]], [[0., 1.]]])  # two confident, disagreeing passes
    assert mutual_info(split)[0] == pytest.approx(np.log(2), abs=1e-6)


def test_enable_mc_dropout_keeps_bn_eval():
    m = tiny()
    m.train()
    enable_mc_dropout(m)
    assert m[2].training is False
    assert m[3].training is True


def test_mc_predict_shape_and_variance():
    torch.manual_seed(0)
    m = tiny()
    enable_mc_dropout(m)
    probs = mc_predict(m, torch.randn(2, 3, 8, 8), T=20)
    assert probs.shape == (20, 2, 4)
    assert probs.sum(-1) == pytest.approx(np.ones((20, 2)))
    assert probs.std(0).max() > 0


@pytest.mark.parametrize("mode", ["RGBA", "L", "RGB"])
def test_preprocess_modes(mode):
    img = Image.new(mode, (300, 200))
    assert tuple(preprocess(img, CFG).shape) == (1, 3, 32, 32)


def test_build_model_output_shape():
    m = build_model(10).eval()
    with torch.no_grad():
        assert tuple(m(torch.zeros(1, 3, 64, 64)).shape) == (1, 10)


def test_load_roundtrip(tmp_path):
    m = build_model(10, p_drop=0.3)
    torch.save(m.state_dict(), tmp_path / "model.pt")
    cfg = dict(CFG, classes=[str(i) for i in range(10)], tau=0.5, p_drop=0.3, T=5)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    m2, cfg2 = load(str(tmp_path), device="cpu")
    out = diagnose(m2, cfg2, Image.new("RGB", (50, 50)))
    assert out["p_mean"].shape == (10,)
    assert out["probs"].shape == (5, 10)
    assert out["H"] >= 0 and out["MI"] >= -1e-9
