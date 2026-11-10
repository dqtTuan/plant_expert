"""Vision + MC Dropout inference for tomato leaf diagnosis."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

EPS = 1e-12


def build_model(num_classes: int, p_drop: float = 0.3, pretrained: bool = False) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    m = models.convnext_tiny(weights=weights)
    in_features = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(nn.Dropout(p_drop), nn.Linear(in_features, num_classes))
    return m


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()


@torch.no_grad()
def mc_predict(model: nn.Module, x: torch.Tensor, T: int = 20) -> np.ndarray:
    enable_mc_dropout(model)
    out = [torch.softmax(model(x), dim=-1).cpu().numpy() for _ in range(T)]
    return np.stack(out, axis=0)


def predictive_entropy(p_mean: np.ndarray) -> np.ndarray:
    p = np.asarray(p_mean, dtype=float)
    return -(p * np.log(p + EPS)).sum(axis=-1)


def mutual_info(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    h_mean = predictive_entropy(probs.mean(axis=0))
    mean_h = -(probs * np.log(probs + EPS)).sum(axis=-1).mean(axis=0)
    return np.maximum(h_mean - mean_h, 0.0)


def preprocess(img: Image.Image, cfg: dict) -> torch.Tensor:
    s = cfg["img_size"]
    tf = transforms.Compose([
        transforms.Resize((s, s)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])
    return tf(img.convert("RGB")).unsqueeze(0)


def load(artifacts_dir: str = "artifacts", device: str | None = None) -> tuple[nn.Module, dict]:
    p = Path(artifacts_dir)
    cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model(len(cfg["classes"]), p_drop=cfg.get("p_drop", 0.3), pretrained=False)
    m.load_state_dict(torch.load(p / "model.pt", map_location=dev, weights_only=True))
    m.to(dev).eval()
    cfg["device"] = dev
    return m, cfg


def diagnose(model: nn.Module, cfg: dict, img: Image.Image) -> dict:
    dev = cfg.get("device", next(model.parameters()).device)
    x = preprocess(img, cfg).to(dev)
    probs = mc_predict(model, x, T=cfg.get("T", 20))[:, 0, :]  # [T, C]
    p_mean = probs.mean(axis=0)
    return {
        "p_mean": p_mean,
        "H": float(predictive_entropy(p_mean)),
        "MI": float(mutual_info(probs[:, None, :])[0]),
        "probs": probs,
    }
