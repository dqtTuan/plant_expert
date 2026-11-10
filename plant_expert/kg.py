"""Agricultural knowledge graph: load, validate, convert to networkx, likelihood matrix."""
import json

import networkx as nx
import numpy as np

CLASSES = ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot",
           "Spider_mites", "Target_Spot", "Yellow_Leaf_Curl_Virus", "Mosaic_virus", "healthy"]

# which ingredient class a treatment must carry, per disease group (spec anti-goal: no fungicide for virus/bacteria)
REQUIRED_CLASS = {"bacteria": "bactericide", "virus": "insecticide", "pest": "acaricide"}


def validate(kg: dict, strict_counts: bool = True) -> list[str]:
    errs = []
    dids = [d["id"] for d in kg.get("diseases", [])]
    for c in CLASSES:
        if c not in dids:
            errs.append(f"missing disease {c}")
    ing = {i["id"]: i for i in kg.get("ingredients", [])}
    for i in ing.values():
        if not i.get("source"):
            errs.append(f"ingredient {i['id']} has no source")
    for d in kg.get("diseases", []):
        if d["group"] == "none" and d.get("treatments"):
            errs.append(f"{d['id']}: healthy class must have no treatments")
        for t in d.get("treatments", []):
            if t["ingredient"] not in ing:
                errs.append(f"{d['id']}: unknown ingredient {t['ingredient']}")
                continue
            if not t.get("source"):
                errs.append(f"{d['id']}/{t['ingredient']}: treatment has no source")
            need = REQUIRED_CLASS.get(d["group"])
            if need and need not in ing[t["ingredient"]]["classes"]:
                errs.append(f"{d['id']} ({d['group']}): {t['ingredient']} is not a {need}")
    for c in kg.get("contraindications", []):
        if not c.get("source"):
            errs.append(f"contraindication {c['id']} has no source")
        for i in c.get("ingredients", []):
            if i not in ing:
                errs.append(f"contraindication {c['id']}: unknown ingredient {i}")
        for d in c.get("diseases", []):
            if d not in dids:
                errs.append(f"contraindication {c['id']}: unknown disease {d}")
    syms = kg.get("context_symptoms", [])
    for s in syms:
        for c in CLASSES:
            v = s.get("p", {}).get(c)
            if v is None:
                errs.append(f"symptom {s['id']}: missing p for {c}")
            elif not 0 <= v <= 1:
                errs.append(f"symptom {s['id']}: p[{c}]={v} out of [0,1]")
    if strict_counts:
        if len(dids) != 10:
            errs.append(f"expected 10 diseases, got {len(dids)}")
        if not 15 <= len(syms) <= 20:
            errs.append(f"expected 15-20 context symptoms, got {len(syms)}")
    return errs


def load_kg(path: str = "kg/kg.json", strict_counts: bool = True) -> dict:
    with open(path, encoding="utf-8") as f:
        kg = json.load(f)
    errs = validate(kg, strict_counts)
    if errs:
        raise ValueError("Invalid KG:\n" + "\n".join(errs))
    return kg


def build_graph(kg: dict) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    ing = {i["id"]: i for i in kg["ingredients"]}
    for i in ing.values():
        G.add_node(f"ingredient:{i['id']}", type="ActiveIngredient", label=i["name"], **{k: v for k, v in i.items() if k not in ("id", "name")})
    for d in kg["diseases"]:
        dn = f"disease:{d['id']}"
        G.add_node(dn, type="Disease", label=d["name_vi"], name_en=d["name_en"], group=d["group"])
        if d.get("pathogen"):
            G.add_node(f"pathogen:{d['pathogen']}", type="Pathogen", label=d["pathogen"])
            G.add_edge(dn, f"pathogen:{d['pathogen']}", rel="CAUSED_BY")
        for n, sign in enumerate(d.get("visual_signs", [])):
            vn = f"visual:{d['id']}:{n}"
            G.add_node(vn, type="VisualSymptom", label=sign)
            G.add_edge(dn, vn, rel="HAS_VISUAL_SIGN")
        for t in d.get("treatments", []):
            G.add_edge(dn, f"ingredient:{t['ingredient']}", rel="TREATED_BY",
                       dose=t.get("dose"), phi_days=t.get("phi_days"), source=t.get("source"))
    for c in kg["contraindications"]:
        cn = f"contra:{c['id']}"
        G.add_node(cn, type="Contraindication", label=c["text_vi"], source=c.get("source"))
        for i in c.get("ingredients", []):
            G.add_edge(f"ingredient:{i}", cn, rel="RESTRICTED_WITH")
        for d in c.get("diseases", []):
            G.add_edge(f"disease:{d}", cn, rel="RESTRICTED_WITH")
    for s in kg["context_symptoms"]:
        sn = f"context:{s['id']}"
        G.add_node(sn, type="ContextSymptom", label=s["question_vi"], obs_type=s.get("type"))
        for d, p in s["p"].items():
            G.add_edge(f"disease:{d}", sn, rel="HAS_CONTEXT_SIGN", p=p)
    return G


def likelihood_matrix(kg: dict, classes: list[str]) -> tuple[list[str], np.ndarray]:
    syms = kg["context_symptoms"]
    return [s["id"] for s in syms], np.array([[s["p"][c] for c in classes] for s in syms], dtype=float)
