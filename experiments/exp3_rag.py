"""Experiment 3: evaluate LLM-only vs Naive RAG vs Agricultural GraphRAG."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plant_expert.graphrag import VectorIndex, advise, ollama_embed, ollama_generate
from plant_expert.kg import build_graph, load_kg


def mentioned_ingredients(text: str, vocab: list[str]) -> set[str]:
    low = text.casefold()
    return {w.casefold() for w in vocab if w.casefold() in low}


def score(answer: str, q: dict, kg: dict, context: str, vocab: list[str]) -> dict:
    d_obj = next(d for d in kg["diseases"] if d["id"] == q["disease"])
    ing_name_by_id = {i["id"]: i["name"].casefold() for i in kg["ingredients"]}
    valid = {ing_name_by_id[t["ingredient"]] for t in d_obj.get("treatments", [])}
    mentioned = mentioned_ingredients(answer, vocab)
    in_question = mentioned_ingredients(q.get("question", ""), vocab)
    halluc = any(m not in valid for m in (mentioned - in_question))
    low_ans = answer.casefold()
    contra_hit = all(kw.casefold() in low_ans for kw in q.get("must_contra_keywords", []))
    if not mentioned:
        faith = 1.0
    else:
        in_ctx = mentioned_ingredients(context, vocab)
        faith = len(mentioned & in_ctx) / len(mentioned)
    return {
        "halluc": bool(halluc),
        "contra_hit": bool(contra_hit),
        "faithfulness": float(faith),
        "mentioned": sorted(mentioned),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="experiments/exp3_questions.json")
    ap.add_argument("--kg", default="kg/kg.json")
    ap.add_argument("--index", default="kg/index.npz")
    args = ap.parse_args()

    kg = load_kg(args.kg)
    G = build_graph(kg)
    idx = VectorIndex.load(args.index) if Path(args.index).exists() else VectorIndex.build("kg/manual")
    q_data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    vocab = q_data["known_ingredients"]
    questions = q_data["questions"]

    out_dir = Path("experiments/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    ans_file = out_dir / "exp3_answers.jsonl"

    rows = []
    with open(ans_file, "w", encoding="utf-8") as f:
        for mode in ("llm", "naive", "graph"):
            for q in questions:
                res = advise(q["question"], q["disease"], kg, G, idx, mode=mode,
                             generate=ollama_generate, embed=ollama_embed)
                sc = score(res["answer"], q, kg, res["context"], vocab)
                rec = {"mode": mode, "id": q["id"], "disease": q["disease"], "trap": q["trap"],
                       "answer": res["answer"], **sc}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows.append(rec)

    df = pd.DataFrame(rows)
    summary = df.groupby("mode", as_index=False).agg(
        faithfulness=("faithfulness", "mean"),
        contra_hit_rate=("contra_hit", "mean"),
        halluc_rate=("halluc", "mean"),
    )
    summary.to_csv(out_dir / "exp3_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
