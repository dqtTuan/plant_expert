"""Build and save the vector index from kg/manual/*.md (uses Ollama bge-m3 if running, else char-gram fallback)."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plant_expert.graphrag import OllamaError, VectorIndex, ollama_embed


def char_ngram_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    # shortcut: deterministic offline fallback when Ollama is not installed yet, rebuild with `bge-m3` once Ollama runs
    vecs = np.zeros((len(texts), dim), dtype=float)
    for i, t in enumerate(texts):
        s = t.lower()
        for j in range(len(s) - 2):
            h = hash(s[j:j + 3]) % dim
            vecs[i, h] += 1.0
        n = np.linalg.norm(vecs[i]) or 1.0
        vecs[i] /= n
    return vecs


if __name__ == "__main__":
    try:
        idx = VectorIndex.build("kg/manual", embed=ollama_embed)
        print("Built index with Ollama bge-m3:", len(idx.chunks), "chunks")
    except OllamaError as e:
        print("Ollama not running, building fallback index:", e)
        idx = VectorIndex.build("kg/manual", embed=char_ngram_embed)
    idx.save("kg/index.npz")
    print("Saved kg/index.npz")
