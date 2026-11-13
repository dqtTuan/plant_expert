"""Build and save the vector index from kg/manual/*.md (uses Ollama bge-m3 if running, else char-gram fallback)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plant_expert.graphrag import OllamaError, VectorIndex, char_ngram_embed, ollama_embed


if __name__ == "__main__":
    try:
        idx = VectorIndex.build("kg/manual", embed=ollama_embed)
        print("Built index with Ollama bge-m3:", len(idx.chunks), "chunks")
    except OllamaError as e:
        print("Ollama not running, building fallback index:", e)
        idx = VectorIndex.build("kg/manual", embed=char_ngram_embed)
    idx.save("kg/index.npz")
    print("Saved kg/index.npz")
