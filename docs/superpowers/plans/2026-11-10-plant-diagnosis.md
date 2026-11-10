# Plant Diagnosis (Uncertainty + Bayes + GraphRAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tomato-leaf diagnosis system that measures uncertainty with MC Dropout, asks follow-up questions with Bayesian updates when unsure, and gives treatment advice through GraphRAG. Deliverables: 3 reproducible experiments and a Streamlit demo.

**Architecture:** The model is trained in one Colab notebook, which exports `artifacts/model.pt`, `artifacts/config.json` and `artifacts/test_mc_probs.csv`. Everything else runs locally from a small package `plant_expert/`:
- `kg.py`: knowledge graph
- `vision.py`: MC Dropout inference
- `bayes.py`: question selection and updates
- `graphrag.py`: retrieval and the Ollama call

These are used by `experiments/`, and by `app.py` for the demo.

**Tech Stack:** Python 3.12, PyTorch + torchvision (ConvNeXt-Tiny), numpy, networkx, Streamlit, plotly, pyvis, pandas, scikit-learn (AUROC only), matplotlib, pytest. Ollama with `qwen2.5:7b-instruct` and `bge-m3`, called through stdlib `urllib`.

**Spec:** `docs/superpowers/specs/2026-11-10-plant-diagnosis-design.md`

## Global Constraints

- Class ids, in this exact order everywhere: `["Bacterial_spot","Early_blight","Late_blight","Leaf_Mold","Septoria_leaf_spot","Spider_mites","Target_Spot","Yellow_Leaf_Curl_Virus","Mosaic_virus","healthy"]`. `config.json["classes"]` is the source of truth at runtime.
- All entropies use the natural log, with eps `1e-12`.
- MC Dropout: T = 20. Norm layers stay in eval mode; only `nn.Dropout` runs in train mode.
- Bayes: k = 3 candidates (drop the 3rd if its prob < 0.05). Confirm at posterior ≥ 0.85. At most 3 questions. "unknown" means no update.
- Answers are always one of `"yes" | "no" | "unknown"`. The UI labels are Có / Không / Không rõ.
- Everything runs offline. Only the Colab notebook downloads data.
- All user-facing text is Vietnamese.
- Never invent a dose or pre-harvest interval. A null in the KG means the answer says "theo hướng dẫn trên nhãn sản phẩm".
- P(s|c) is labelled "ước lượng chuyên gia" wherever it is shown.
- Local Python: the Windows Store `python` works only outside the sandbox. Create `.venv` with `python -m venv .venv` and run everything through `.venv\Scripts\python`.

## Review Focus

1. **Ollama is not running, or a model has not been pulled.** `generate` raises `OllamaError`. The app shows a Vietnamese error, and the diagnosis and charts still show (Task 6, Task 9).
2. **Odd uploads** (PNG with alpha, grayscale, very large image). `preprocess` converts to RGB and resizes, with no crash (Task 4).
3. **The user answers "Không rõ" to every question.** The dialogue stops (questions run out, or 3 are asked), returns the argmax, and is flagged as not confirmed (Task 3).
4. **The true class falls outside the top-k candidates in exp2.** It is counted as wrong, with no crash (Task 7).
5. **The final disease is `healthy`.** No treatment and no LLM call. The app says the plant is healthy (Task 6, Task 9).

---

### Task 1: Project skeleton + KG loader/validator

**Files:**
- Create: `requirements.txt`, `.gitignore` (`.venv/`, `artifacts/`, `data/`, `__pycache__/`, `kg/index.npz`), `plant_expert/__init__.py`, `plant_expert/kg.py`
- Test: `tests/test_kg.py`, `tests/fixtures/mini_kg.json` (a valid KG with the 10 classes, 2 context symptoms and 1 ingredient)

**KG JSON schema** (`kg/kg.json`):
```json
{
  "diseases": [{"id": "Late_blight", "name_vi": "...", "name_en": "...",
                "group": "fungus|oomycete|bacteria|virus|pest|none",
                "pathogen": "Phytophthora infestans" , "visual_signs": ["..."],
                "treatments": [{"ingredient": "mancozeb", "dose": "string|null", "phi_days": 7, "source": "..."}]}],
  "ingredients": [{"id": "mancozeb", "name": "Mancozeb", "classes": ["fungicide"], "toxicity_group": "III", "source": "..."}],
  "contraindications": [{"id": "c01", "text_vi": "...", "ingredients": ["..."], "diseases": ["..."], "source": "..."}],
  "context_symptoms": [{"id": "s01", "question_vi": "...", "type": "stem|root|weather|pest|position|other",
                        "p": {"Bacterial_spot": 0.1, "...all 10 classes...": 0.0}}]
}
```

**Interfaces:**
- Produces:
  - `CLASSES: list[str]` (the Global Constraints order)
  - `load_kg(path: str = "kg/kg.json", strict_counts: bool = True) -> dict` raises `ValueError` listing every problem
  - `validate(kg: dict, strict_counts: bool = True) -> list[str]`
  - `build_graph(kg: dict) -> nx.MultiDiGraph`. Node ids are `"disease:<id>"`, `"pathogen:<name>"`, `"visual:<disease>:<i>"`, `"context:<sid>"`, `"ingredient:<id>"`, `"contra:<id>"`. Node attrs are `type`, `label`. Edge attr `rel` is one of CAUSED_BY, HAS_VISUAL_SIGN, HAS_CONTEXT_SIGN (+`p`), TREATED_BY (+`dose`, `phi_days`, `source`), RESTRICTED_WITH (ingredient→contra and disease→contra).
  - `likelihood_matrix(kg: dict, classes: list[str]) -> tuple[list[str], np.ndarray]`: symptom ids plus an `[M, C]` array of P(s=yes|c).

- [ ] **Step 1: Set up the venv and requirements.** `python -m venv .venv`, then `.venv\Scripts\pip install -r requirements.txt` (torch is the CPU/CUDA default wheel). Expected: `.venv\Scripts\python -c "import torch, networkx, streamlit"` exits 0.
- [ ] **Step 2: Write failing tests**
```python
def test_mini_kg_valid(): assert validate(json.load(open(MINI)), strict_counts=False) == []
def test_missing_class_prob_reported():  # delete p["healthy"] of s01
    assert any("s01" in e and "healthy" in e for e in validate(kg, strict_counts=False))
def test_prob_out_of_range_reported():   # p = 1.2
def test_treatment_without_source_reported():
def test_unknown_ingredient_reported():
def test_virus_with_fungicide_reported():   # rule: virus→only "insecticide"; bacteria→must include "bactericide"; pest→"acaricide"; none→no treatments
def test_strict_counts_requires_15_to_20_symptoms_and_10_diseases():
def test_build_graph_edges():  # disease:Late_blight -TREATED_BY-> ingredient:mancozeb -RESTRICTED_WITH-> contra:c01
def test_likelihood_matrix_shape_and_order():  # shape (2,10), column 9 == p["healthy"]
```
- [ ] **Step 3: Run** `.venv\Scripts\python -m pytest tests/test_kg.py -v`. Expected: FAIL (import error).
- [ ] **Step 4: Implement `plant_expert/kg.py`** to the interfaces above.
- [ ] **Step 5: Run tests.** Expected: PASS.
- [ ] **Step 6: Commit** `feat: project skeleton and KG loader/validator`

### Task 2: Author KG content + extension manual

**Files:**
- Create: `kg/kg.json`, `kg/manual/<class_id>.md` (one per non-healthy class, 3–6 paragraphs: symptoms, conditions that favour the disease, cultural control, chemical control, safety), `kg/SOURCES.md` (full citations)
- Test: `tests/test_kg_content.py`

**Interfaces:**
- Consumes: `load_kg`, `build_graph` (Task 1)
- Produces: `kg/kg.json` that passes `load_kg(strict_counts=True)`, and `kg/manual/*.md`.

Content rules (from spec §5):
- Ingredients must be on the current Vietnamese "Danh mục thuốc BVTV được phép sử dụng" for tomato or vegetables. Use web search. Every ingredient, treatment and contraindication has a `source`.
- Dose and PHI are per (ingredient, disease). Use null when unsourced.
- Viruses (TYLCV, mosaic): treatments are vector insecticides only, plus a contraindication against fungicides and antibiotics. `healthy` has no treatments.
- 15–20 context symptoms that separate the confusable pairs: Early vs Target Spot; Septoria vs Bacterial spot; Late blight vs Early blight; Leaf Mold vs Late blight; TYLCV vs Mosaic; Spider mites vs Mosaic. P values in [0.02, 0.98] ("thường gặp" ≈ 0.8, "hiếm" ≈ 0.1). Write the "ước lượng chuyên gia" note in `SOURCES.md`.

- [ ] **Step 1: Write failing tests**
```python
def test_real_kg_valid(): load_kg("kg/kg.json")            # strict
def test_each_sick_class_has_manual(): ...all 9 files exist and are non-empty
def test_every_confusable_pair_has_a_separating_symptom():  # some s with |p[a]-p[b]| >= 0.4 for each pair above
def test_every_sick_class_has_a_reachable_contraindication(): # via build_graph, within 2 hops
```
- [ ] **Step 2: Run.** Expected: FAIL (no file).
- [ ] **Step 3: Research and write `kg/kg.json`, the manuals and `SOURCES.md`.**
- [ ] **Step 4: Run** `pytest tests/test_kg_content.py tests/test_kg.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: tomato knowledge graph and extension manual`

### Task 3: Bayesian questioning (`bayes.py`)

**Files:** Create `plant_expert/bayes.py`. Test: `tests/test_bayes.py`

**Interfaces:**
- Consumes: the `L` matrix from `likelihood_matrix` (`[M, C]`)
- Produces:
  - `entropy(p: np.ndarray) -> float`
  - `candidate_prior(p: np.ndarray, k: int = 3, min_p: float = 0.05) -> tuple[np.ndarray, np.ndarray]`: (class indices sorted by prob desc, renormalized prior)
  - `info_gain(prior: np.ndarray, lik: np.ndarray) -> float`: lik = P(yes|c) for the candidates
  - `update(prior, lik, answer: str) -> np.ndarray`
  - `@dataclass State: cand: np.ndarray; post: np.ndarray; asked: list[int]; history: list[tuple[int, str, np.ndarray]]; gated: bool`
  - `start(p: np.ndarray, H: float, tau: float, k: int = 3) -> State` (`gated=False` means H ≤ τ: no questions, `post` = full p)
  - `next_question(st: State, L: np.ndarray, strategy: str = "ig", rng: np.random.Generator | None = None) -> int | None`. `"random"` picks uniformly among unasked symptoms. Returns None when done.
  - `answer(st: State, s: int, ans: str, L: np.ndarray) -> State`
  - `is_done(st: State, conf: float = 0.85, max_q: int = 3) -> bool`
  - `final(st: State) -> tuple[int, float, bool]`: (class index, prob, confirmed = prob ≥ 0.85 or not gated)
  - `run_dialogue(p, H, tau, L, answer_fn: Callable[[int], str], strategy="ig", rng=None) -> tuple[int, int]`: (pred class index, #questions asked)

`next_question` returns None if `is_done` holds, if every symptom has been asked, or (for "ig") if max IG ≤ 1e-9. Count a question as asked even when the answer is "unknown".

- [ ] **Step 1: Write failing tests**
```python
def test_entropy_uniform(): assert entropy(np.ones(4)/4) == pytest.approx(np.log(4))
def test_info_gain_perfect_split(): assert info_gain(np.array([.5,.5]), np.array([1.,0.])) == pytest.approx(np.log(2), abs=1e-6)
def test_info_gain_useless(): assert info_gain(np.array([.5,.5]), np.array([.7,.7])) == pytest.approx(0, abs=1e-9)
def test_update_yes_no_unknown():
    pr, lik = np.array([.5,.5]), np.array([.9,.1])
    assert update(pr, lik, "yes") == pytest.approx([.9,.1]); assert update(pr, lik, "no") == pytest.approx([.1,.9])
    assert update(pr, lik, "unknown") == pytest.approx(pr)
def test_candidate_prior_drops_small_third():  # p=[.5,.45,.03,...] -> 2 candidates, prior [.5/.95,.45/.95]
def test_not_gated_when_low_entropy(): st=start(onehot_ish, H=0.1, tau=0.5); assert next_question(st, L) is None and final(st)[2]
def test_ig_picks_most_separating(): # L rows: s0 useless, s1 splits c0/c1 -> picks 1
def test_stops_at_085():  # one decisive "yes" -> is_done True
def test_all_unknown_terminates_unconfirmed():  # run_dialogue with answer_fn=lambda s:"unknown" -> n<=3, final confirmed False
def test_random_strategy_uses_rng_and_never_repeats():
```
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** IG = `entropy(prior) − Σ_o P(o)·entropy(post_o)` with `P(yes) = prior·lik`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: active Bayesian questioning`

### Task 4: Vision + MC Dropout (`vision.py`)

**Files:** Create `plant_expert/vision.py`. Test: `tests/test_vision.py`

**Interfaces:**
- Produces:
  - `build_model(num_classes: int, p_drop: float = 0.3, pretrained: bool = False) -> nn.Module`: torchvision `convnext_tiny`, with `classifier[2]` replaced by `nn.Sequential(nn.Dropout(p_drop), nn.Linear(768, num_classes))`
  - `enable_mc_dropout(model) -> None`: `model.eval()`, then `.train()` on every `nn.Dropout`
  - `mc_predict(model, x: torch.Tensor, T: int = 20) -> np.ndarray`: shape `[T, N, C]` softmax probs, under `torch.no_grad`
  - `predictive_entropy(p_mean: np.ndarray) -> np.ndarray` (last axis), `mutual_info(probs: np.ndarray) -> np.ndarray` (probs `[T, N, C]`)
  - `preprocess(img: PIL.Image.Image, cfg: dict) -> torch.Tensor` (`[1,3,S,S]`; `.convert("RGB")`, resize to `cfg["img_size"]`, normalize with `cfg["mean"]` / `cfg["std"]`)
  - `load(artifacts_dir: str = "artifacts", device: str | None = None) -> tuple[nn.Module, dict]`: reads `config.json` (`classes`, `tau`, `p_drop`, `img_size`, `mean`, `std`, `T`) and `model.pt` (state_dict)
  - `diagnose(model, cfg, img) -> dict`: keys `p_mean` (`[C]`), `H`, `MI`, `probs` (`[T, C]`)

- [ ] **Step 1: Write failing tests** (use a tiny `nn.Sequential(Flatten, Linear, BatchNorm1d, Dropout(0.5), Linear)` except where noted)
```python
def test_entropy_and_mi_values():
    u = np.ones((1,4))/4; assert predictive_entropy(u)[0] == pytest.approx(np.log(4))
    same = np.tile(np.array([[[.7,.1,.1,.1]]]), (20,1,1)); assert mutual_info(same)[0] == pytest.approx(0, abs=1e-9)
def test_enable_mc_dropout_keeps_bn_eval(): # bn.training False, dropout.training True
def test_mc_predict_shape_and_variance(): # shape (20,2,C), rows sum to 1, std over T > 0
def test_preprocess_rgba_and_grayscale(): # both -> shape (1,3,S,S)
def test_build_model_output_shape(): # build_model(10) on zeros(1,3,64,64) -> (1,10)
def test_load_roundtrip(tmp_path): # save build_model(10) state_dict + config.json, load(), diagnose() keys/shapes
```
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: MC Dropout vision module`

### Task 5: Colab notebook (training + Experiment 1)

**Files:** Create `notebooks/train_uncertainty.ipynb`. Test: `tests/test_notebook.py`

**Interfaces:**
- Consumes: `plant_expert/vision.py` (`build_model`, `enable_mc_dropout`, `mc_predict`, `predictive_entropy`, `mutual_info`). The notebook gets the repo either by zipping it to Drive or with `files.upload()` of `vision.py`, then `sys.path.insert`.
- Produces, in `artifacts/` (downloaded as a zip at the end):
  - `model.pt` and `config.json` (keys as in Task 4; `T` = 20)
  - `test_mc_probs.csv`: columns `image_id, split` (`clean|blur|occlusion|brightness|plantdoc`), `label`, `p_<class>` ×10, `H`, `MI`, for test splits and PlantDoc
  - `exp1_metrics.csv`: rows = method (`softmax`, `temp_scaling`, `mc_dropout`) × split; cols = `acc`, `ece`, `auroc_err`, `aurc`
  - `exp1_reliability.png`, `exp1_risk_coverage.png`, `exp1_entropy_hist.png`

Cells, in order:
1. A `SMOKE = False` flag (True = 20 imgs/class and 1 epoch, for a local check), plus a device print.
2. Data:
   - PlantVillage: sparse git clone of `https://github.com/spMohanty/PlantVillage-Dataset` restricted to `raw/color/Tomato___*`, with folders mapped to the class ids.
   - PlantDoc: `https://github.com/pratikkayal/PlantDoc-Dataset`, tomato folders mapped to the class ids (unmappable dropped).
   - Split 70/15/15 stratified with seed 42.
3. Train: `build_model(10, p_drop=0.3, pretrained=True)`, AdamW, standard augmentations, with the epoch count left to the builder. Keep the best val acc.
4. Noisy test sets: blur, occlusion (RandomErasing) and brightness, 1 level each (spec deadline cut).
5. The three methods:
   - softmax: single pass in eval mode.
   - temp_scaling: fit T by NLL on val, using LBFGS.
   - mc_dropout: T = 20.
   - Uncertainty score = H. For mc_dropout, also report AUROC with MI.
6. Metrics: ECE (15 equal-width bins), AUROC for misclassification (sklearn), and the risk-coverage curve with AURC. Write the CSV and the plots.
7. τ: the smallest H threshold on val such that risk on the accepted set ≤ 5% (if unreachable, the τ at coverage 0.8), with mc_dropout. Write `config.json`.
8. Write `test_mc_probs.csv`, then zip and download `artifacts/`.

- [ ] **Step 1: Write the test.** `test_notebook_cells_parse`: load the file with `json`, then `ast.parse` every code cell after stripping lines that start with `!` or `%`. `test_notebook_writes_expected_artifacts`: the source text contains each artifact filename above.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Write the notebook.**
- [ ] **Step 4: Run the tests.** Expected: PASS. Optionally, smoke run locally with `SMOKE=True` on the RTX 4060 via `jupyter nbconvert --execute` and check that the `artifacts/` files exist.
- [ ] **Step 5: Commit** `feat: Colab training + Experiment 1 notebook`

### Task 6: GraphRAG (`graphrag.py`)

**Files:** Create `plant_expert/graphrag.py`, `scripts/build_index.py`. Test: `tests/test_graphrag.py`

**Interfaces:**
- Consumes: `build_graph`, `load_kg` (Task 1), `kg/manual/*.md` (Task 2)
- Produces:
  - `class OllamaError(RuntimeError)`
  - `ollama_generate(prompt: str, model: str = "qwen2.5:7b-instruct", host: str = "http://localhost:11434") -> str`: POST `/api/generate`, `stream: false`, `options: {"temperature": 0}`, timeout 300 s. Raises `OllamaError` (Vietnamese message telling the user to run `ollama serve` / `ollama pull ...`) on connection or HTTP errors.
  - `ollama_embed(texts: list[str], model: str = "bge-m3", host=...) -> np.ndarray`: POST `/api/embed`, L2-normalized
  - `subgraph(G, disease_id: str, hops: int = 2) -> nx.MultiDiGraph`: undirected-reach ego graph around `disease:<id>`, excluding `context:` nodes
  - `facts(sub, disease_id) -> list[str]`: one Vietnamese sentence per edge. TREATED_BY lines include the dose and PHI, or "theo hướng dẫn trên nhãn sản phẩm" when null, plus the source. RESTRICTED_WITH lines are prefixed "CẢNH BÁO:".
  - `class VectorIndex`: `build(manual_dir: str, embed=ollama_embed) -> VectorIndex` (chunks = paragraphs; each chunk keeps its class id from the filename), `save(path)`, `load(path)`, `search(query: str, k: int = 4, embed=ollama_embed) -> list[str]`
  - `build_prompt(question: str, disease_vi: str, mode: str, facts: list[str], passages: list[str]) -> str`. Modes:
    - `"llm"`: no context.
    - `"naive"`: passages only.
    - `"graph"`: facts + passages, plus the rules: answer only from the context; list ingredient, dose and PHI; a mandatory "Cảnh báo chống chỉ định" section; never name an ingredient that is not in the context.
  - `advise(question, disease_id, kg, G, index, mode="graph", generate=ollama_generate, embed=ollama_embed) -> dict`: keys `answer`, `facts`, `passages`, `context` (the exact text given to the LLM). For `healthy` it returns a fixed Vietnamese "cây khỏe" answer without calling `generate`.
  - `scripts/build_index.py` builds `kg/index.npz`.

- [ ] **Step 1: Write failing tests** (use `tests/fixtures/mini_kg.json`, a fake `embed` = bag-of-chars vector, and a fake `generate` that echoes the prompt)
```python
def test_subgraph_reaches_contraindication(): # contra:c01 in subgraph(G,"Late_blight")
def test_subgraph_excludes_context_nodes():
def test_facts_has_warning_and_label_fallback(): # "CẢNH BÁO:" line; null dose -> "theo hướng dẫn trên nhãn sản phẩm"
def test_prompt_modes():  # llm: no fact text; naive: passages but no facts; graph: facts + "Cảnh báo chống chỉ định"
def test_healthy_skips_llm(): # generate=Mock(); advise(..., "healthy") ; generate.assert_not_called()
def test_ollama_down_raises(): # host="http://127.0.0.1:9" -> OllamaError
def test_index_roundtrip(tmp_path):  # build on tmp manual dir with fake embed, save/load, search returns k strings
```
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS. Then, if Ollama is installed: `ollama pull qwen2.5:7b-instruct`, `ollama pull bge-m3`, `.venv\Scripts\python scripts/build_index.py` creates `kg/index.npz`.
- [ ] **Step 5: Commit** `feat: agricultural GraphRAG`

### Task 7: Experiment 2 (active questioning)

**Files:** Create `experiments/simulate_answers.py`, `experiments/exp2_active.py`. Test: `tests/test_exp2.py`

**Interfaces:**
- Consumes: `run_dialogue` (Task 3), `likelihood_matrix`/`load_kg` (Task 1), `artifacts/test_mc_probs.csv` and `config.json["tau"]` (Task 5)
- Produces:
  - `simulate(labels: dict[str, str], kg: dict, perturb: float = 0.15, noise: float = 0.1, seed: int = 0) -> pd.DataFrame`, with columns `image_id, symptom_id, answer`:
    - The "true" table is `clip(L + U(−perturb, perturb), 0.02, 0.98)`, drawn once per call.
    - With prob `noise` the answer is corrupted: half the time flipped, half the time set to "unknown".
  - CLI: `python experiments/simulate_answers.py --probs artifacts/test_mc_probs.csv --out experiments/answers_sim.csv [--perturb --noise --seed]`
  - `evaluate(probs_df, answers_df, L, sym_ids, classes, tau, strategy: str) -> dict` (`acc`, `macro_f1`, `avg_q`, `n`), computed on hard rows only (H > τ):
    - `strategy ∈ {"vision", "random", "ig"}`.
    - "random" uses rng seed 0.
    - A missing (image, symptom) answer counts as "unknown".
  - CLI: `python experiments/exp2_active.py --answers experiments/answers_sim.csv [--sweep]`, which writes `experiments/out/exp2_results.csv`. `--sweep` also writes `exp2_sweep.csv` and `exp2_sweep.png` (acc vs noise ∈ {0, .1, .2, .3} and perturb ∈ {0, .15, .3}). Results files from simulated answers have `simulated` in the filename.

- [ ] **Step 1: Write failing tests** (on a synthetic probs_df built in the test: 3 classes with ambiguous p over classes 0/1 and H > τ)
```python
def test_simulate_deterministic_when_no_noise(): # p in {0.02,0.98}, perturb=0, noise=0 -> answers match the majority
def test_simulate_noise_rate():  # noise=0.2 on 5000 draws -> corrupted fraction within 0.2±0.03
def test_ig_beats_vision_on_synthetic(): # evaluate(... "ig")["acc"] > evaluate(... "vision")["acc"]
def test_true_class_outside_candidates_counts_wrong(): # label=class 2 with p tiny -> acc 0, no exception
def test_missing_answers_are_unknown(): # empty answers_df -> ig acc == vision acc, avg_q <= 3
```
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: Experiment 2 active questioning`

### Task 8: Experiment 3 (GraphRAG quality)

**Files:** Create `experiments/exp3_questions.json`, `experiments/exp3_rag.py`. Test: `tests/test_exp3.py`

**Interfaces:**
- Consumes: `advise`, `VectorIndex`, `build_graph`, `load_kg`
- Produces:
  - `exp3_questions.json`: `{"known_ingredients": [...about 40 common pesticide names not limited to the KG...], "questions": [{"id", "disease", "question", "must_contra_keywords": [...], "trap": bool}]}`. About 30 questions, ≥ 8 of them traps (e.g. a fungicide for a virus or bacteria).
  - `mentioned_ingredients(text: str, vocab: list[str]) -> set[str]`: casefolded substring match
  - `score(answer: str, q: dict, kg: dict, context: str, vocab: list[str]) -> dict`:
    - `halluc`: any mentioned ingredient that is not one of the disease's treatment ingredients
    - `contra_hit`: every `must_contra_keywords` item appears, casefolded
    - `faithfulness`: mentioned ingredients that appear in `context`, divided by all mentioned ingredients (1.0 if none are mentioned)
  - CLI: `python experiments/exp3_rag.py`, which runs all 3 modes and writes `experiments/out/exp3_answers.jsonl` and `exp3_summary.csv` (mode × halluc rate, contra hit rate, mean faithfulness).

- [ ] **Step 1: Write failing tests**
```python
def test_mentioned_ingredients_casefold(): assert mentioned_ingredients("Phun MANCOZEB 80WP", ["mancozeb","metalaxyl"]) == {"mancozeb"}
def test_score_halluc(): # answer names an ingredient not in the disease treatments -> halluc True
def test_score_contra_hit(): # all keywords present -> True; one missing -> False
def test_faithfulness_no_mentions_is_one():
def test_questions_file_valid(): # >=30 questions, >=8 traps, every disease id in CLASSES
```
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement and write the question set.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: Experiment 3 GraphRAG evaluation`

### Task 9: Streamlit demo (`app.py`)

**Files:** Create `app.py`, `README.md` (setup: venv, Ollama pulls, put the Colab `artifacts/` in place, `build_index.py`, `streamlit run app.py`, how to run the experiments). Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `vision.load` / `vision.diagnose`; `bayes.start` / `next_question` / `answer` / `is_done` / `final`; `kg.load_kg` / `build_graph` / `likelihood_matrix`; `graphrag.VectorIndex.load` / `advise` / `subgraph` / `OllamaError`

Behaviour:
- Load the model, KG, graph and index once with `st.cache_resource`. If `artifacts/` is missing, show `st.error` with instructions and stop.
- **Left column** (`st.columns([1,1])`):
  - Uploader, then the chat (`st.chat_message`).
  - The current question with three buttons: Có / Không / Không rõ. Show the "ước lượng chuyên gia" caption.
  - Once done: the diagnosis, then a `st.chat_input` for the farmer's question (default question "Cách điều trị bệnh này?"), then `advise(mode="graph")`. On `OllamaError`, show `st.warning` and still show the facts.
- **Right column:**
  - Plotly gauge of H with a τ marker.
  - Grouped bar chart of candidate probabilities, before and after each answer, from `State.history`.
  - pyvis HTML of `subgraph`, embedded via `st.components.v1.html`. Disease→ingredient→contraindication paths highlighted red.
- Keep state in `st.session_state` (`state`, `diag`, `messages`), and reset it when a new file is uploaded.

- [ ] **Step 1: Write the failing test.** `test_app_without_artifacts_shows_error`: monkeypatch the artifacts dir to an empty `tmp_path` via the env var `PLANT_ARTIFACTS`, which `app.py` reads with default `"artifacts"`. `AppTest.from_file("app.py").run()`, then `len(at.error) == 1` and `not at.exception`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement `app.py` and `README.md`.**
- [ ] **Step 4: Run the test.** Expected: PASS. Manual check, if artifacts exist: `streamlit run app.py`, then upload a leaf image, answer the questions and get the advice.
- [ ] **Step 5: Run the full suite.** `pytest -v`: all PASS.
- [ ] **Step 6: Commit** `feat: Streamlit demo`
