# Hệ thống Chẩn đoán Bệnh Cà chua Đa phương thức (MC Dropout + Active Bayes + GraphRAG)

Đồ án môn **Học máy Nâng cao**: kết hợp ước lượng độ bất định nhận thức bằng **Monte Carlo Dropout**, hỏi đáp triệu chứng phụ theo **Information Gain & cập nhật hậu nghiệm Bayes**, và tư vấn phác đồ điều trị qua **Đồ thị Tri thức Nông nghiệp (Agricultural GraphRAG)**.

## 1. Cài đặt môi trường trên máy cục bộ

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Cài đặt **Ollama** (`https://ollama.com`) để chạy mô hình ngôn ngữ lớn và mô hình embedding hoàn toàn offline trên GPU RTX 4060:

```powershell
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
.venv\Scripts\python scripts/build_index.py
```

*(Nếu chưa bật Ollama, `scripts/build_index.py` sẽ tự tạo chỉ mục dự phòng để kiểm thử nhanh).*

## 2. Huấn luyện mô hình & chạy Thí nghiệm 1 (Google Colab hoặc máy cục bộ)

1. Mở `notebooks/train_uncertainty.ipynb` trên Google Colab (chọn GPU T4) hoặc chạy trực tiếp trên máy.
2. Notebook tự động tải PlantVillage (10 lớp cà chua) + PlantDoc, huấn luyện `ConvNeXt-Tiny` có lớp Dropout, chạy Thí nghiệm 1 (`Softmax`, `Temperature Scaling`, `MC Dropout` với `T=20`) và nén toàn bộ kết quả thành `artifacts.zip`.
3. Giải nén `artifacts.zip` vào thư mục `artifacts/` ở gốc dự án (gồm `model.pt`, `config.json`, `test_mc_probs.csv`, `exp1_metrics.csv`, và 3 biểu đồ `.png`).

## 3. Chạy Thí nghiệm 2 (Active Bayesian Questioning)

```powershell
# Tạo câu trả lời giả lập từ bảng P(s|c) có nhiễu (dùng khi chưa có câu trả lời khảo sát thật)
.venv\Scripts\python experiments/simulate_answers.py --probs artifacts/test_mc_probs.csv --out experiments/answers_sim.csv

# Đánh giá Vision-only vs Random vs Active IG (kèm biểu đồ độ nhạy theo mức nhiễu)
.venv\Scripts\python experiments/exp2_active.py --answers experiments/answers_sim.csv --sweep
```

Khi có file khảo sát thật cùng định dạng CSV (`image_id,symptom_id,answer`), chỉ cần truyền `--answers path/to/real.csv`.

## 4. Chạy Thí nghiệm 3 (Agricultural GraphRAG vs Naive RAG vs LLM thuần)

Đảm bảo `ollama serve` đang chạy, sau đó:

```powershell
.venv\Scripts\python experiments/exp3_rag.py
```

Kết quả được lưu tại `experiments/out/exp3_summary.csv` và `experiments/out/exp3_answers.jsonl`.

## 5. Chạy giao diện Web Demo 2 khung (Streamlit)

```powershell
.venv\Scripts\streamlit run app.py
```

## 6. Chạy bộ kiểm thử tự động (Pytest)

```powershell
.venv\Scripts\python -m pytest -v
```
