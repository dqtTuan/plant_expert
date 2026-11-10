# Thiết kế: Hệ thống chẩn đoán bệnh cà chua có ước lượng bất định + hỏi đáp Bayes + GraphRAG

Nguồn yêu cầu gốc: tài liệu đặc tả "Uncertainty-Aware Multimodal Plant Disease Diagnosis and Knowledge Graph-Enhanced Advisory System" (5 trang, học phần Học máy Nâng cao). Tài liệu này ghi lại những gì đã chốt khi trao đổi và những chỗ lệch so với đặc tả gốc.

## 1. Mục đích và lý do

Đồ án môn Học máy Nâng cao, để chấm điểm và demo trước giảng viên. Cần chứng minh 3 đóng góp bằng thực nghiệm định lượng:
1. MC Dropout cho độ bất định tốt hơn Softmax thường hoặc Temperature Scaling.
2. Hỏi thêm triệu chứng phụ theo Bayes giúp tăng độ chính xác trên ảnh khó.
3. GraphRAG giảm ảo giác và bắt đúng chống chỉ định so với LLM thuần và Naive RAG.

Sản phẩm gồm: code, 3 thí nghiệm (bảng và biểu đồ dùng cho Chương 4 của báo cáo) và web demo hai khung.

## 2. Mục tiêu, ngoài phạm vi, phản mục tiêu

**Mục tiêu**
- Pipeline 4 phân hệ chạy từ đầu đến cuối trên một ảnh.
- 3 thí nghiệm tái lập được, đầu ra là file CSV/PNG.
- Demo chạy offline hoàn toàn trên laptop của người dùng.

**Ngoài phạm vi**
- Cây lúa, các bệnh ngoài 10 lớp cà chua của PlantVillage.
- Đăng nhập, nhiều người dùng, triển khai lên server, chạy trên thiết bị di động.
- Neo4j và mọi cơ sở dữ liệu ngoài.
- Viết cuốn báo cáo. Code chỉ sinh số liệu và hình cho báo cáo.

**Phản mục tiêu** (kể cả khi chạy được vẫn coi là hỏng)
- LLM nêu hoạt chất hoặc liều lượng không có trong đồ thị.
- Tư vấn thuốc trừ nấm cho bệnh do vi khuẩn hay virus.
- Trình bày P(s|c) như số liệu đo đạc. Phải ghi rõ đó là ước lượng chuyên gia.
- Demo cần Internet hoặc API trả phí.

## 3. Ràng buộc

- **Train:** Google Colab (GPU T4). Notebook phải chạy được cả trên máy cục bộ.
- **Máy chạy demo:** Windows, RTX 4060 Laptop 8 GB VRAM, RAM 16 GB. Python trên máy hiện chỉ là stub của Windows Store, nên phải cài Python thật (khuyên dùng 3.11 hoặc 3.12) và tạo venv.
- **LLM offline:** Ollama + `qwen2.5:7b-instruct` (Q4). Embedding: `bge-m3` qua Ollama. Cả hai phải vừa VRAM khi chạy cùng ConvNeXt-Tiny.
- **Dữ liệu ảnh:** PlantVillage, 10 lớp cà chua: Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot, Spider_mites, Target_Spot, Tomato_Yellow_Leaf_Curl_Virus, Tomato_mosaic_virus, healthy. PlantDoc (các lớp cà chua ánh xạ được sang 10 lớp trên) làm tập thực địa.
- **Ngôn ngữ giao diện và tư vấn:** tiếng Việt.
- **Thời hạn:** càng sớm càng tốt (người dùng: "hoàn thiện bây giờ"). Ưu tiên bản tối thiểu chạy được từ đầu đến cuối. Thứ tự cắt khi thiếu thời gian: lưới nhiễu chỉ 1 mức cho mỗi loại; Thí nghiệm 3 khoảng 30 câu; pyvis giữ lại vì đặc tả yêu cầu.

## 4. Các hướng đã cân nhắc

**Web demo**
| Hướng | Đánh đổi | Kết quả |
|---|---|---|
| A. Streamlit | Thuần Python, có sẵn chat, cột và upload. Khó tùy biến giao diện, trang chạy lại mỗi lần tương tác. | **Người dùng chọn** vì ít code nhất mà vẫn đủ hai khung. |
| B. Gradio | Thuần Python, chat sẵn. Bảng điều khiển phức tạp và trạng thái nhiều lượt khó làm. | Loại |
| C. FastAPI + HTML/JS | Giao diện đẹp nhất, nhưng code gấp 2–3 lần và không thêm điểm ML. | Loại |

**Lưu đồ thị:** NetworkX + file JSON (đề xuất của mình). Neo4j chỉ thêm việc cài đặt mà không thêm điểm.

**LLM:** Qwen2.5-7B qua Ollama (người dùng chọn chạy offline, model do mình chọn). Đã loại Vistral-7B vì bám ngữ cảnh kém hơn, và Gemma-2-9B vì sát giới hạn VRAM.

**Backbone:** ConvNeXt-Tiny pretrained ImageNet, thêm `Dropout(p=0.3)` trước lớp phân loại (mình chọn). Swin nặng hơn mà không thêm giá trị cho báo cáo.

## 5. Các quyết định đã chốt

### Phân hệ 1–2: Vision và Uncertainty (Colab notebook)
- Train trên PlantVillage cà chua, chia train/val/test theo tỉ lệ cố định với seed cố định.
- **Tập test nhiễu:** áp blur, che một phần lá (random erasing) và đổi độ sáng lên tập test, nhiều mức độ.
- **MC Dropout:** BatchNorm/LayerNorm để ở chế độ eval, chỉ bật Dropout, chạy T = 20. Tính p̂, entropy H(p̂) và MI theo mục 3.1 của đặc tả.
- **Thí nghiệm 1:** so sánh Softmax, Temperature Scaling (fit T trên tập val) và MC Dropout. Độ đo: ECE (15 bin), AUROC phát hiện mẫu đoán sai (điểm bất định là H), đường Risk-Coverage. Chạy trên test sạch, test nhiễu và PlantDoc.
- **Chọn τ** trên tập val từ đường Risk-Coverage (mình đề xuất: τ nhỏ nhất sao cho risk trên phần được chấp nhận ≤ 5%; người xây dựng được đổi tiêu chí nhưng phải ghi lại).
- **Đầu ra:** `model.pt` (state_dict), `config.json` (tên lớp theo đúng thứ tự, τ, p_drop, kích thước ảnh, mean/std), cùng các hình và CSV của Thí nghiệm 1.

### Phân hệ 3: Active Bayesian Questioning (máy cục bộ)
- Nếu H(p̂) ≤ τ thì chốt luôn bệnh argmax.
- Nếu H > τ: lấy top-k bệnh (k = 3, hoặc 2 nếu bệnh thứ 3 có xác suất rất nhỏ), chuẩn hóa lại thành prior.
- **Chọn câu hỏi theo Information Gain** (lệch so với công thức hiệu tuyệt đối trong đặc tả, vì IG dùng được cho k > 2): chọn s* chưa hỏi có giá trị lớn nhất của H(prior) − E_o[H(posterior | s = o)].
- Cập nhật Bayes đúng công thức mục 3.2. Câu trả lời **"Không rõ"** thì không cập nhật và chuyển sang câu hỏi tiếp theo.
- Dừng khi một bệnh có posterior ≥ 0.85, hoặc đã hỏi 3 câu, hoặc hết câu hỏi có IG > 0.
- Phải ghi rõ trong báo cáo: giả định các triệu chứng độc lập có điều kiện theo bệnh (naive Bayes).

### Phân hệ 4: Agricultural GraphRAG
- **Ontology:** 6 loại node và 5 loại cạnh (đặc tả ghi "5 loại node" nhưng thực tế liệt kê 6): Disease, Pathogen, VisualSymptom, ContextSymptom, ActiveIngredient, Contraindication. Các cạnh: CAUSED_BY, HAS_VISUAL_SIGN, HAS_CONTEXT_SIGN (thuộc tính `p` = P(s|c)), TREATED_BY, RESTRICTED_WITH. Mọi node ActiveIngredient và Contraindication có trường `source` (trích nguồn).
- **Nội dung do mình soạn từ nguồn công khai:** Danh mục thuốc BVTV được phép sử dụng tại Việt Nam (Thông tư hiện hành của Bộ NN&PTNT), tài liệu khuyến nông công khai, nhãn thuốc. Gồm 15–20 ContextSymptom, mỗi cái có câu hỏi tiếng Việt và giá trị P(s|c) cho cả 10 lớp.
- **Liều lượng và thời gian cách ly** *(quyết định của mình)*: ở Việt Nam, liều và thời gian cách ly ghi theo từng sản phẩm thương mại, không theo hoạt chất. Vì vậy lưu một giá trị tham khảo cho mỗi cặp (hoạt chất, bệnh) làm thuộc tính của cạnh TREATED_BY (`dose`, `phi_days`, `source`). Không tìm được nguồn thì để trống, khi đó câu trả lời ghi "theo hướng dẫn trên nhãn sản phẩm". Cấm bịa số liệu.
- **Bệnh virus** (TYLCV, mosaic): TREATED_BY chỉ trỏ tới biện pháp hoặc thuốc trừ côn trùng môi giới, kèm chống chỉ định "không dùng thuốc trừ nấm hay kháng sinh để trị virus". **Lớp healthy:** không có phác đồ.
- **Truy xuất lai:** lấy đồ thị con trong bán kính 2 bước quanh bệnh đã chốt, tuần tự hóa thành các bộ ba văn bản. Kết hợp với top-k đoạn văn từ vector search trên cẩm nang khuyến nông (mình soạn, mỗi bệnh vài đoạn).
- **Prompt LLM:** chỉ được dùng ngữ cảnh đã cho. Liệt kê hoạt chất, liều, thời gian cách ly và bắt buộc có mục "Cảnh báo chống chỉ định".
- **Index vector:** dùng numpy cosine trên embedding lưu ra file. Không cần vector DB.

### Thí nghiệm 2
- Tập "ảnh khó": các ảnh test (gồm nhiễu và PlantDoc) có H > τ.
- So sánh: chỉ ảnh, ảnh + hỏi triệu chứng ngẫu nhiên (cùng số câu), và ảnh + IG. Độ đo: Top-1 accuracy, Macro-F1, số câu hỏi trung bình.
- **Câu trả lời đọc từ CSV** (`image_id, symptom_id, answer` với answer ∈ {yes, no, unknown}). Người dùng sẽ thu thập câu trả lời thật sau. Trong lúc chờ, có một script sinh file giả lập bằng cách lấy mẫu từ P(s | nhãn thật), có tham số tỉ lệ nhiễu (lật hoặc thành unknown). Đổi file thì không phải sửa code. Báo cáo phải ghi rõ kết quả nào dùng dữ liệu giả lập.
- **Tránh vòng lặp tự chứng minh** *(quyết định của mình)*: nếu bộ giả lập lấy mẫu từ đúng bảng P(s|c) mà bộ Bayes dùng, thí nghiệm chỉ kiểm tra mô hình trên chính giả định của nó. Vì vậy bộ giả lập dùng một bảng P(s|c) "thật" riêng, nhiễu hóa từ bảng trong `kg.json` (tham số `perturb`, ví dụ ±0.15 rồi cắt về [0.02, 0.98]), cộng thêm tỉ lệ trả lời sai. Báo cáo kèm biểu đồ accuracy theo `perturb` và theo tỉ lệ nhiễu. Câu trả lời thật cho triệu chứng ngoài lá (thời tiết, côn trùng, cắt thân) không thể suy ra từ ảnh, nên nhiều khả năng **kết quả giả lập sẽ là kết quả chính**, còn dữ liệu thật (nếu có) là phần bổ sung.

### Thí nghiệm 3
- Bộ khoảng 30–50 câu hỏi tư vấn có đáp án, do mình soạn. Mỗi câu kèm danh sách hoạt chất đúng và chống chỉ định phải nêu. Có cả câu bẫy, ví dụ hỏi thuốc trừ nấm cho bệnh do vi khuẩn.
- So sánh LLM thuần, Naive RAG (chỉ vector) và GraphRAG, cùng một model Qwen2.5-7B.
- **Bệnh được cho sẵn trong câu hỏi** *(quyết định của mình)*, không lấy từ pipeline ảnh. Như vậy chỉ so sánh truy xuất và sinh câu trả lời, lỗi của mô hình ảnh không lẫn vào.
- **Độ đo tự động:**
  - Tỉ lệ ảo giác = câu trả lời có nhắc hoạt chất không thuộc danh sách hợp lệ của bệnh đó (so khớp chuỗi với danh mục hoạt chất trong đồ thị cộng danh sách hoạt chất phổ biến).
  - Tỉ lệ phát hiện chống chỉ định = câu trả lời có nêu chống chỉ định bắt buộc (so khớp từ khóa).
  - Faithfulness = tỉ lệ hoạt chất được nhắc mà có trong ngữ cảnh truy xuất.
- Không dùng LLM-làm-giám-khảo, vì chỉ có model offline 7B.

### Web demo (Streamlit)
- **Cột trái:** upload ảnh; khung chat hiện kết quả, câu hỏi kèm 3 nút Có / Không / Không rõ, rồi phác đồ do LLM sinh ra.
- **Cột phải:** đồng hồ H(p̂) so với τ (Plotly gauge); biểu đồ cột phân phối xác suất top bệnh trước và sau mỗi lượt hỏi; đồ thị con dạng pyvis, tô sáng đường Bệnh → Hoạt chất → Chống chỉ định.
- Trạng thái hội thoại giữ trong `st.session_state`. Model ảnh nạp một lần qua `st.cache_resource`.

### Khi kết quả không như kỳ vọng *(quyết định của mình)*
Báo cáo trung thực, không chỉnh để ra đúng các con số kỳ vọng trong đặc tả (76% → 92%, 100%). MC Dropout thường không thắng Temperature Scaling về ECE, nên Thí nghiệm 1 báo cáo cả H và MI làm điểm bất định. Chỉ chỉnh lại một vòng (p_drop, T) trước khi chốt.

## 6. Cấu trúc thư mục (đề xuất)

```
notebooks/train_uncertainty.ipynb   # Phân hệ 1–2 + Thí nghiệm 1 (Colab)
artifacts/                          # model.pt, config.json (tải từ Colab về)
kg/kg.json, kg/manual/*.md          # đồ thị + cẩm nang khuyến nông
plant_expert/vision.py              # nạp model, MC Dropout, H, MI
plant_expert/bayes.py               # chọn câu hỏi IG, cập nhật Bayes
plant_expert/graphrag.py            # đồ thị con, vector search, prompt, gọi Ollama
experiments/exp2_active.py, exp3_rag.py, simulate_answers.py, exp3_questions.json
app.py                              # Streamlit
tests/                              # test cho bayes.py và việc nạp/kiểm tra kg.json
```

## 7. Để người xây dựng tự quyết

- Siêu tham số train (epoch, lr, augmentation), kích thước ảnh, p_drop trong khoảng 0.2–0.5.
- Cách ánh xạ cụ thể từ lớp PlantDoc sang lớp PlantVillage. Lớp nào không ánh xạ được thì bỏ.
- Cách chia đoạn văn cẩm nang, giá trị top-k khi truy xuất, chi tiết câu chữ trong prompt.
- Bố cục chi tiết và màu sắc của demo.
- Kiểm tra `kg.json` khi nạp: mỗi bệnh phải có đủ P(s|c) với mọi ContextSymptom, và mọi hoạt chất phải có `source`.
