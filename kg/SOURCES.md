# Nguồn tham khảo Đồ thị Tri thức Nông nghiệp (`kg/kg.json`)

## 1. Văn bản pháp quy & Danh mục thuốc BVTV
1. **Thông tư số 25/2024/TT-BNNPTNT** (ban hành ngày 16/12/2024, hiệu lực từ 30/01/2025, thay thế **Thông tư số 09/2023/TT-BNNPTNT**) của Bộ Nông nghiệp và Phát triển nông thôn về *Ban hành Danh mục thuốc bảo vệ thực vật được phép sử dụng tại Việt Nam và Danh mục thuốc bảo vệ thực vật cấm sử dụng tại Việt Nam* (`ppd.gov.vn`).
   - Phụ lục I dùng để đối chiếu 12 hoạt chất hợp pháp trên rau màu/cà chua: *Mancozeb, Metalaxyl, Cymoxanil, Chlorothalonil, Azoxystrobin, Difenoconazole, Copper hydroxide, Kasugamycin, Abamectin, Pyridaben, Imidacloprid, Pymetrozine*.

## 2. Tiêu chuẩn Quốc gia & Tài liệu kỹ thuật chuyên ngành
2. **TCVN 9962:2013** — *Hạt giống rau họ cà – Yêu cầu kỹ thuật* (Viện Nghiên cứu Rau quả – FAVRI biên soạn).
3. **TCVN 12561:2018** — *Thuốc bảo vệ thực vật – Khảo nghiệm hiệu lực sinh học của thuốc trên đồng ruộng* (Cục Bảo vệ thực vật biên soạn).
4. **Quy trình kỹ thuật Quản lý dịch hại tổng hợp (IPM) trên cây cà chua** — Viện Khoa học Nông nghiệp Việt Nam (VAAS), Viện Nghiên cứu Rau quả (FAVRI) & Trung tâm Khuyến nông Quốc gia (`khuyennongvn.gov.vn`).
5. **Hướng dẫn sử dụng thuốc BVTV an toàn theo nguyên tắc 4 đúng và quản lý tính kháng thuốc (FRAC/IRAC)** — Cục Bảo vệ thực vật (`ppd.gov.vn`).

## 3. Ghi chú học thuật quan trọng

- **Liều lượng tham khảo (`dose`) và Thời gian cách ly (`phi_days`):** Tại Việt Nam, liều pha cụ thể phụ thuộc vào từng tên thương phẩm và hàm lượng dạng công thức (`WP, SC, WG, EC, SL`). Vì vậy, trường `dose` trong `kg.json` ghi rõ mức liều tham khảo theo dạng công thức phổ biến kèm nhắc nhở đối chiếu đúng nhãn sản phẩm thương mại được đăng ký hợp pháp, cùng thời gian cách ly tham khảo 7 ngày trên cà chua.
- **Ma trận xác suất có điều kiện `P(s | c)` của 16 triệu chứng phụ (`context_symptoms`):** Các giá trị xác suất này là **ước lượng chuyên gia (expert-elicited priors/likelihoods)** được lượng hóa từ mô tả định tính trong tài liệu bệnh cây ("thường gặp / điển hình" ≈ 0.85–0.92, "đôi khi gặp" ≈ 0.30–0.50, "hiếm gặp" ≈ 0.02–0.15), **không phải** tần suất thống kê đo đạc từ khảo sát đồng ruộng.
