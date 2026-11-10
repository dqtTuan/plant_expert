# Nguồn tham khảo Đồ thị Tri thức Nông nghiệp (`kg/kg.json`)

1. **Danh mục thuốc bảo vệ thực vật được phép sử dụng tại Việt Nam** — Bộ Nông nghiệp và Phát triển nông thôn (nay là Bộ Nông nghiệp và Môi trường), Cục Bảo vệ thực vật (`ppd.gov.vn`). Dùng để đối chiếu các hoạt chất được phép đăng ký trên rau/cà chua: Mancozeb, Metalaxyl, Cymoxanil, Chlorothalonil, Azoxystrobin, Difenoconazole, Copper hydroxide, Kasugamycin, Abamectin, Pyridaben, Imidacloprid, Pymetrozine.
2. **Tài liệu kỹ thuật Quản lý dịch hại tổng hợp (IPM) trên cây cà chua** — Viện Khoa học Nông nghiệp Việt Nam (VAAS) & Trung tâm Khuyến nông Quốc gia (`khuyennongvn.gov.vn`).
3. **Hướng dẫn sử dụng thuốc BVTV an toàn, hiệu quả (nguyên tắc 4 đúng) và quản lý tính kháng thuốc (FRAC/IRAC)** — Cục Bảo vệ thực vật.

## Ghi chú học thuật quan trọng

- **Liều lượng (`dose`) và Thời gian cách ly (`phi_days`):** Tại Việt Nam, liều pha cụ thể được quy định theo từng tên thương phẩm và hàm lượng công thức (WP, SC, WG, EC...). Để tránh bịa số liệu trái với phản mục tiêu của đề tài, trường `dose` trong `kg.json` được để `null` và hệ thống tự động khuyến cáo *"pha theo hướng dẫn trên nhãn sản phẩm thương mại được đăng ký hợp pháp"*, kèm thời gian cách ly tối thiểu tham khảo 7 ngày trên cà chua.
- **Ma trận xác suất có điều kiện `P(s | c)` của 16 triệu chứng phụ (`context_symptoms`):** Các giá trị xác suất này là **ước lượng chuyên gia (expert-elicited priors/likelihoods)** được lượng hóa từ mô tả định tính trong tài liệu bệnh cây ("thường gặp / điển hình" ≈ 0.85–0.92, "đôi khi gặp" ≈ 0.30–0.50, "hiếm gặp" ≈ 0.02–0.15), **không phải** tần suất thống kê đo đạc từ khảo sát đồng ruộng.
