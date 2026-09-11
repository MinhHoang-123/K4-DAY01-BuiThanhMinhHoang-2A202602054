# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**11/09/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:**Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không thay đổi nội dung xử lý chính; sử dụng các output do notebook tạo.

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):`class_id=468`, `class_name=cab`, `rank=1`, `score=0.510915`, `taxonomy_name=ImageNet-1K`.
- Record này mô tả toàn ảnh như thế nào? Model dự đoán nội dung tổng thể của ảnh `traffic` phù hợp nhất với lớp `cab`. Đây là prediction ở cấp độ toàn ảnh, không cho biết vị trí cụ thể của từng vật thể trong ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?Class list được định nghĩa bởi taxonomy/dataset dùng để huấn luyện checkpoint. Với `yolo11n-cls.pt`, taxonomy trong evidence là `ImageNet-1K`, nên model dự đoán các class thuộc danh sách của taxonomy này.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?Giữ cả 3 trường giúp đảm bảo tính minh bạch và nhất quán: `class_id` (định danh số duy nhất), `class_name` (tên dễ đọc) và `taxonomy_name` (ngữ cảnh/nguồn gốc của class list) để tránh nhầm lẫn và dễ dàng truy vết, đối chiếu với dữ liệu huấn luyện.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Với task phân loại cấp ảnh (image-level classification), guideline cần quy định nếu ảnh có nhiều chủ thể thì dự đoán sẽ ưu tiên: (1) chủ thể nổi bật nhất, chiếm diện tích lớn nhất, hoặc (2) chủ thể được coi là “chính” theo miền ứng dụng (ví dụ: xe cộ trong ảnh giao thông). Trong trường hợp không rõ ràng, có thể cần tách thành nhiều ảnh hoặc dùng detection/segmentation thay vì classification.
- Vì sao model score không phải ground truth?Score là xác suất/độ tin cậy nội tại của model cho một prediction dựa trên dữ liệu đã thấy, không phản ánh thực tế khách quan. Ground truth là nhãn đúng được con người xác nhận. Score chỉ là ước lượng, có thể sai do nhiễu ảnh, bias mô hình hoặc sự mơ hồ của dữ liệu.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):`class_name=person`, `score=0.912625`, `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`, `bbox_width=113.58`, `bbox_height=279.68`.
- Diễn giải vị trí box bằng lời:Box bao quanh một người trong ảnh, với các tọa độ góc trên bên trái là (385.33, 69.24) và góc dưới bên phải là (498.92, 348.92) theo hệ tọa độ của ảnh. Người này có chiều rộng khoảng 113.58 đơn vị và chiều cao khoảng 279.68 đơn vị.
- So sánh số prediction ở hai threshold:`Thresh=0.25` có 14 predictions; `Thresh=0.75` chỉ còn 2 predictions (ít hơn đáng kể), chủ yếu là các object có độ tin cậy cao (person, bottle). Điều này cho phép giảm khối lượng đánh giá nhưng có thể bỏ sót các object ít nổi bật hoặc bị che khuất.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?Khi giảm threshold (ví dụ từ 0.75 xuống 0.25), số lượng bounding box tăng lên, dẫn đến độ bao phủ tăng (nhiều object được phát hiện hơn) nhưng khối lượng reviewer cần xem cũng tăng đáng kể. Reviewer phải kiểm tra nhiều box hơn, bao gồm cả các box có độ tin cậy thấp hơn.
- Đề xuất một quy tắc box chặt:Một box được coi là “chặt” nếu nó bao sát chủ thể mà không chứa quá nhiều vùng nền (background) hoặc các object không liên quan. Quy tắc đề xuất:Intersection over Union (IoU) tối đa với box của chủ thể là 0.1, hoặc bao phủ ít nhất 80% diện tích hiển thị của chủ thể.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?Với object bị che khuất hoặc cắt mép, guideline cần quy định: (1) Nếu phần hiển thị của object vẫn đủ để nhận dạng rõ ràng, annotator có thể tạo box/mask dựa trên phần hiện có; (2) Nếu object bị che khuất hơn 50% hoặc không thể nhận dạng, annotator nên đánh dấu là “ignored” hoặc “partially visible” và có thể cần escalation để quyết định giữ nguyên hay loại bỏ.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): `instance_id=2`, `class_name=person`, `score=0.976875`, `num_points=13`, `polygon_xy=[313, 371, 336, 367, ..., 423, 224, 405, 131]`
- Polygon bổ sung chi tiết gì so với box?Polygon cung cấp hình dạng chi tiết của vật thể, bao gồm cả các đường cong và góc không đều, giúp phân biệt rõ ràng với các vật thể xung quanh và loại bỏ phần nền (background) không cần thiết. Điều này hữu ích cho các tác vụ yêu cầu độ chính xác cao về vị trí và hình dạng. 
- `instance_id` dùng để làm gì và không phải loại ID nào?
- `instance_id` là định danh duy nhất cho từng vật thể cụ thể được phát hiện trong ảnh (ví dụ: person 1, person 2), giúp phân biệt các instance khác nhau của cùng một lớp. Nó khác với `class_id` (định danh loại vật thể) và không phải là ID của ảnh hay ID của model.
- Đề xuất một quy tắc biên mask:
- Quy tắc đề xuất cho biên mask: (1) Bắt buộc bao phủ toàn bộ vật thể có thể nhìn thấy mà không cắt vào vật thể khác; (2) Không chứa vùng nền quá 10% diện tích mask; (3) Tại các vùng chồng lấp với vật thể khác, ưu tiên giữ ranh giới rõ ràng và đánh dấu rõ ràng khu vực chồng lấp.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?Với vùng mờ, tiếp xúc hoặc bị che khuất, guideline cần quy định: (1) Nếu vật thể có thể nhận dạng được ít nhất 50% diện tích, tạo mask dựa trên phần hiện có và đánh dấu “partially visible”; (2) Nếu vật thể bị che khuất hơn 50% hoặc không thể nhận dạng, đánh dấu là “ignored” và có thể cần escalation để quyết định giữ nguyên hay loại bỏ. Nếu vùng tiếp xúc rõ ràng, tạo mask riêng biệt cho từng vật thể.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Class label ở cấp ảnh, kèm class ID theo taxonomy | Ảnh có nhiều chủ thể hoặc khó xác định nội dung chính | Đọc guideline và chọn class phù hợp với nội dung ảnh | Kiểm tra class, ID, taxonomy và tính nhất quán với guideline |
| Phát hiện vật thể |Class + bounding box `xyxy` | Box quá rộng/hẹp, object bị che khuất hoặc cắt mép, false positive/false negative | Vẽ box sát phần object nhìn thấy và gán đúng class | Kiểm tra class, vị trí box, độ bao phủ và các trường hợp khó |
| Instance segmentation |Class + polygon/mask cho từng instance | Biên mask mờ, object tiếp xúc/chồng lấn hoặc bị che khuất | Vẽ polygon bám sát biên object và tạo instance riêng cho từng object | Kiểm tra class, instance, biên polygon/mask và tính nhất quán giữa các instance |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:Chỉ sử dụng và lưu trữ dữ liệu cần thiết cho bài thực hành; không đưa họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân/nhạy cảm vào repository.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:Giảng viên, người phụ trách bài thực hành hoặc labcoach

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
