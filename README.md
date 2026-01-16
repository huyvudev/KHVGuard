# 🛡️ KHVGuard - Vietnamese Prompt Injection Detection

**KHVGuard** là hệ thống phát hiện các câu lệnh độc hại (Prompt Injection/Jailbreak) dành riêng cho các mô hình ngôn ngữ lớn (LLM) tiếng Việt. Dự án sử dụng mô hình **PhoBERT** (VinAI) làm backbone để phân loại văn bản thành hai nhãn:
- **✅ An toàn (Benign)**
- **⚠️ Nguy hiểm (Attack)**

Ngoài ra, dự án cũng thực hiện so sánh hiệu năng với mô hình **DeBERTa (v3)** được huấn luyện lại trên cùng tập dữ liệu tiếng Việt để đánh giá tính hiệu quả.

## 📂 Cấu trúc dự án

```text
KHVGuard/
├── Datasets/              # Chứa dữ liệu huấn luyện
│   ├── train.json         # Dữ liệu train
│   └── valid.json         # Dữ liệu validation
├── TrainModel/            # Source code huấn luyện (Jupyter Notebook)
│   ├── PhoBERT.ipynb      # Notebook huấn luyện model chính (PhoBERT)
│   └── DeBERT_base_v3.ipynb # Notebook huấn luyện model so sánh (DeBERTa)
├── KHVGuard.py            # Ứng dụng giao diện web (Gradio App) để demo
├── requirements.txt       # Danh sách thư viện cần thiết
├── .gitignore             # File cấu hình bỏ qua venv và model nặng
└── README.md              # Hướng dẫn sử dụng
🚀 Cài đặt môi trường
Để chạy được dự án trên máy cá nhân, vui lòng làm theo các bước sau:

1. Clone dự án
Bash

git clone https://github.com/username-cua-ban/KHVGuard.git
cd KHVGuard
2. Tạo môi trường ảo (Virtual Environment)
Khuyến khích sử dụng Python 3.10 trở lên.

Trên Windows:

Bash

python -m venv venv
venv\Scripts\activate
(Nếu gặp lỗi SecurityError trên PowerShell, hãy chạy lệnh: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser)

Trên Linux/MacOS:

Bash

python3 -m venv venv
source venv/bin/activate
3. Cài đặt thư viện
Bash

pip install -r requirements.txt
📥 Tải Model (Trọng số)
Do file trọng số mô hình (.pth) có dung lượng lớn, chúng tôi không lưu trực tiếp trên GitHub. Bạn vui lòng tải về theo hướng dẫn sau:

Truy cập link Google Drive chính thức: TẢI MODEL TẠI ĐÂY

Tải file về máy.

Đổi tên file thành best_model.pth (nếu tên file tải về khác).

Copy file best_model.pth vào thư mục gốc của dự án (ngang hàng với file KHVGuard.py).

Cấu trúc sau khi copy đúng sẽ trông như sau:

Plaintext

KHVGuard/
├── ...
├── KHVGuard.py
└── best_model.pth  <-- File nằm ở đây
🖥️ Hướng dẫn sử dụng (Demo)
Sau khi đã cài đặt thư viện và tải model, bạn có thể khởi chạy giao diện demo bằng lệnh:

Bash

python KHVGuard.py
Chờ một chút để hệ thống tải PhoBERT. Khi thấy dòng chữ: Running on local URL: http://127.0.0.1:7860 Hãy mở trình duyệt và truy cập địa chỉ trên để sử dụng tool.

📊 Huấn luyện & Dữ liệu
Dự án bao gồm 2 phần thử nghiệm nằm trong thư mục TrainModel:

PhoBERT.ipynb:

Sử dụng vinai/phobert-base-v2.

Tối ưu hóa cho ngôn ngữ tiếng Việt.

Đây là mô hình được sử dụng trong ứng dụng demo (KHVGuard.py).

DeBERT_base_v3.ipynb:

Sử dụng microsoft/deberta-v3-base.

Được fine-tune lại trên bộ dữ liệu tiếng Việt (Datasets/) để làm cơ sở so sánh hiệu năng với PhoBERT.

⚠️ Lưu ý quan trọng
File best_model.pth và thư mục venv/ đã được thêm vào .gitignore để tránh đẩy lên Github (do giới hạn dung lượng và xung đột môi trường).

Nếu muốn train lại model, hãy đảm bảo đường dẫn tới Datasets/train.json trong notebook là chính xác.
