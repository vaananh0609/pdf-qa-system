# PDF Q&A System (Flask)

Hệ thống hỏi đáp theo nội dung PDF: upload/tổ chức tài liệu, xem nội dung theo từng chunk, và đặt câu hỏi để hệ thống truy xuất các đoạn liên quan rồi sinh câu trả lời.

## Tính năng chính
- Upload & quản lý PDF (giao diện người dùng + trang admin).
- Xem PDF + metadata/chunks.
- Hỏi đáp theo 1 file hoặc toàn bộ file.
- Lưu lịch sử hỏi đáp (MongoDB).
- Hỗ trợ PDF dạng **text** và (tuỳ chọn) PDF **scanned** qua Vintern API (chạy trên Colab GPU).

## Công nghệ
- Backend: Flask
- Database: MongoDB (PyMongo)
- AI:
  - Google Gemini (google-generativeai)
  - Fallback Groq (tuỳ chọn)
  - Vintern embedding (tuỳ chọn, chạy Colab)
- Xử lý PDF: PyPDF2, PyMuPDF (fitz)

## Cấu hình (biến môi trường)
Tạo file `.env` (KHÔNG commit) dựa trên `.env.example`.

Các biến quan trọng:
- `SECRET_KEY`: secret cho Flask session.
- `ADMIN_USERNAME`, `ADMIN_PASSWORD`: đăng nhập trang `/admin`.
- `MONGODB_URI`: URI MongoDB.
- `GEMINI_API_KEYS` (khuyến nghị) **hoặc** `GEMINI_API_KEY_PRIMARY`/`GEMINI_API_KEY_SECONDARY`.
- `GROQ_API_KEY`: (optional) fallback.
- `VINTERN_API_URL`: (optional) URL Vintern API (ngrok) để xử lý scanned PDF.

## Chạy local (Windows)
> Yêu cầu: Python 3.10+ (khuyến nghị), MongoDB (local hoặc Atlas).

```powershell
cd "E:\Chuyên đề Công nghệ thông tin\pdf-qa-system"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Tạo .env từ .env.example và điền giá trị thật
# Sau đó chạy:
python app.py
```

Mặc định Flask sẽ chạy tại: `http://127.0.0.1:5000`
- Trang người dùng: `/`
- Admin login: `/admin`

## Chạy Vintern API trên Colab GPU (tuỳ chọn, cho scanned PDF)
1. Mở notebook `colab_vintern_server.ipynb` trên Google Colab.
2. **Clear outputs trước khi commit GitHub** để tránh lưu URL ngrok trong output.
3. Nhập `NGROK_AUTH_TOKEN` theo cách an toàn (không hardcode).
4. Chạy notebook để lấy `Public URL`.
5. Ở máy local, set `VINTERN_API_URL=<public_url>` trong `.env` rồi restart app.

## Gợi ý ảnh minh hoạ cho README
Nếu bạn muốn README “CV-ready”, bạn chụp giúp mình 3–5 ảnh (PNG/JPG) rồi mình sẽ chèn vào README:
1. Trang chủ (list PDF + ô đặt câu hỏi).
2. Kết quả hỏi đáp (answer + sources).
3. Trang admin dashboard.
4. Trang PDF viewer (hiển thị chunks).

Bạn có thể để ảnh vào thư mục `docs/screenshots/` rồi mình link vào README.

## Lưu ý bảo mật
- Không commit `.env`, token, API keys, file trong `uploads/`.
- Với notebook `.ipynb`: luôn **Clear All Outputs** trước khi commit.
- Nếu bạn đã từng lỡ commit key/token trước đó: hãy rotate/thu hồi key.

## License
Dự án phục vụ mục đích học tập/portfolio.
