# FastAPI Server với Docker

## Hướng dẫn build và chạy Docker container

### 1. Build Docker Image

Chạy lệnh sau trong thư mục chứa `Dockerfile` để tạo image với tên `fastapi-server`:

```bash
docker build -t fastapi-server .

```

### 2. Run Docker Container

Mở CMD và chạy lệnh sau để khởi động server:

```bash
docker run -d -p 8000:8000 --name fastapi fastapi-server

```

### 3. Lưu ý
Phải mở docker desktop rồi mới run server
