# FastAPI Server với Docker

## Build Docker Image

Chạy lệnh sau trong thư mục chứa `Dockerfile` để build image:

```bash
docker build -t fastapi-server .


Chạy container với tên fastapi, map cổng 8000 để truy cập API:

```bash
docker run -d -p 8000:8000 --name fastapi fastapi-server


Trên máy chạy Docker: http://localhost:8000
