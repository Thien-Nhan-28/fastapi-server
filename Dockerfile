FROM python:3.11-slim

WORKDIR /app

COPY requirements_base.txt .
COPY requirements_tensorflow.txt .

RUN pip install --default-timeout=1000 -r requirements_base.txt
RUN pip install --default-timeout=1000 -r requirements_tensorflow.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
