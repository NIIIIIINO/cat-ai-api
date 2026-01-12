FROM python:3.10-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY . .

# Cloud Run ใช้ PORT จาก env
ENV PORT=8080

# start API (predict)
CMD ["python", "main.py"]
