FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

# ENTRYPOINT untuk menjalankan training saat container dijalankan
ENTRYPOINT ["python", "modeling.py"]
