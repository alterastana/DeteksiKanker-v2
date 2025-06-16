# Gunakan Python image ringan
FROM python:3.10-slim

# Install dependency sistem untuk build package berat
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Salin semua file proyek ke dalam image
COPY . .

# Install pip dan dependencies Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Jalankan app.py (pastikan app.run() diatur host=0.0.0.0)
CMD ["python", "app.py"]
