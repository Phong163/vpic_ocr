# Sử dụng image cơ bản hỗ trợ GPU
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Thiết lập biến môi trường để tối ưu hóa
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Cài đặt các gói hệ thống và Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép mã nguồn (không bao gồm weights, config, output vì sẽ mount)
COPY . .

# Thêm nhãn
LABEL description="Docker image for YOLO and OCR license plate detection"

# Lệnh mặc định
CMD ["python", "main.py"]