# Sử dụng image chính thức của Python 3.12 slim
FROM python:3.12-slim

# Cài đặt các gói hệ thống cần thiết cho scipy, opencv, paddlepaddle và các thư viện khác
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file requirements.txt và cài đặt các thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào image
COPY . .

# Lệnh mặc định khi chạy container
CMD ["python", "main.py"]