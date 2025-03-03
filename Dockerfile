FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 暴露WebSocket端口
EXPOSE 8765

# 设置环境变量
ENV PYTHONPATH=/app

# 启动服务
CMD ["python3", "server.py"] 