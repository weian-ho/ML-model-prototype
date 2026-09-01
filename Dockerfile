# 1. 使用轻量级官方 Python 3.11 镜像
FROM python:3.11-slim

# 2. 设置容器内的工作目录
WORKDIR /workspace

# 3. 复制依赖清单并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 复制应用代码、特征工程模块与打包好的模型
COPY app/ app/
COPY src/ src/
COPY model/ model/

# 5. 暴露 8000 端口
EXPOSE 8000

# 6. 容器启动时默认运行的命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]