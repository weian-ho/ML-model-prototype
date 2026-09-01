# 1. Use the official lightweight Python 3.11 slim base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /workspace

# 3. Copy dependency manifest and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code, feature engineering module, and serialized model artifact
COPY app/ app/
COPY src/ src/
COPY model/ model/

# 5. Expose port 8000
EXPOSE 8000

# 6. Default execution command when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]