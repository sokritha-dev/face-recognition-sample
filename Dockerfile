# Use the official YOLOv8 base image
FROM ultralytics/ultralytics:latest

# Set working directory
WORKDIR /app

# Copy your source code
COPY . .

# Install additional Python libraries
RUN pip install --no-cache-dir \
    insightface \
    sqlite-utils \
    matplotlib \
    seaborn

# Optional: preload insightface model to avoid runtime download
RUN python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])"

# Run your app
CMD ["python", "main.py"]
