# 🧠 Real-Time Face Recognition System

> Version: **v1.1.1**  
> FPS: ~12–16 on CPU (buffalo_s)  
> Accuracy: 80% - 95%  

---

## 📌 Overview

This is a real-time face recognition system built with:

- **InsightFace** for face detection and embedding  
- **Cosine similarity** for matching against a local SQLite DB  
- **OpenCV** for webcam access  
- **Threading + queue** to increase frame freshness  
- **Debouncing** to avoid redundant DB logs  

---

## ✅ Features

- 🧠 Unified face detection & embedding (SCRFD + ArcFace)
- 🚀 Real-time cosine similarity matching
- 💾 SQLite logging (`name`, `confidence`, `timestamp`)
- 🔁 Threaded webcam frame capture (non-blocking)
- ⏱️ Time-based log debouncing to reduce noise
- 🧩 Modular, readable architecture
- 🐳 Docker & Docker Compose support

---

## 🔄 Version Comparison

| Feature                    | v1.0.0                                   | v1.1.0                                   | v1.1.1                                                  |
|----------------------------|------------------------------------------|------------------------------------------|---------------------------------------------------------|
| Detection                  | YOLOv8-face                              | InsightFace (SCRFD)                      | InsightFace (SCRFD)                                     |
| Embedding                  | ArcFace (buffalo_l / buffalo_s)         | ArcFace (buffalo_s)                      | ArcFace (buffalo_s)                                     |
| FPS                        | ~4–6 (buffalo_l), ~8–11 (buffalo_s)      | ~12–16 (buffalo_s)                       | ~12–16 (buffalo_s)                                      |
| Accuracy                   | <90%                                     | <90%                                     | **≥ 95%**                                               |
| Logging                    | SQLite                                   | SQLite                                   | SQLite + **debounce**                                   |
| Pipeline                   | 2-stage (YOLO → Embedder)               | Unified Detector + Embedder              | Unified + **Threaded Webcam + Debouncing**              |
| Frame Freshness            | ❌ Blocking                              | ✅ Improved                              | ✅ + **Thread-safe**                                     |
| Code Complexity            | Medium                                   | Low                                      | Slightly higher (threading + debounce)                 |

---

## 📊 Architecture (v1.1.1)

```mermaid
graph TD
    A[Webcam [OpenCV]<br/>→ Thread A] --> Q[Queue]
    Q --> B[Face Detection & Embedding<br/>[InsightFace - ArcFace]<br/>→ Main Thread]
    B --> C[DB Embedding]
    B --> D[Face Similarity [NumPy - Cosine]]
    D --> E[logs (name, timestamp, confidence)<br/>+ Debounce]
    E --> F[DB logging]
```

---

## 🚀 Quickstart

### 📦 Prerequisites

- Python 3.10+
- Webcam (integrated or USB)
- Docker (optional but recommended)

---

### ▶️ Run Locally (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
python main.py
```

---

### 🐳 Run with Docker

```bash
# Build the container
docker build -t face-recognition .

# Run the container
docker run --rm -it --device=/dev/video0 face-recognition
```

If using Docker Compose (with volume or env configs):

```bash
docker compose up --build
```

---

### 📁 Folder Structure

```bash
face-recognition/
│
├── config/            # App settings
├── core/              # Core logic (timing, FPS counter)
├── db/                # SQLite, logger, face DB
├── modules/           # Detection, embedding, matching
├── utils/             # Helpers (draw, debounce, etc.)
│
├── main.py            # Entry point
├── face_registration.py  # CLI to register new face
├── requirements.txt
├── requirements_local.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🧪 Benchmarks

| Model       | FPS     | Accuracy |
|-------------|---------|----------|
| buffalo_l   | ~4–6    | <90%     |
| buffalo_s   | ~12–16  | ≥ 95%    |

Tested on: **CPU (Intel Core i7 10th Gen)** — No GPU used.

---

## 🔒 Privacy Note

This app **does not store or transmit** any images or videos. Only face **embeddings and match logs** are stored locally for demo purposes.

---

## 🧼 Dev Tools (optional)

In `requirements_local.txt`:

- `black`: auto code formatter  
- `ruff`: fast Python linter  

To install:

```bash
pip install -r requirements_local.txt
```

---

## 📋 TODOs

- [ ] Add spoofing detection
- [ ] Add unit tests and CI

---

## 📜 License

MIT © 2025 Sokritha Yen