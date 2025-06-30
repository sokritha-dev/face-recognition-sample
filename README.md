# 🧠 Real-Time Face Recognition System

> Version: **v1.1.0**  
> FPS: ~12–16 on CPU (buffalo_s)  
> Accuracy: ~≥ 90%  
> Powered by [InsightFace](https://github.com/deepinsight/insightface)

---

## 📌 Overview

This is a real-time face recognition system using **InsightFace** for both **face detection** and **embedding**, making it fast, accurate, and CPU-friendly.

It compares detected faces from the webcam against stored embeddings using **cosine similarity**, then logs recognized individuals with timestamp and confidence score into an SQLite database.

---

## ✅ Features

- 🧠 **Single-Pipeline Face Detection & Embedding** using `buffalo_s`
- 🔍 Real-time **face matching** with cosine similarity
- 🧾 **Logging** of recognized faces (`name`, `timestamp`, `confidence`)
- 🧰 Lightweight and modular structure
- ⚙️ Works on **CPU** (ideal for laptops or edge devices)
- 📈 ~2x performance improvement over V1.0

---

## 🧱 Architecture

### 🔄 Version Comparison

| Feature                    | v1.0.0                                       | v1.1.0                                       |
|----------------------------|----------------------------------------------|----------------------------------------------|
| Detection                  | YOLOv8-face                                  | InsightFace (SCRFD)                          |
| Embedding                  | ArcFace (buffalo_l)                          | ArcFace (buffalo_s)                          |
| FPS                        | ~4–6 (buffalo_l), ~8–11 (buffalo_s)          | ~12–16 (buffalo_s)                           |
| Accuracy                   | ~90%                                         | ~90%                                         |
| Logging                    | SQLite                                       | SQLite                                       |
| Pipeline                   | 2-stage: YOLO → Embedder                     | 1-stage: Unified Detector + Embedder         |
| Code Complexity            | Medium                                       | Low (simplified)                             |

### 📊 Updated Pipeline (v1.1.0)

```mermaid
graph TD
    A[Webcam [OpenCV]] --> B[Face Detection & Embedding<br/>[InsightFace - ArcFace]]
    B --> C[DB Embedding]
    B --> D[Face Similarity [NumPy - Cosine]]
    D --> E[logs (name, timestamp, confidence)]
    E --> F[DB logging]
