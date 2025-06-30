#!/bin/bash

set -e

echo "📦 Checking Docker Compose installation..."
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install it: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker Compose is installed."

# Check if webcam device exists (Linux only)
if [ ! -e /dev/video0 ]; then
    echo "⚠️  No webcam found at /dev/video0. This is normal for macOS or Windows."
    echo "ℹ️  You can change camera input in main.py to use an IP or RTSP stream."
    read -p "Continue anyway? (y/n) " choice
    [[ "$choice" != "y" ]] && exit 1
fi

echo "🚀 Starting the app using Docker Compose..."
docker-compose up --build
