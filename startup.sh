#!/bin/bash

# Update package list and install ffmpeg and ffprobe
apt-get update
apt-get install -y ffmpeg

# Ensure ffmpeg and ffprobe are in the PATH
export PATH=$PATH:/usr/bin

# Verify installation (optional, can be removed after confirmation)
echo "ffmpeg version: $(ffmpeg -version)"
echo "ffprobe version: $(ffprobe -version)"

# Start Gunicorn server
gunicorn --bind=0.0.0.0:8000 app_s:app

