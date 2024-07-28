#!/bin/bash

# Update and install ffmpeg
apt-get update && apt-get install -y ffmpeg

# Check if ffmpeg and ffprobe are installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found"
    exit 1
fi

if ! command -v ffprobe &> /dev/null
then
    echo "ffprobe could not be found"
    exit 1
fi

# Add /usr/bin to PATH if not already present
if [[ ":$PATH:" != *":/usr/bin:"* ]]; then
    export PATH=$PATH:/usr/bin
fi

# Start the application using Gunicorn
exec gunicorn --bind=0.0.0.0:8000 app_s:app
