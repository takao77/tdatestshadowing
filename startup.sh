#!/bin/bash

# Install ffmpeg from apt.txt
if [ -f "apt.txt" ]; then
    while read -r package; do
        apt-get -y install "$package"
    done < apt.txt
fi

# Ensure ffmpeg and ffprobe are in the PATH
export PATH=$PATH:/usr/bin

# Start Gunicorn server
gunicorn --bind=0.0.0.0:8000 app_s:app


