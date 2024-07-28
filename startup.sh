#!/bin/bash

# Install ffmpeg from apt.txt
if [ -f "apt.txt" ]; then
    while read -r package; do
        apt-get -y install "$package"
    done < apt.txt
fi

# Start Gunicorn server
gunicorn --bind=0.0.0.0:8000 app_s:app


